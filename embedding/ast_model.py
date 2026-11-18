#!/usr/bin/env python3
"""Train an AST language model on JSON datasets produced by wsd.py prepare."""
import argparse
import json
from pathlib import Path

import pgf
import torch
import torch.nn.functional as F
from tqdm import tqdm
from wsd import Node2Vec, build_taxonomy_graph, load_dataset

CLS, EOS, PAD = "[CLS]", "[EOS]", "[PAD]"

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)

class RotaryPositionalEncoding(torch.nn.Module):
    def __init__(self, dim, max_seq_len=1024):
        super().__init__()
        N = 10000
        inv_freq = 1. / (N ** (torch.arange(0, dim, 2).float() / dim))
        position = torch.arange(max_seq_len).float()
        inv_freq = torch.cat((inv_freq, inv_freq), dim=-1)
        sinusoid_inp = torch.outer(position, inv_freq)
        self.register_buffer("cos", sinusoid_inp.cos())
        self.register_buffer("sin", sinusoid_inp.sin())

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.size(2)
        if seq_len > self.cos.size(0):
            dim = self.cos.size(1)
            freq = 1. / (10000 ** (torch.arange(0, dim, 2, device=x.device).float() / dim))
            angles = torch.outer(torch.arange(seq_len, device=x.device).float(), torch.cat((freq, freq)))
            cos, sin = angles.cos(), angles.sin()
        else:
            cos, sin = self.cos[:seq_len], self.sin[:seq_len]
        cos = cos.view(1, 1, seq_len, -1)
        sin = sin.view(1, 1, seq_len, -1)
        return apply_rotary_pos_emb(x, cos, sin)

class SwiGLU(torch.nn.Module):
    def __init__(self, hidden_dim, intermediate_dim):
        super().__init__()
        self.gate = torch.nn.Linear(hidden_dim, intermediate_dim)
        self.up = torch.nn.Linear(hidden_dim, intermediate_dim)
        self.down = torch.nn.Linear(intermediate_dim, hidden_dim)
        self.act = torch.nn.SiLU()

    def forward(self, x):
        x = self.act(self.gate(x)) * self.up(x)
        x = self.down(x)
        return x

class GQA(torch.nn.Module):
    def __init__(self, hidden_dim, num_heads, num_kv_heads=None, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = hidden_dim // num_heads
        self.num_groups = num_heads // num_kv_heads
        self.dropout = dropout
        self.q_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = torch.nn.Linear(hidden_dim, self.num_kv_heads * self.head_dim)
        self.v_proj = torch.nn.Linear(hidden_dim, self.num_kv_heads * self.head_dim)
        self.out_proj = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, q, k, v, mask=None, rope=None):
        q_batch_size, q_seq_len, hidden_dim = q.shape
        k_batch_size, k_seq_len, hidden_dim = k.shape
        v_batch_size, v_seq_len, hidden_dim = v.shape

        # projection
        q = self.q_proj(q).view(q_batch_size, q_seq_len, -1, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(k_batch_size, k_seq_len, -1, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(v_batch_size, v_seq_len, -1, self.head_dim).transpose(1, 2)

        # apply rotary positional encoding
        if rope:
            q = rope(q)
            k = rope(k)

        # compute grouped query attention
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        output = F.scaled_dot_product_attention(q, k, v,
                                                attn_mask=mask,
                                                dropout_p=self.dropout if self.training else 0.0,
                                                enable_gqa=True)
        output = output.transpose(1, 2).reshape(q_batch_size, q_seq_len, hidden_dim).contiguous()
        output = self.out_proj(output)
        return output

class DecoderLayer(torch.nn.Module):
    def __init__(self, hidden_dim, num_heads, num_kv_heads, dropout=0.1):
        super().__init__()
        self.self_attn = GQA(hidden_dim, num_heads, num_kv_heads, dropout)
        self.mlp = SwiGLU(hidden_dim, 4 * hidden_dim)
        self.norm1 = torch.nn.RMSNorm(hidden_dim)
        self.norm2 = torch.nn.RMSNorm(hidden_dim)

    def forward(self, x, mask=None, rope=None):
        # self-attention sublayer
        out = self.norm1(x)
        out = self.self_attn(out, out, out, mask, rope)
        x = out + x
        # MLP sublayer
        out = self.norm2(x)
        out = self.mlp(out)
        return out + x

class ASTGenerationModel(torch.nn.Module):
    def __init__(self, vocab, max_seq_len, config, taxonomy=None):
        super().__init__()
        self.vocab, self.max_seq_len, self.config = vocab, max_seq_len, config
        self.corpus_non_taxonomy_only = config.get("corpus_non_taxonomy_only", False)
        dim = config["embedding_dim"]
        self.node2vec = None
        self.taxonomy_seeds = set()
        if taxonomy is not None:
            if Node2Vec is None:
                raise SystemExit("--semantics requires torch-geometric and pyg-lib or torch-cluster")
            try:
                self.node2vec = Node2Vec(
                    taxonomy.edge_index, embedding_dim=dim, num_nodes=len(taxonomy.indices),
                    walk_length=config["walk_length"], context_size=config["context_size"],
                    walks_per_node=config["walks_per_node"], num_negative_samples=1)
            except ImportError as error:
                raise SystemExit("Node2Vec requires pyg-lib or torch-cluster") from error
            self.embedding = self.node2vec.embedding
            self.taxonomy_seeds = set(taxonomy.edge_index.flatten().tolist())
        else:
            self.embedding = torch.nn.Embedding(len(vocab), dim)
        taxonomy_mask = torch.zeros(self.embedding.num_embeddings, dtype=torch.bool)
        if self.taxonomy_seeds:
            taxonomy_mask[list(self.taxonomy_seeds)] = True
        self.register_buffer("taxonomy_mask", taxonomy_mask, persistent=False)
        self.rope = RotaryPositionalEncoding(dim // config["num_heads"], max_seq_len)
        self.decoders = torch.nn.ModuleList([
            DecoderLayer(dim, config["num_heads"], config["num_kv_heads"], config["dropout"])
            for _ in range(config["num_layers"])
        ])
        self.norm = torch.nn.RMSNorm(dim)
        self.out = torch.nn.Linear(dim, len(vocab))

    def forward(self, ids):
        mask = torch.triu(torch.full((ids.shape[1], ids.shape[1]), float("-inf"),
                                     device=ids.device), diagonal=1)
        x = self.embedding(ids)
        if self.corpus_non_taxonomy_only:
            x = torch.where(self.taxonomy_mask[ids].unsqueeze(-1), x.detach(), x)
        for decoder in self.decoders:
            x = decoder(x, mask, self.rope)
        return self.out(self.norm(x))

    def taxonomy_loss(self, batch):
        seeds = sorted(set(batch.flatten().tolist()) & self.taxonomy_seeds)
        if not seeds:
            return self.embedding.weight.new_zeros(())
        positive, negative = self.node2vec.sample(torch.tensor(seeds))
        device = self.embedding.weight.device
        return self.node2vec.loss(positive.to(device), negative.to(device))

    def save(self, path):
        state = {k: v for k, v in self.state_dict().items() if not k.startswith("node2vec.")}
        state["embedding.weight"] = self.embedding.weight[:len(self.vocab)].detach().clone()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"version": 1, "vocab": self.vocab, "max_seq_len": self.max_seq_len,
                    "config": self.config, "state_dict": state}, path)

    @classmethod
    def load(cls, path, device):
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        if checkpoint.get("version") != 1:
            raise ValueError("unsupported AST model checkpoint")
        model = cls(checkpoint["vocab"], checkpoint["max_seq_len"], checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        return model.to(device)


def encode_data(data, vocab):
    encoded = {}
    for split in ("training", "evaluation"):
        encoded[split] = []
        for row in data[split]:
            candidates = []
            for expr in [row["reference"], *row["negatives"]]:
                try:
                    candidates.append([vocab[CLS]] + [vocab[f] for f in pgf.exprFunctions(expr)] + [vocab[EOS]])
                except KeyError as error:
                    raise ValueError(f"function {error.args[0]!r} is missing from the model vocabulary") from error
            encoded[split].append(candidates)
    return encoded


def padded(sequences, pad, device):
    tensors = [torch.tensor(s, dtype=torch.long) for s in sequences]
    return torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=pad).to(device)


def sequence_scores(model, sequences, device):
    batch = padded(sequences, model.vocab[PAD], device)
    logits = model(batch[:, :-1])
    targets = batch[:, 1:]
    log_probs = F.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (log_probs * targets.ne(model.vocab[PAD])).sum(dim=1), targets.ne(model.vocab[PAD]).sum(dim=1)


def evaluate(model, rows, device, batch_size):
    """Reference token loss and accuracy of ranking reference against negatives."""
    model.eval()
    loss, tokens, correct, ranked = 0.0, 0, 0, 0
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            scores, lengths = sequence_scores(model, [c for row in batch for c in row], device)
            offset = 0
            for row in batch:
                loss -= scores[offset].item()
                tokens += lengths[offset].item()
                if len(row) > 1:
                    correct += int(scores[offset:offset + len(row)].argmax().item() == 0)
                    ranked += 1
                offset += len(row)
    return {"loss": loss / tokens if tokens else None,
            "accuracy": correct / ranked if ranked else None, "examples": len(rows), "ranked_examples": ranked}


def train(args):
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    grammar = pgf.readNGF(str(args.grammar))
    data = load_dataset(args.dataset)
    vocab = {token: i for i, token in enumerate([CLS, EOS, PAD] + list(grammar.functions))}
    rows = encode_data(data, vocab)
    if not rows["training"] or not rows["evaluation"]:
        raise SystemExit("dataset must contain both training and evaluation examples")
    max_len = max(len(c) - 1 for split in rows.values() for row in split for c in row)
    taxonomy = build_taxonomy_graph(grammar, vocab, args.semantics) if args.semantics else None
    config = {"embedding_dim": args.dimension, "num_layers": args.layers, "num_heads": args.heads,
              "num_kv_heads": args.kv_heads, "dropout": args.dropout,
              "walk_length": args.walk_length, "context_size": args.context_size,
              "walks_per_node": args.walks_per_node, "corpus_non_taxonomy_only": args.corpus_non_taxonomy_only}
    model = ASTGenerationModel(vocab, max_len, config, taxonomy).to(device)
    if args.corpus_non_taxonomy_only:
        other_parameters = [p for p in model.parameters() if p is not model.embedding.weight]
        optimizer = torch.optim.AdamW([
            {"params": other_parameters},
            {"params": [model.embedding.weight], "weight_decay": 0.0},
        ], lr=args.learning_rate)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(rows["training"])).tolist()
        graph_loss, batches = 0.0, 0
        for start in tqdm(range(0, len(order), args.batch_size), desc=f"epoch {epoch}"):
            sequences = [rows["training"][i][0] for i in order[start:start + args.batch_size]]
            batch = padded(sequences, vocab[PAD], device)
            logits = model(batch[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1), ignore_index=vocab[PAD])
            optimizer.zero_grad(set_to_none=True)
            taxonomy_grad = None
            if taxonomy is not None:
                semantic_loss = model.taxonomy_loss(batch)
                graph_loss += semantic_loss.item()
                if args.corpus_non_taxonomy_only and semantic_loss.requires_grad:
                    taxonomy_grad = torch.autograd.grad(semantic_loss, model.embedding.weight)[0]
                    taxonomy_grad = taxonomy_grad * model.taxonomy_mask.unsqueeze(-1)
                elif not args.corpus_non_taxonomy_only:
                    loss = loss + args.taxonomy_weight * semantic_loss
            loss.backward()
            if taxonomy_grad is not None:
                model.embedding.weight.grad.add_(taxonomy_grad, alpha=args.taxonomy_weight)
            torch.nn.utils.clip_grad_norm_(other_parameters if args.corpus_non_taxonomy_only else model.parameters(), 6.0)
            optimizer.step()
            batches += 1
        training = evaluate(model, rows["training"], device, args.batch_size)
        evaluation = evaluate(model, rows["evaluation"], device, args.batch_size)
        report = {"epoch": epoch, "training_loss": training["loss"], "evaluation_loss": evaluation["loss"],
                  "training_accuracy": training["accuracy"], "evaluation_accuracy": evaluation["accuracy"]}
        if taxonomy is not None:
            report["training_taxonomy_loss"] = graph_loss / batches
        print(json.dumps(report), flush=True)
        if evaluation["loss"] < best_loss:
            best_loss = evaluation["loss"]
            model.save(args.output)


def evaluate_checkpoint(args):
    device = torch.device(args.device)
    model = ASTGenerationModel.load(args.model, device)
    grammar = pgf.readNGF(str(args.grammar))
    if set(grammar.functions) - set(model.vocab):
        raise ValueError("grammar contains functions missing from the checkpoint")
    rows = encode_data(load_dataset(args.dataset), model.vocab)
    for split in ("training", "evaluation"):
        print(json.dumps({"split": split, **evaluate(model, rows[split], device, args.batch_size)}))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="train on a prepared dataset")
    training.add_argument("dataset", type=Path, help="JSON produced by wsd.py prepare")
    training.add_argument("--grammar", type=Path, default=Path("ParseEng.pgf"))
    training.add_argument("--semantics", type=Path, help="enable joint training with this semantics database")
    training.add_argument("--output", type=Path, default=Path("build/ast-model.pt"))
    training.add_argument("--epochs", type=int, default=70)
    training.add_argument("--batch-size", type=int, default=32)
    training.add_argument("--learning-rate", type=float, default=0.001)
    training.add_argument("--dimension", type=int, default=64)
    training.add_argument("--layers", type=int, default=8)
    training.add_argument("--heads", type=int, default=8)
    training.add_argument("--kv-heads", type=int, default=4)
    training.add_argument("--dropout", type=float, default=0.1)
    training.add_argument("--taxonomy-weight", type=float, default=1.0)
    training.add_argument("--corpus-non-taxonomy-only", action="store_true",
                          help="train taxonomy-connected function vectors only from the taxonomy")
    training.add_argument("--walk-length", type=int, default=20)
    training.add_argument("--context-size", type=int, default=15)
    training.add_argument("--walks-per-node", type=int, default=10)
    training.add_argument("--seed", type=int, default=13)
    training.set_defaults(run=train)
    evaluation = commands.add_parser("evaluate", help="evaluate a saved model")
    evaluation.add_argument("dataset", type=Path)
    evaluation.add_argument("model", type=Path)
    evaluation.add_argument("--grammar", type=Path, default=Path("ParseEng.pgf"))
    evaluation.add_argument("--batch-size", type=int, default=32)
    evaluation.set_defaults(run=evaluate_checkpoint)
    for command in (training, evaluation):
        command.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main():
    args = build_parser().parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    if args.command == "train":
        if args.corpus_non_taxonomy_only and not args.semantics:
            raise SystemExit("--corpus-non-taxonomy-only requires --semantics")
        if min(args.epochs, args.dimension, args.layers, args.heads, args.kv_heads, args.walk_length, args.context_size, args.walks_per_node) < 1:
            raise SystemExit("training dimensions, epochs and walk sizes must be positive")
        if args.dimension % args.heads or (args.dimension // args.heads) % 2 or args.heads % args.kv_heads:
            raise SystemExit("dimension must divide into even-sized heads, and heads must be divisible by kv-heads")
        if not 2 <= args.context_size <= args.walk_length:
            raise SystemExit("context size must be between 2 and walk length")
        if args.learning_rate <= 0 or args.taxonomy_weight < 0 or not 0 <= args.dropout < 1:
            raise SystemExit("invalid learning rate, taxonomy weight or dropout")
    args.run(args)


if __name__ == "__main__":
    main()
