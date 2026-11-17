#!/usr/bin/env python3
"""Train a word2vec-style disambiguator for GF abstract syntax trees.

The model assigns every GF function one vector and learns a shared context
projection matrix.  The score of a tree with functions ``w_1 ... w_m`` is

    sum_i dot(vector[w_i], mean(W vector[w_j] for j != i))

and the probability of a tree is the softmax of its score among the trees
produced by reparsing the same sentence.  Training uses cross entropy with
the checked ``abs:`` tree as class zero and sampled reparses as negatives.

Data preparation requires the GF Python binding; training and inference use
PyTorch.
"""

from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable):
        return iterable

import pgf

try:
    import torch
    from torch import Tensor, nn
    from torch.nn import functional as F
    try:
        from torch_geometric.nn import Node2Vec
    except ModuleNotFoundError:
        Node2Vec = None
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None
    Tensor = object
    nn = None
    F = None
    Node2Vec = None


UNKNOWN = "<UNK>"


@dataclass
class TaxonomyGraph:
    indices: dict[str,int]
    edge_index: Tensor


def load_wordnet(grammar: object, semantics: Path | None) -> object:
    """Load gf-wordnet against the grammar and semantics database in use."""
    if not semantics.exists():
        raise FileNotFoundError(f"WordNet semantics database not found: {semantics}")

    # gf-wordnet normally imports a generated Parse module.  Supplying this
    # lightweight module lets it use the already loaded NGF without modifying
    # the installed package or reading examples.txt.
    def parse_attribute(name: str) -> object:
        try:
            function_type = grammar.functionType(name)
        except KeyError as error:
            raise AttributeError(name) from error
        if not function_type.hypos:
            return pgf.ExprFun(name)

        def apply(*arguments: object) -> object:
            expression = pgf.ExprFun(name)
            for argument in arguments:
                expression = pgf.ExprApp(expression, argument)
            return expression

        return apply

    parse_module = types.ModuleType("Parse")
    parse_module.__pgf__ = grammar
    parse_module.__file__ = str(semantics.resolve().parent / "Parse.ngf")
    parse_module.__getattr__ = parse_attribute
    data_module = types.ModuleType("wordnet.data")
    data_module.__path__ = []
    data_module.Parse = parse_module
    for name in list(sys.modules):
        if name == "wordnet" or name.startswith("wordnet."):
            del sys.modules[name]
    sys.modules["Parse"] = parse_module
    sys.modules["wordnet.data"] = data_module
    sys.modules["wordnet.data.Parse"] = parse_module
    return importlib.import_module("wordnet")


def build_taxonomy_graph(
    grammar: pgf.PGF, indices: dict[str,int], semantics: Path
) -> TaxonomyGraph:
    """Build the same WordNet/VerbNet graph used by embedding/ast_model.py."""
    require_torch()
    if Node2Vec is None:
        raise SystemExit(
            "taxonomy training requires torch-geometric and a Node2Vec random-walk "
            "backend; install torch-geometric and pyg-lib or torch-cluster"
        )
    taxonomy_indices = dict(indices)
    wordnet = load_wordnet(grammar, semantics)
    edges: set[tuple[int, int]] = set()

    def get_id(w):
        return taxonomy_indices.setdefault(w, len(taxonomy_indices))
    def add_edge(source: str, target: str) -> None:
        edges.add((get_id(source), get_id(target)))

    synsets = wordnet.all_synsets()
    for synset in tqdm(synsets, "build graph"):
        relations = (
            synset.hypernyms()
            + synset.instance_hypernyms()
            + synset.hyponyms()
            + synset.instance_hyponyms()
            + synset.member_holonyms()
            + synset.substance_holonyms()
            + synset.part_holonyms()
            + synset.member_meronyms()
            + synset.substance_meronyms()
            + synset.part_meronyms()
        )
        for relation in relations:
            add_edge(synset.synsetOffset, relation.synsetOffset)
        for lexeme in synset.lexemes():
            lexical_relations = (
                lexeme.antonyms()
                + lexeme.participle()
                + lexeme.alsosee()
                + lexeme.derived()
                + lexeme.male()
                + lexeme.female()
            )
            for relation in lexical_relations:
                add_edge(lexeme.lex_fun, relation.lex_fun)
            for domain_id in lexeme.domain_ids:
                domain = f"D{domain_id}"
                add_edge(lexeme.lex_fun, domain)
                add_edge(domain, lexeme.lex_fun)
            for frame_id in lexeme.frame_ids:
                frame = f"F{frame_id}"
                add_edge(lexeme.lex_fun, frame)
                add_edge(frame, lexeme.lex_fun)
            if relations or lexical_relations or lexeme.domain_ids or lexeme.frame_ids:
                add_edge(synset.synsetOffset, lexeme.lex_fun)

    for domain in wordnet.all_domains():
        if domain.parent:
            child, parent = f"D{domain.id}", f"D{domain.parent}"
            add_edge(child, parent)
            add_edge(parent, child)
    for frame in wordnet.verb_frames():
        frame_node, class_node = f"F{frame.id}", f"C{frame.class_id}"
        add_edge(frame_node, class_node)
        add_edge(class_node, frame_node)
    for verb_class in wordnet.verb_classes():
        if verb_class.super_id:
            child, parent = f"C{verb_class.id}", f"C{verb_class.super_id}"
            add_edge(child, parent)
            add_edge(parent, child)

    edge_index = torch.tensor(sorted(edges), dtype=torch.long).t().contiguous()
    return TaxonomyGraph(taxonomy_indices, edge_index)

def read_checked_examples(path: Path) -> Iterator[pgf.Expr]:
    """Yield only complete, checked ``abs:`` trees"""
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.startswith("abs:") or line.startswith("abs+"):
                yield pgf.readExpr(line[5:])

def reparse_reference(
    concrete: pgf.Concr,
    expression: pgf.Expr,
    count: int,
    reparse_limit: int,
    rng: random.Random,
) -> dict | None:
    """Linearize a reference and return its top reparse and random negatives."""
    if count < 1:
        return None
    linearization = concrete.linearize(expression)
    if not linearization:
        return None
    seen: set[pgf.Expr] = set()
    candidates: list[pgf.Expr] = []
    try:
        parses = concrete.parse(linearization, n=reparse_limit)
        for tree, _parser_probability in parses:
            candidate = tree
            if candidate in seen:
                continue
            seen.add(candidate)
            candidates.append(candidate)
    except (pgf.ParseError, TypeError):
        return None
    if not candidates:
        return None
    top = candidates[0]

    try:
        candidates.remove(expression)
    except ValueError:
        # The reference lies below the reparse limit.
        pass

    if count < len(candidates):
        negatives = rng.sample(candidates, count)
    else:
        negatives = candidates

    return {
        "linearization": linearization,
        "top": str(top),
        "reference": str(expression),
        "negatives": [str(expr) for expr in negatives]
    }


def prepare_dataset(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    grammar = pgf.readNGF(str(args.grammar))
    try:
        concrete = grammar.languages[args.language]
    except KeyError as error:
        available = ", ".join(sorted(grammar.languages))
        raise SystemExit(
            f"language {args.language!r} is not in {args.grammar}; available: {available}"
        ) from error

    rows = []
    for number, example in tqdm(enumerate(read_checked_examples(args.examples), 1)):
        reparse = reparse_reference(
            concrete,
            example,
            args.negative_samples,
            args.reparse_limit,
            rng,
        )
        if reparse:
            rows.append(reparse)
        if args.report_every and number % args.report_every == 0:
            print(json.dumps({"reparsed": number, "usable": len(rows)}))

    rng.shuffle(rows)
    evaluation_size = round(len(rows) * args.evaluation_fraction)
    if len(rows) > 1:
        evaluation_size = min(max(evaluation_size, 1), len(rows) - 1)
    evaluation = rows[:evaluation_size]
    training = rows[evaluation_size:]
    result = {
        "version": 2,
        "input": str(args.examples),
        "negative_samples": args.negative_samples,
        "reparse_limit": args.reparse_limit,
        "language": args.language,
        "seed": args.seed,
        "training": training,
        "evaluation": evaluation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as destination:
        json.dump(result, destination, ensure_ascii=False)
    print(
        json.dumps(
            {
                "training": len(training),
                "evaluation": len(evaluation),
                "output": str(args.output),
            }
        )
    )

def load_dataset(path: Path) -> dict:
    with path.open(encoding="utf-8") as source:
        data = json.load(source)
    if data.get("version") != 2:
        raise ValueError(
            f"unsupported dataset version in {path}; rerun the prepare command"
        )
    for split in ("training", "evaluation"):
        if not isinstance(data.get(split), list):
            raise ValueError(f"dataset has no {split!r} list")
        for row in data[split]:
            required = {"linearization", "top", "reference", "negatives"}
            missing = required.difference(row)
            if missing:
                raise ValueError(f"dataset row is missing: {', '.join(sorted(missing))}")
            row["top"] = pgf.readExpr(row["top"])
            row["reference"] = pgf.readExpr(row["reference"])
            row["negatives"] = [pgf.readExpr(expr) for expr in row["negatives"]]
    return data

def encode_rows(
    gr: pgf.PGF, rows: Iterable[dict],
    indices: dict[str, int],
    device: "torch.device"
) -> list[tuple[Tensor, float]]:
    unknown = indices[UNKNOWN]
    encoded = []
    for row in rows:
        candidates = []
        for expression in [row["reference"], *row["negatives"]]:
            tokens = [
                indices.get(function, unknown)
                for function in pgf.exprFunctions(expression)
                if len(gr.functionType(function).hypos) == 0
            ]
            candidates.append((torch.tensor(tokens, dtype=torch.long, device=device),gr.exprProbability(expression)))
        if len(candidates) > 1 and all(len(candidate) > 1 for candidate,_ in candidates):
            encoded.append(candidates)
    return encoded


def require_torch() -> None:
    if torch is None:
        raise SystemExit(
            "training requires PyTorch; install it with: python3 -m pip install torch"
        )


if torch is not None:
    class WSDModel(nn.Module):
        """One function embedding table plus a learned context projection."""

        def __init__(
            self,
            indices: dict[str,int],
            dimension: int,
            seed: int = 13,
            taxonomy: TaxonomyGraph | None = None,
            node2vec_config: dict | None = None,
        ):
            super().__init__()
            if len(indices) < 2:
                raise ValueError("the vocabulary must contain at least one function")
            self.indices = indices
            self.dimension = dimension
            self.node2vec_config = dict(node2vec_config or {})
            torch.manual_seed(seed)
            if taxonomy is not None:
                if Node2Vec is None:
                    raise SystemExit(
                        "loading this model requires torch-geometric and a Node2Vec "
                        "random-walk backend"
                    )
                self.taxonomy = taxonomy
                try:
                    self.node2vec = Node2Vec(
                        taxonomy.edge_index,
                        embedding_dim=dimension,
                        num_nodes=len(taxonomy.indices),
                        sparse=False,
                        **self.node2vec_config,
                    )
                except ImportError as error:
                    raise SystemExit(
                        "Node2Vec requires either pyg-lib or torch-cluster"
                    ) from error
                self.embedding = self.node2vec.embedding
            else:
                self.taxonomy = None
                self.embedding = nn.Embedding(
                    len(self.indices), dimension
                )
            nn.init.normal_(self.embedding.weight, std=dimension**-0.5)
            

        def tree_score(self, tokens: Tensor) -> Tensor:
            """Sum ``v_i dot W v_j`` over all distinct function positions."""
            if tokens.numel() < 2:
                return self.embedding.weight.new_zeros(())
            context = F.normalize(self.embedding(tokens))
            all_pairs = torch.dot(context.sum(dim=0), context.sum(dim=0))
            self_pairs = (context * context).sum()
            return (all_pairs - self_pairs) / (tokens.numel() * (tokens.numel() - 1))

        def forward(
            self, candidates: list[tuple[Tensor,float]]
        ) -> Tensor:
            """Return learned plus context-free log scores for candidate trees."""
            return torch.stack([self.tree_score(tokens)-context_free_cost for tokens,context_free_cost in candidates])

        def probabilities(
            self, candidates: list[tuple[Tensor,float]]
        ) -> Tensor:
            return F.softmax(self(candidates), dim=0)

        def taxonomy_loss(self, candidate_batches: Sequence[Sequence[Tensor]]) -> Tensor:
            """Sample Node2Vec walks rooted at lexical functions in this batch."""
            seed_ids = {
                int(token)
                for candidates in candidate_batches
                for tokens, _ in candidates
                for token in tokens.detach().cpu().tolist()
            }
            return self.taxonomy_node_loss(seed_ids)

        def taxonomy_node_loss(self, seed_ids: Iterable[int]) -> Tensor:
            """Sample Node2Vec walks rooted at the supplied taxonomy nodes."""
            if self.node2vec is None:
                return self.embedding.weight.new_zeros(())
            seed_ids = sorted(set(seed_ids))
            if not seed_ids:
                return self.embedding.weight.new_zeros(())
            seeds = torch.tensor(seed_ids, dtype=torch.long)
            positive_walks, negative_walks = self.node2vec.sample(seeds)
            device = self.embedding.weight.device
            return self.node2vec.loss(
                positive_walks.to(device), negative_walks.to(device)
            )

        def save(self, path: Path) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "version": 3,
                    "indices": self.indices,
                    "dimension": self.dimension,
                    "node2vec_config": self.node2vec_config,
                    #"taxonomy_indices": self.taxonomy.indices,
                    #"edge_index": (
                    #    self.taxonomy.edge_index.detach().cpu()
                    #    if self.taxonomy.edge_index is not None
                    #    else None
                    #),
                    "state_dict": self.state_dict(),
                },
                path,
            )

        def export_embeddings(self, directory: Path) -> None:
            """Export stored vectors and their matrix-projected contexts."""
            directory.mkdir(parents=True, exist_ok=True)
            ids = torch.tensor(
                list(self.indices.values()),
                dtype=torch.long,
                device=self.embedding.weight.device,
            )
            function_vectors = self.embedding(ids)
            target = function_vectors.detach().cpu().tolist()
            files = {
                "meta.tsv": ([function] for function in self.indices),
                "vectors.tsv": target,
            }
            for filename, rows in files.items():
                with (directory / filename).open("w", encoding="utf-8") as output:
                    for row in rows:
                        output.write("\t".join(map(str, row)) + "\n")

        @classmethod
        def load(
            cls, path: Path, device: "torch.device"
        ) -> "WSDModel":
            try:
                checkpoint = torch.load(path, map_location=device, weights_only=True)
            except TypeError:  # PyTorch before the weights_only argument.
                checkpoint = torch.load(path, map_location=device)
            if checkpoint.get("version") != 3:
                raise ValueError(f"unsupported model version in {path}")
            taxonomy = None
            if checkpoint["edge_index"] is not None:
                taxonomy = TaxonomyGraph(
                    checkpoint["taxonomy_indices"],
                    checkpoint["edge_index"].cpu(),
                )
            model = cls(
                checkpoint["indices"],
                checkpoint["dimension"],
                taxonomy=taxonomy,
                node2vec_config=checkpoint["node2vec_config"],
            )
            model.load_state_dict(checkpoint["state_dict"])
            return model.to(device)
else:
    class WSDModel:  # Allows the dependency-free `prepare` command to run.
        def __init__(self, *args: object, **kwargs: object):
            require_torch()

def evaluate(
    model: WSDModel, rows: Sequence[list[tuple[Tensor,float]]]
) -> tuple[float, float]:
    if not rows:
        return float("nan"), float("nan")
    loss = 0.0
    correct = 0
    with torch.no_grad():
        for candidates in rows:
            scores = model(candidates)
            label = scores.new_zeros(1, dtype=torch.long)
            loss += F.cross_entropy(scores.unsqueeze(0), label).item()
            correct += int(scores.argmax().item() == 0)
    return loss / len(rows), correct / len(rows)

def print_first_parse_baseline(rows: Sequence[dict]) -> None:
    correct = sum(
        row["top"] == row["reference"]
        for row in rows
    )
    evaluated = len(rows)
    accuracy = correct / evaluated if evaluated else float("nan")
    print(
        json.dumps(
            {
                "baseline": "first_parse",
                "evaluation_examples": evaluated,
                "evaluation_accuracy": accuracy,
            }
        )
    )


def train_model(args: argparse.Namespace) -> None:
    require_torch()
    if args.taxonomy_only and args.semantics is None:
        raise SystemExit("--taxonomy-only requires --semantics")
    torch.manual_seed(args.seed)
    data = load_dataset(args.dataset)
    print_first_parse_baseline(data["evaluation"])
    grammar = pgf.readNGF(str(args.grammar))
    vocabulary = [UNKNOWN] + [fun for fun in grammar.functions if len(grammar.functionType(fun).hypos) == 0]
    indices = {name: index for index, name in enumerate(vocabulary)}
    if args.semantics != None:
        taxonomy = build_taxonomy_graph(grammar, indices, args.semantics)
    else:
        taxonomy = None
    device = torch.device(args.device)
    training = encode_rows(grammar, data["training"], indices, device)
    evaluation_rows = encode_rows(grammar, data["evaluation"], indices, device)
    if not training:
        raise SystemExit("the dataset contains no usable training examples")
    node2vec_config = {
        "walk_length": args.walk_length,
        "context_size": args.context_size,
        "walks_per_node": args.walks_per_node,
        "p": args.return_parameter,
        "q": args.inout_parameter,
        "num_negative_samples": args.taxonomy_negative_samples,
    }
    model = WSDModel(
        indices,
        args.dimension,
        args.seed,
        taxonomy,
        node2vec_config,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    rng = random.Random(args.seed)

    stats = {
                "training": len(training),
                "evaluation": len(evaluation_rows),
                "functions": len(vocabulary),
                "device": str(device),
                "taxonomy_only": args.taxonomy_only,
            }
    if taxonomy != None:
        stats.update({ "taxonomy_nodes": len(taxonomy.indices),
                       "taxonomy_edges": taxonomy.edge_index.shape[1]
                     })

    print(json.dumps(stats))

    for epoch in range(1, args.epochs + 1):
        rng.shuffle(training)
        epoch_wsd_loss = 0.0
        epoch_taxonomy_loss = 0.0
        epoch_total_loss = 0.0
        epoch_correct = 0
        batches = 0

        if args.taxonomy_only:
            starts = range(0, len(training), args.batch_size)
            for start in tqdm(starts, f"epoch {epoch}"):
                batch = training[start : start + args.batch_size]
                optimizer.zero_grad(set_to_none=True)
                taxonomy_loss = model.taxonomy_loss(batch)
                loss = args.taxonomy_weight * taxonomy_loss
                loss.backward()
                if args.clip_grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()

                epoch_taxonomy_loss += taxonomy_loss.item()
                epoch_total_loss += loss.item()
                batches += 1
        else:
            starts = range(0, len(training), args.batch_size)
            for start in tqdm(starts, f"epoch {epoch}"):
                batch = training[start : start + args.batch_size]
                optimizer.zero_grad(set_to_none=True)
                losses = []
                for candidates in batch:
                    scores = model(candidates)
                    loss = F.cross_entropy(
                        scores.unsqueeze(0), scores.new_zeros(1, dtype=torch.long)
                    )
                    losses.append(loss)
                    epoch_correct += int(scores.argmax().item() == 0)
                wsd_loss = torch.stack(losses).mean()

                loss = wsd_loss
                if taxonomy != None:
                    taxonomy_loss = model.taxonomy_loss(batch)
                    loss += args.taxonomy_weight * taxonomy_loss

                loss.backward()
                if args.clip_grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()

                epoch_wsd_loss += wsd_loss.item()
                if taxonomy != None:
                    epoch_taxonomy_loss += taxonomy_loss.item()
                epoch_total_loss += loss.item()
                batches += 1

        evaluation_loss, evaluation_accuracy = evaluate(model, evaluation_rows)
        stats = {
                    "epoch": epoch,
                    "training_loss": epoch_total_loss / batches,
                    "evaluation_accuracy": evaluation_accuracy,
                }
        if args.taxonomy_only:
            training_wsd_loss, training_accuracy = evaluate(model, training)
            stats.update({"training_accuracy": training_accuracy})
        else:
            training_wsd_loss = epoch_wsd_loss / batches
            training_accuracy = epoch_correct / len(training)
            stats.update({"evaluation_loss": evaluation_loss,
                          "training_wsd_loss": training_wsd_loss,
                          "training_accuracy": training_accuracy})
        if taxonomy != None:
            stats.update({"taxonomy_loss": epoch_taxonomy_loss / batches})
        print(json.dumps(stats))

    model.save(args.output)
    embeddings_directory = args.embeddings_directory or args.output.parent
    model.export_embeddings(embeddings_directory)
    print(
        json.dumps(
            {
                "model": str(args.output),
                "meta": str(embeddings_directory / "meta.tsv"),
                "vectors": str(embeddings_directory / "vectors.tsv"),
            }
        )
    )


def evaluate_checkpoint(args: argparse.Namespace) -> None:
    require_torch()
    data = load_dataset(args.dataset)
    device = torch.device(args.device)
    grammar = pgf.readNGF(str(args.grammar))
    model = WSDModel.load(args.model, device)
    model.eval()
    for split in ("training", "evaluation"):
        rows = encode_rows(grammar, data[split], model.indices, device)
        loss, accuracy = evaluate(model, rows)
        print(
            json.dumps(
                {
                    "split": split,
                    "examples": len(rows),
                    "loss": loss,
                    "accuracy": accuracy,
                }
            )
        )


def disambiguate(args: argparse.Namespace) -> None:
    """Reparse one sentence and rank its trees with a saved model."""
    require_torch()
    device = torch.device(args.device)
    grammar = pgf.readNGF(str(args.grammar))
    model = WSDModel.load(args.model, device)
    model.eval()
    try:
        concrete = grammar.languages[args.language]
    except KeyError as error:
        available = ", ".join(sorted(grammar.languages))
        raise SystemExit(
            f"language {args.language!r} is not in {args.grammar}; available: {available}"
        ) from error

    expressions = []
    seen = set()
    for tree, _parser_probability in concrete.parse(args.sentence, n=args.candidates):
        expression = str(tree)
        if expression not in seen:
            seen.add(expression)
            expressions.append(expression)
    if not expressions:
        raise SystemExit("the grammar did not parse the sentence")
    candidates, candidate_expressions = encode_rows(grammar,
        [
            {
                "reference": expressions[0],
                "negatives": expressions[1:],
            }
        ],
        model.indices,
        device
    )
    if not encoded:
        raise SystemExit("the grammar produced fewer than two usable candidate trees")
    with torch.no_grad():
        probabilities = model.probabilities(
            candidates, candidate_expressions
        ).cpu().tolist()
    ranked = sorted(zip(probabilities, expressions), reverse=True)
    for probability, expression in ranked:
        print(json.dumps({"probability": probability, "expression": expression}, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="reparse abs: examples and sample negatives")
    prepare.add_argument("--examples", type=Path, default=Path("examples.txt"))
    prepare.add_argument("--grammar", type=Path, default=Path("ParseEng.pgf"))
    prepare.add_argument("--language", default="ParseEng")
    prepare.add_argument("--output", type=Path, default=Path("build/wsd-examples.json"))
    prepare.add_argument("--negative-samples", type=int, default=5)
    prepare.add_argument("--reparse-limit", type=int, default=100)
    prepare.add_argument(
        "--evaluation-fraction",
        type=float,
        default=0.2,
        help="fraction reserved for evaluation (default: 0.2, an 80/20 split)",
    )
    prepare.add_argument("--seed", type=int, default=13)
    prepare.add_argument("--report-every", type=int, default=100)
    prepare.set_defaults(run=prepare_dataset)

    train = commands.add_parser("train", help="train embeddings and projection matrix")
    train.add_argument("dataset", type=Path)
    train.add_argument("--grammar", type=Path, default=Path("ParseEng.pgf"))
    train.add_argument(
        "--semantics",
        type=Path,
        default=None,
        help="WordNet semantics database used to construct the taxonomy graph",
    )
    train.add_argument("--output", type=Path, default=Path("build/wsd-model.pt"))
    train.add_argument(
        "--embeddings-directory",
        type=Path,
        help="directory for meta.tsv, target.tsv and context.tsv (default: model directory)",
    )
    train.add_argument("--dimension", type=int, default=64)
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--weight-decay", type=float, default=1e-5)
    train.add_argument("--clip-grad-norm", type=float, default=5.0)
    train.add_argument("--minimum-count", type=int, default=1)
    train.add_argument("--taxonomy-weight", type=float, default=1.0)
    train.add_argument(
        "--taxonomy-only",
        "--no-corpus-training",
        action="store_true",
        help=(
            "train only with taxonomy random walks seeded from the training "
            "corpus; use corpus labels only for metric reporting"
        ),
    )
    train.add_argument("--walk-length", type=int, default=20)
    train.add_argument("--context-size", type=int, default=15)
    train.add_argument("--walks-per-node", type=int, default=10)
    train.add_argument("--return-parameter", type=float, default=1.0, help="Node2Vec p")
    train.add_argument("--inout-parameter", type=float, default=1.0, help="Node2Vec q")
    train.add_argument("--taxonomy-negative-samples", type=int, default=1)
    train.add_argument("--seed", type=int, default=13)
    train.add_argument(
        "--device",
        default="cuda" if torch is not None and torch.cuda.is_available() else "cpu",
    )
    train.set_defaults(run=train_model)

    evaluate_parser = commands.add_parser("evaluate", help="evaluate a saved model")
    evaluate_parser.add_argument("dataset", type=Path)
    evaluate_parser.add_argument("model", type=Path)
    evaluate_parser.add_argument("--grammar", type=Path, default=Path("ParseEng.pgf"))
    evaluate_parser.add_argument(
        "--device",
        default="cuda" if torch is not None and torch.cuda.is_available() else "cpu",
    )
    evaluate_parser.set_defaults(run=evaluate_checkpoint)

    disambiguate_parser = commands.add_parser(
        "disambiguate", help="rank reparses of a sentence with a saved model"
    )
    disambiguate_parser.add_argument("model", type=Path)
    disambiguate_parser.add_argument("sentence")
    disambiguate_parser.add_argument("--grammar", type=Path, default=Path("ParseEng.pgf"))
    disambiguate_parser.add_argument("--language", default="ParseEng")
    disambiguate_parser.add_argument("--candidates", type=int, default=100)
    disambiguate_parser.add_argument(
        "--device",
        default="cuda" if torch is not None and torch.cuda.is_available() else "cpu",
    )
    disambiguate_parser.set_defaults(run=disambiguate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if getattr(args, "negative_samples", 1) < 1:
        raise SystemExit("--negative-samples must be positive")
    if hasattr(args, "evaluation_fraction") and not 0.0 < args.evaluation_fraction < 1.0:
        raise SystemExit("--evaluation-fraction must be between 0 and 1")
    if getattr(args, "batch_size", 1) < 1:
        raise SystemExit("--batch-size must be positive")
    if hasattr(args, "context_size") and args.context_size > args.walk_length:
        raise SystemExit("--context-size cannot exceed --walk-length")
    for option in ("walk_length", "context_size", "walks_per_node", "taxonomy_negative_samples"):
        if hasattr(args, option) and getattr(args, option) < 1:
            raise SystemExit(f"--{option.replace('_', '-')} must be positive")
    for option in ("return_parameter", "inout_parameter"):
        if hasattr(args, option) and getattr(args, option) <= 0:
            raise SystemExit(f"--{option.replace('_', '-')} must be greater than zero")
    if hasattr(args, "taxonomy_weight") and args.taxonomy_weight < 0:
        raise SystemExit("--taxonomy-weight cannot be negative")
    args.run(args)


if __name__ == "__main__":
    main()
