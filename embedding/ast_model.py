import os
import sys
import pgf
import wordnet
import torch
import torch.nn.functional as F
import torch_geometric.nn
from tqdm import tqdm

def exprFunctions(e):
    funs = []
    def iterate(e):
        if type(e) == pgf.ExprApp:
            iterate(e.fun)
            iterate(e.arg)
        elif type(e) == pgf.ExprFun:
            funs.append(e.name)
    iterate(e)
    return funs

class ASTDataset(torch.utils.data.Dataset):
    def __init__(self, fpath):
        dataset = []
        self.graph   = []
        self.vocab = {"[CLS]": 0, "[EOS]": 1, "[PAD]": 2}
        for fun in wordnet.grammar.functions:
            self.vocab[fun] = len(self.vocab)

        encoded = []
        self.max_seq_len = 0
        with open(fpath) as f:
            for line in tqdm(f, desc="Encoding the examples"):
                if line.startswith("abs:"):
                    sent = ["[CLS]"]+exprFunctions(pgf.readExpr(line[5:]))
                    sent_ids = []
                    for fun in sent:
                        id = self.vocab[fun]
                        sent_ids.append(id)
                    if self.max_seq_len < len(sent):
                        self.max_seq_len = len(sent)
                    dataset.append(sent)
                    encoded.append(sent_ids)

        self.max_seq_len += 1
        for sent_ids in encoded:
            sent_ids.append(1)
            while len(sent_ids) < self.max_seq_len:
                sent_ids.append(2)

        self.training_data = encoded[:-32]
        self.testing_data  = encoded[-32:]

        def get_id(w):
            id = self.vocab.get(w)
            if id == None:
                id = len(self.vocab)
                self.vocab[w] = id
            return id

        def add_edge(w1,w2):
            self.graph.append([get_id(w1),get_id(w2)])

        self.lex_funs = {}
        self.lex_fun_ids = set()
        synsets = wordnet.all_synsets()
        for i in tqdm(range(len(synsets)), desc="Building the semantic graph"):
            synset = synsets.pop()
            countS = 0
            for rel in (synset.hypernyms()+
                        synset.instance_hypernyms()+
                        synset.hyponyms()+
                        synset.instance_hyponyms()+
                        synset.member_holonyms()+
                        synset.substance_holonyms()+
                        synset.part_holonyms()+
                        synset.member_meronyms()+
                        synset.substance_meronyms()+
                        synset.part_meronyms()):
                add_edge(get_id(synset.synsetOffset), get_id(rel.synsetOffset))
                countS += 1
            for lexeme in synset.lexemes():
                count = countS
                for rel in (lexeme.antonyms()+
                            lexeme.participle()+
                            lexeme.alsosee()+
                            lexeme.derived()+
                            lexeme.male()+
                            lexeme.female()):
                    add_edge(lexeme.lex_fun, rel.lex_fun)
                    count += 1
                for domain_id in lexeme.domain_ids:
                    add_edge(lexeme.lex_fun, "D"+str(domain_id))
                    add_edge("D"+str(domain_id), lexeme.lex_fun)
                    count += 1
                for frame_id in lexeme.frame_ids:
                    add_edge(lexeme.lex_fun, "F"+str(frame_id))
                    add_edge("F"+str(frame_id), lexeme.lex_fun)
                    count += 1
                if count:
                    add_edge(synset.synsetOffset, lexeme.lex_fun)
                    self.lex_funs[lexeme.lex_fun] = synset.gloss
                    self.lex_fun_ids.add(get_id(lexeme.lex_fun))
        for domain in wordnet.all_domains():
            if domain.parent:
                add_edge("D"+str(domain.id), "D"+str(domain.parent))
                add_edge("D"+str(domain.parent), "D"+str(domain.id))
        for frame in wordnet.verb_frames():
            add_edge("F"+str(frame.id), "C"+str(frame.class_id))
            add_edge("C"+str(frame.class_id), "F"+str(frame.id))
        for cls in wordnet.verb_classes():
            if cls.super_id:
                add_edge("C"+str(cls.id), "C"+str(cls.super_id))
                add_edge("C"+str(cls.super_id), "C"+str(cls.id))

    def __len__(self):
        return len(self.training_data)

    def __getitem__(self, idx):
        return self.training_data[idx]


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
            seq_len = x.size(1)
        cos = self.cos[:seq_len].view(1, seq_len, 1, -1)
        sin = self.sin[:seq_len].view(1, seq_len, 1, -1)
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
        self.k_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = torch.nn.Linear(hidden_dim, hidden_dim)
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
                                                dropout_p=self.dropout,
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

class ASTGenerationModel(torch_geometric.nn.Node2Vec):
    def __init__(self, dataset, embedding_dim,
                 num_layers, num_heads, num_kv_heads,
                 walk_length,
                 context_size, walks_per_node,
                 p, q,
                 num_negative_samples, dropout=0.1):
        super().__init__(torch.tensor(dataset.graph).to(device).T,
                         embedding_dim=embedding_dim,
                         walk_length=walk_length,
                         context_size=context_size,
                         p=p,q=q,
                         num_negative_samples=num_negative_samples,
                         num_nodes=len(dataset.vocab))
        self.dataset = dataset
        self.keys  = list(dataset.vocab.keys())
        self.rope = RotaryPositionalEncoding(embedding_dim // num_heads, dataset.max_seq_len)
        self.decoders = torch.nn.ModuleList([
            DecoderLayer(embedding_dim, num_heads, num_kv_heads, dropout)
            for _ in range(num_layers)
        ])
        self.norm = torch.nn.RMSNorm(embedding_dim)
        self.out = torch.nn.Linear(embedding_dim, len(dataset.vocab))

    def loader(self, **kwargs):
        return torch.utils.data.DataLoader(self.dataset, collate_fn=self.sample, **kwargs)

    def sample(self,batch):
        batch = torch.tensor(batch)
        x = batch[:,:-1]
        y = batch[:,1: ]
        ids = [id.item() for id in torch.unique(batch) if id.item() in self.dataset.lex_fun_ids]
        pos_rw, neg_rw = super().sample(ids)
        return x, y, pos_rw, neg_rw

    def forward(self, ids, mask=None):
        x = self.embedding(ids)
        for decoder in self.decoders:
            x = decoder(x, mask, self.rope)
        x = self.norm(x)
        return self.out(x)

    def encode(self, funs):
        fun_ids = []
        for fun in funs:
            fun_ids.append(self.dataset.vocab[fun])
        return fun_ids

    def decode(self, fun_ids):
        funs = []
        for fun_id in fun_ids:
            funs.append(self.keys[fun_id])
        return funs

    def token_to_id(self,tok):
        return self.dataset.vocab[tok]

    def id_to_token(self,fun_id):
        return self.keys[fun_id]
    
    def prune(self):
        del self.dataset.encoded
        del self.dataset.lex_funs
        del self.dataset.lex_fun_ids

        indices = []
        vocab   = {}
        for tok in ["[CLS]", "[EOS]", "[PAD]"]+wordnet.grammar.functions:
            indices.append(self.dataset.vocab[tok])
            vocab[tok] = len(vocab)
        self.embedding.weight = torch.nn.Parameter(self.embedding.weight[indices])
        self.out.weight = torch.nn.Parameter(self.out.weight[indices])
        self.out.bias   = torch.nn.Parameter(self.out.bias[indices])
        self.dataset.graph = [[0,0]]
        self.dataset.vocab = vocab


def create_causal_mask(seq_len, device):
    """Create a causal mask for autoregressive attention."""
    mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device), diagonal=1)
    return mask

# Training configuration
model_config = {
    "num_layers": 8,
    "num_heads": 8,
    "num_kv_heads": 4,
    "embedding_dim": 64,
    "dropout": 0.1,
    "walk_length": 20,
    "context_size": 15,
    "walks_per_node": 10,
    "p": 1.0,
    "q": 1.0,
    "num_negative_samples": 1
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32

# Training loop
if sys.argv[1] == "train":

    N_EPOCHS = 70
    LR = 0.05
    WARMUP_STEPS = 2000
    CLIP_NORM = 6.0

    dataset = ASTDataset("../../GF/gf-wordnet/examples.txt")
    model = ASTGenerationModel(dataset,
                               **model_config).to(device)
    dataloader = model.loader(batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=model.token_to_id("[PAD]"))

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=WARMUP_STEPS)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS * len(dataloader) - WARMUP_STEPS, eta_min=0)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[WARMUP_STEPS])

    print(f"Training for {N_EPOCHS} epochs with {len(dataloader)} steps per epoch")
    best_loss = float('inf')

    testing_data = torch.tensor(dataset.testing_data)
    test_x = testing_data[:,:-1].to(device)
    test_y = testing_data[:,1: ].to(device)

    for epoch in range(N_EPOCHS):
        epoch_loss = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{N_EPOCHS}")
        for x, y, pos_rw, neg_rw in progress_bar:
            x = x.to(device)
            y = y.to(device)
            pos_rw = pos_rw.to(device)
            neg_rw = neg_rw.to(device)

            # Create causal mask
            #
            mask = create_causal_mask(x.shape[1], device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(x, mask.unsqueeze(0))

            # Compute loss
            loss1 = model.loss(pos_rw, neg_rw)
            loss2 = loss_fn(outputs.view(-1, outputs.shape[-1]), y.view(-1))
            loss  = loss1+loss2

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), CLIP_NORM, error_if_nonfinite=True
            )
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

            # Show loss in tqdm
            progress_bar.set_postfix_str(f'lr={optimizer.param_groups[0]["lr"]:.4f} loss={loss1.item():.4f}+{loss2.item():.4f}={loss.item():.4f}')

        with torch.no_grad():
            mask = create_causal_mask(test_x.shape[1], device)
            outputs = model(test_x, mask.unsqueeze(0))
            loss = loss_fn(outputs.view(-1, outputs.shape[-1]), test_y.view(-1)).item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{N_EPOCHS}; Avg loss: {avg_loss:.4f}; Test loss: {loss:.4f}")

        # Save checkpoint if loss improved
        if loss < best_loss:
            print("Saving...")
            best_loss = loss
            torch.save((model.state_dict(),dataset), "ast_model.pth")


elif sys.argv[1] == "prune":
    state_dict,dataset = torch.load("ast_model.pth",weights_only=False)
    model = ASTGenerationModel(dataset,
                               **model_config).to(device)
    model.load_state_dict(state_dict)

    with open("vectors.tsv","w") as vf, open("meta.tsv","w") as mf:
        mf.write("function\tgloss\tcategory\n")
        for lex_fun in wordnet.grammar.functions:
            gloss = dataset.lex_funs.get(lex_fun,"")
            vf.write("\t".join(map(str,model.embedding.weight[dataset.vocab[lex_fun]].tolist()))+"\n")
            mf.write(lex_fun+"\t"+gloss+"\t"+str(wordnet.grammar.functionType(lex_fun))+"\n")

    model.prune()
    torch.save((model.state_dict(),dataset), "ast_model.pth")
    
    print("done")

elif sys.argv[1] == "test":

    state_dict,dataset = torch.load("ast_model.pth",weights_only=False)
    model = ASTGenerationModel(dataset,
                               **model_config).to(device)
    model.load_state_dict(state_dict)

    # Generation function
    def generate_ast(model, prompt, max_length=100, temperature=0.7):
        model.eval()
        device = next(model.parameters()).device

        # Encode the prompt
        input_ids = torch.tensor(model.encode(prompt)).unsqueeze(0).to(device)
        i = 0

        def to_expr(cat):
            nonlocal input_ids, i

            if i < len(prompt):
                fun = prompt[i]
                i += 1
            else:
                outputs = model(input_ids)
                next_token_logits = outputs[:, -1, :] / temperature
                
                for id in range(0,len(model.dataset.vocab)):
                    fun = model.id_to_token(id)
                    try:
                        if type(fun) != str or wordnet.grammar.functionType(fun).cat != cat:
                            next_token_logits[0,id] = -float('inf')
                    except KeyError:
                        next_token_logits[0,id] = -float('inf')

                # Sample from the distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                # Append to input_ids
                input_ids = torch.cat([input_ids, next_token], dim=1)
                # Stop if we predict the end token
                fun = model.id_to_token(next_token[0].item())
                prob = probs[0,next_token[0].item()].item()

            if fun == "[CLS]":
                return to_expr(cat)
            else:
                ty = wordnet.grammar.functionType(fun)
                e  = pgf.ExprFun(fun)
                for hypo in ty.hypos:
                    arg,arg_prob = to_expr(hypo[2].cat)
                    e = pgf.ExprApp(e,arg)
                    prob += arg_prob

                return e,prob

        with torch.no_grad():
            res = to_expr("Phr")

        return res

    print('-'*10)
    for i in range(3):
        generated,prob = generate_ast(model, ["[CLS]"], temperature=0.9)
        print(generated,prob)
        print(wordnet.linearize("eng",generated))
