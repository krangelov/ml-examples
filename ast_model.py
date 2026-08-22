import os
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import pgf

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

gr = pgf.readNGF('/usr/local/share/x86_64-linux-ghc-9.4.7/gf-4.0.0/www/Parse.ngf')
cnc = gr.languages["ParseEng"]

# Create PyTorch dataset
class ASTDataset(torch.utils.data.Dataset):
    def __init__(self, fpaths):
        dataset = []
        self.encoded = []
        self.vocab = {"[CLS]": 0, "[EOS]": 1, "[PAD]": 2}
        self.max_seq_len = 0
        for fpath in fpaths:
            with open(fpath) as f:
                for line in f:
                    if line.startswith("abs:"):
                        sent = ["[CLS]"]+exprFunctions(pgf.readExpr(line[5:]))
                        sent_ids = []
                        for fun in sent:
                            id = self.vocab.get(fun)
                            if id == None:
                                id = len(self.vocab)
                                self.vocab[fun] = id
                            sent_ids.append(id)
                        if self.max_seq_len < len(sent):
                            self.max_seq_len = len(sent)
                        dataset.append(sent)
                        self.encoded.append(sent_ids)

        self.max_seq_len += 1
        for sent_ids in self.encoded:
            sent_ids.append(1)
            while len(sent_ids) < self.max_seq_len:
                sent_ids.append(2)

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        chunk = self.encoded[idx]
        x = torch.tensor(chunk[:-1])
        y = torch.tensor(chunk[1:])
        return x, y

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)

class RotaryPositionalEncoding(nn.Module):
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

class SwiGLU(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, intermediate_dim)
        self.up = nn.Linear(hidden_dim, intermediate_dim)
        self.down = nn.Linear(intermediate_dim, hidden_dim)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.act(self.gate(x)) * self.up(x)
        x = self.down(x)
        return x

class GQA(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_kv_heads=None, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = hidden_dim // num_heads
        self.num_groups = num_heads // num_kv_heads
        self.dropout = dropout
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

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

class DecoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_kv_heads, dropout=0.1):
        super().__init__()
        self.self_attn = GQA(hidden_dim, num_heads, num_kv_heads, dropout)
        self.mlp = SwiGLU(hidden_dim, 4 * hidden_dim)
        self.norm1 = nn.RMSNorm(hidden_dim)
        self.norm2 = nn.RMSNorm(hidden_dim)

    def forward(self, x, mask=None, rope=None):
        # self-attention sublayer
        out = self.norm1(x)
        out = self.self_attn(out, out, out, mask, rope)
        x = out + x
        # MLP sublayer
        out = self.norm2(x)
        out = self.mlp(out)
        return out + x

class ASTGenerationModel(nn.Module):
    def __init__(self, num_layers, num_heads, num_kv_heads, hidden_dim,
                 max_seq_len, vocab, dropout=0.1):
        super().__init__()
        self.vocab = vocab
        self.keys  = list(vocab.keys())
        self.rope = RotaryPositionalEncoding(hidden_dim // num_heads, max_seq_len)
        self.embedding = nn.Embedding(len(vocab), hidden_dim)
        self.decoders = nn.ModuleList([
            DecoderLayer(hidden_dim, num_heads, num_kv_heads, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.RMSNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, len(vocab))

    def forward(self, ids, mask=None):
        x = self.embedding(ids)
        for decoder in self.decoders:
            x = decoder(x, mask, self.rope)
        x = self.norm(x)
        return self.out(x)

    def encode(self, funs):
        fun_ids = []
        for fun in funs:
            fun_ids.append(self.vocab[fun])
        return fun_ids

    def decode(self, fun_ids):
        funs = []
        for fun_id in fun_ids:
            funs.append(self.keys[fun_id])
        return funs

    def token_to_id(self,tok):
        return self.vocab[tok]

    def id_to_token(self,fun_id):
        return self.keys[fun_id]

def create_causal_mask(seq_len, device):
    """Create a causal mask for autoregressive attention."""
    mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device), diagonal=1)
    return mask

# Training configuration
model_config = {
    "num_layers": 8,
    "num_heads": 8,
    "num_kv_heads": 4,
    "hidden_dim": 512,
    "dropout": 0.1,
}

dataset = ASTDataset(["../../GF/gf-wordnet/examples.txt","log_unique.txt"])

# Initialize model, optimizer, etc.
device = torch.device('cpu') #'cuda' if torch.cuda.is_available() else 'cpu')
model = ASTGenerationModel(**model_config, vocab=dataset.vocab, max_seq_len=dataset.max_seq_len).to(device)

# Create dataset and dataloader
BATCH_SIZE = 32
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Training loop
if os.path.exists("astgen_model.pth"):
    model.load_state_dict(torch.load("astgen_model.pth"))
else:
    N_EPOCHS = 30
    LR = 0.0005
    WARMUP_STEPS = 2000
    CLIP_NORM = 6.0

    optimizer = optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss(ignore_index=model.token_to_id("[PAD]"))

    # Learning rate scheduling
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=WARMUP_STEPS)
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS * len(dataloader) - WARMUP_STEPS, eta_min=0)
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[WARMUP_STEPS])

    print(f"Training for {N_EPOCHS} epochs with {len(dataloader)} steps per epoch")
    best_loss = float('inf')

    for epoch in range(N_EPOCHS):
        model.train()
        epoch_loss = 0

        progress_bar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{N_EPOCHS}")
        for x, y in progress_bar:
            x = x.to(device)
            y = y.to(device)

            # Create causal mask
            mask = create_causal_mask(x.shape[1], device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(x, mask.unsqueeze(0))

            # Compute loss
            loss = loss_fn(outputs.view(-1, outputs.shape[-1]), y.view(-1))

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), CLIP_NORM, error_if_nonfinite=True
            )
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

            # Show loss in tqdm
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{N_EPOCHS}; Avg loss: {avg_loss:.4f}")

        # Save checkpoint if loss improved
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "astgen_model.pth")

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
            
            for id in range(3,len(model.vocab)):
                fun = model.id_to_token(id)
                if gr.functionType(fun).cat != cat:
                    next_token_logits[0,id] = -float('inf')

            # Sample from the distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            # Append to input_ids
            input_ids = torch.cat([input_ids, next_token], dim=1)
            # Stop if we predict the end token
            fun = model.id_to_token(next_token[0].item())

        if fun == "[CLS]":
            return to_expr(cat)
        elif fun == "[EOS]":
            return pgf.ExprMeta(0)
        else:
            ty = gr.functionType(fun)
            e  = pgf.ExprFun(fun)
            for hypo in ty.hypos:
                arg = to_expr(hypo[2].cat)
                e = pgf.ExprApp(e,arg)

            return e

    with torch.no_grad():
        e = to_expr("Phr")

    return e

print('-'*10)
if False:
    with open("ex_emb.tsv","w") as f:
        for fun,id in model.vocab.items():
            f.write("\t".join(str(d.item()) for d in model.embedding(torch.LongTensor([id]).to(device))[0])+"\n")
    with open("ex_emb_meta.tsv","w") as f:
        for fun,id in model.vocab.items():
            f.write(fun+"\n")
generated = generate_ast(model, ["[CLS]"])
print(generated)
print(cnc.linearize(generated))
