import math, torch
import torch.nn as nn
import torch.nn.functional as F

def device_pick():
    if torch.cuda.is_available(): return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): return "mps"
    return "cpu"

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, attn_pdrop=0.0, resid_pdrop=0.0):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1,1,block_size,block_size))

    def forward(self, x):
        B,T,C = x.size()
        k = self.key(x).view(B,T,self.n_head,self.head_dim).transpose(1,2)   # (B,h,T,hd)
        q = self.query(x).view(B,T,self.n_head,self.head_dim).transpose(1,2) # (B,h,T,hd)
        v = self.value(x).view(B,T,self.n_head,self.head_dim).transpose(1,2) # (B,h,T,hd)
        att = (q @ k.transpose(-2,-1)) / math.sqrt(self.head_dim)            # (B,h,T,T)
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v                                                            # (B,h,T,hd)
        y = y.transpose(1,2).contiguous().view(B,T,C)                          # (B,T,C)
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, attn_pdrop=0.0, resid_pdrop=0.0, mlp_pdrop=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, attn_pdrop, resid_pdrop)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.GELU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(mlp_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPTMini(nn.Module):
    def __init__(self, vocab_size, block_size=128, n_layer=4, n_head=4, n_embd=128, embd_pdrop=0.0):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(embd_pdrop)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B,T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=0, top_p=0.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k > 0:
                v,_ = torch.topk(logits, top_k)
                thresh = v[..., -1, None]
                logits = torch.where(logits < thresh, torch.full_like(logits, float('-inf')), logits)
            if top_p > 0.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                probs = torch.softmax(sorted_logits, dim=-1)
                cumprobs = torch.cumsum(probs, dim=-1)
                mask = cumprobs > top_p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = False
                sorted_logits = torch.where(mask, torch.full_like(sorted_logits, float('-inf')), sorted_logits)
                logits = torch.scatter(logits, -1, sorted_idx, sorted_logits)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
