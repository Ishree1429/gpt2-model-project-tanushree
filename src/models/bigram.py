import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- helper lives at module level (NOT inside the class) ----
def _top_k_top_p_filtering(logits, top_k=0, top_p=0.0):
    # logits: (B, C)
    if top_k and top_k > 0:
        v, _ = torch.topk(logits, top_k)
        thresh = v[..., -1, None]
        logits = torch.where(logits < thresh, torch.full_like(logits, float('-inf')), logits)
    if top_p and top_p > 0.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(probs, dim=-1)
        mask = cumprobs > top_p
        # keep at least the top token
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        sorted_logits = torch.where(mask, torch.full_like(sorted_logits, float('-inf')), sorted_logits)
        logits = torch.scatter(logits, -1, sorted_idx, sorted_logits)
    return logits

class BigramLM(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  # (B,T,C)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, top_k: int = 0, top_p: float = 0.0):
        t = max(float(temperature), 1e-6)
        for _ in range(max_new_tokens):
            logits, _ = self(idx)            # (B,T,C)
            logits = logits[:, -1, :] / t    # (B,C)
            logits = _top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, 1)  # (B,1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
