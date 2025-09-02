import torch, torch.nn as nn

class BigramLM(nn.Module):
    """
    Learns P(next_char | current_char) via an embedding table used as logits.
    """
    def __init__(self, vocab_size: int):
        super().__init__()
        self.table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx):  # idx: (B,T) of token ids
        # returns logits for next token: (B,T,V)
        return self.table(idx)
