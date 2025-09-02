import argparse, os
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from src.dataset import load_lyrics_text
from src.tokenizer import CharTokenizer
from src.models.bigram import BigramLM

class CharDataset(Dataset):
    def __init__(self, tokens, block_size=128):
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.block_size = block_size
    def __len__(self):
        return len(self.tokens) - self.block_size
    def __getitem__(self, i):
        x = self.tokens[i:i+self.block_size]
        y = self.tokens[i+1:i+self.block_size+1]
        return x, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/lyrics_clean.csv")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--block_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--ckpt", default="checkpoints/bigram.pt")
    args = ap.parse_args()

    text = load_lyrics_text(args.csv)
    tok = CharTokenizer(text)
    ids = tok.encode(text)

    ds = CharDataset(ids, block_size=args.block_size)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BigramLM(tok.vocab_size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    os.makedirs("checkpoints", exist_ok=True)
    for ep in range(args.epochs):
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            B, T, V = logits.shape
            loss = loss_fn(logits.view(B*T, V), y.view(B*T))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        print(f"epoch {ep+1}: loss={loss.item():.3f}")

    torch.save({"state": model.state_dict(), "stoi": tok.stoi, "itos": tok.itos}, args.ckpt)
    print("saved", args.ckpt)

if __name__ == "__main__":
    main()
