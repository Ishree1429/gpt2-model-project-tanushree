import argparse, torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from src.dataset import load_lyrics_text
from src.tokenizer import CharTokenizer
from src.models.bigram import BigramLM

class CharDataset(Dataset):
    def __init__(self, tokens, block_size=128):
        self.block_size = block_size
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        n = int(0.9 * len(self.tokens))
        self.train = self.tokens[:n]
        self.val = self.tokens[n:]

    def split(self, name):
        return self.train if name == "train" else self.val

    def get_batch(self, name, batch_size):
        data = self.split(name)
        ix = torch.randint(len(data) - self.block_size, (batch_size,))
        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])
        return x, y

def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/lyrics.txt")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--block_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--eval_interval", type=int, default=200)
    ap.add_argument("--out", default="checkpoints/bigram.pt")
    args = ap.parse_args()

    # Build tokenizer and ids
    text = load_lyrics_text(args.data)
    tok = CharTokenizer(text)
    ids = tok.encode(text)

    ds = CharDataset(ids, block_size=args.block_size)
    device = pick_device()
    model = BigramLM(tok.vocab_size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for step in tqdm(range(args.steps), desc=f"training on {device}"):
        if step % args.eval_interval == 0:
            with torch.no_grad():
                xb, yb = ds.get_batch("val", args.batch_size)
                xb, yb = xb.to(device), yb.to(device)
                _, l = model(xb, yb)
                print(f"\nval step {step}: loss {l.item():.4f}")

        xb, yb = ds.get_batch("train", args.batch_size)
        xb, yb = xb.to(device), yb.to(device)
        _, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    torch.save({"model": model.state_dict(), "stoi": tok.stoi, "itos": tok.itos}, args.out)
    print(f"✅ saved checkpoint: {args.out}")

if __name__ == "__main__":
    main()
