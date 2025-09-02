import argparse, torch
from tqdm import tqdm
from src.dataset import load_lyrics_text
from src.tokenizer import CharTokenizer
from src.models.gptmini import GPTMini, device_pick

def get_batches(ids, block_size, batch_size, device, split=0.9):
    import torch
    ids = torch.tensor(ids, dtype=torch.long)
    n = int(split * len(ids))
    train, val = ids[:n], ids[n:]
    def batch(which):
        data = train if which=="train" else val
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix]).to(device)
        y = torch.stack([data[i+1:i+block_size+1] for i in ix]).to(device)
        return x,y
    return batch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/lyrics.txt")
    ap.add_argument("--block_size", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--eval_interval", type=int, default=200)
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--n_embd", type=int, default=128)
    ap.add_argument("--out", default="checkpoints/gptmini.pt")
    args = ap.parse_args()

    text = load_lyrics_text(args.data)
    tok = CharTokenizer(text)
    ids = tok.encode(text)

    device = device_pick()
    model = GPTMini(tok.vocab_size, block_size=args.block_size, n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    getb = get_batches(ids, args.block_size, args.batch_size, device)
    for step in tqdm(range(args.steps), desc=f"training GPT on {device}"):
        if step % args.eval_interval == 0:
            model.eval()
            with torch.no_grad():
                xb, yb = getb("val")
                _, vloss = model(xb, yb)
                print(f"\nval step {step}: loss {vloss.item():.4f}")
            model.train()
        xb, yb = getb("train")
        _, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    torch.save({"model": model.state_dict(), "stoi": tok.stoi, "itos": tok.itos, "cfg": vars(args)}, args.out)
    print(f"✅ saved: {args.out}")

if __name__ == "__main__":
    main()
