import argparse, torch
from tqdm import tqdm
from src.dataset import load_lyrics_text
from src.tokenizer import CharTokenizer
from src.models.gptmini import GPTMini, device_pick
from src.metrics import evaluate_next_token_metrics

def get_batches(ids, block_size, batch_size, device, split=0.9):
    import torch
    ids = torch.tensor(ids, dtype=torch.long, device=device)
    n = int(split * len(ids))
    train, val = ids[:n], ids[n:]

    def batch(which):
        data = train if which == "train" else val
        ix = torch.randint(len(data) - block_size, (batch_size,), device=device)
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        return x, y

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
    ap.add_argument("--embd_pdrop", type=float, default=0.1)
    ap.add_argument("--out", default="checkpoints/gptmini.pt")
    args = ap.parse_args()

    # Data & tokenizer
    text = load_lyrics_text(args.data)
    tok = CharTokenizer(text)
    ids = tok.encode(text)

    # Model
    device = device_pick()
    model = GPTMini(tok.vocab_size,
                    block_size=args.block_size,
                    n_layer=args.n_layer,
                    n_head=args.n_head,
                    n_embd=args.n_embd,
                    embd_pdrop=args.embd_pdrop).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Batching
    getb = get_batches(ids, args.block_size, args.batch_size, device)

    for step in tqdm(range(args.steps), desc=f"training GPT on {device}"):
        # Evaluation
        if step % args.eval_interval == 0:
            model.eval()
            with torch.no_grad():
                xb, yb = getb("val")
                _, vloss = model(xb, yb)
                metrics = evaluate_next_token_metrics(model, getb, iters=20)
                print(f"\nval step {step}: loss {vloss.item():.4f} | ppl {metrics['perplexity']:.2f} | acc {metrics['next_token_acc']:.3f}")
            model.train()

        # Train step
        xb, yb = getb("train")
        _, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    # Save
    torch.save({"model": model.state_dict(),
                "stoi": tok.stoi, "itos": tok.itos,
                "cfg": vars(args)}, args.out)
    print(f"✅ saved: {args.out}")

if __name__ == "__main__":
    main()
