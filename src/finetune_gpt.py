import argparse, torch
from tqdm import tqdm
from src.dataset import load_lyrics_text
from src.tokenizer import CharTokenizer
from src.models.gptmini import GPTMini, device_pick
from src.dataset_multistyle import MultiStyleBatcher

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_ckpt", required=True)
    ap.add_argument("--style0", required=True)
    ap.add_argument("--style1", required=True)
    ap.add_argument("--prob0", type=float, default=0.2)
    ap.add_argument("--prob1", type=float, default=0.8)
    ap.add_argument("--block_size", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--eval_interval", type=int, default=200)
    ap.add_argument("--out", default="checkpoints/gptmini_ft.pt")
    args = ap.parse_args()

    device = device_pick()

    # Load corpora and shared tokenizer (must cover both)
    text0 = load_lyrics_text(args.style0)
    text1 = load_lyrics_text(args.style1)
    tok = CharTokenizer(text0 + text1)
    ids0 = tok.encode(text0); ids1 = tok.encode(text1)

    # Build model with style embeddings
    model = GPTMini(tok.vocab_size, block_size=args.block_size, n_layer=4, n_head=4, n_embd=128, num_styles=2).to(device)

    # Load base checkpoint weights (ignore missing style_emb at first)
    ckpt = torch.load(args.base_ckpt, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded base. missing={missing} unexpected={unexpected}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    getb = MultiStyleBatcher([ids0, ids1], [args.prob0, args.prob1],
                             block_size=args.block_size, batch_size=args.batch_size, device=device)

    for step in tqdm(range(args.steps), desc=f"finetuning on {device}"):
        if step % args.eval_interval == 0:
            model.eval()
            with torch.no_grad():
                xb, yb, sb = getb.get_batch("val")
                _, vloss = model(xb, yb, style=sb)
                print(f"\nval step {step}: loss {vloss.item():.4f}")
            model.train()
        xb, yb, sb = getb.get_batch("train")
        _, loss = model(xb, yb, style=sb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    torch.save({"model": model.state_dict(), "stoi": tok.stoi, "itos": tok.itos,
                "cfg": {"block_size": args.block_size, "n_layer": 4, "n_head": 4, "n_embd": 128, "num_styles": 2}}, args.out)
    print(f"✅ saved fine-tuned: {args.out}")

if __name__ == "__main__":
    main()
