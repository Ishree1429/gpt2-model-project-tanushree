import argparse, torch
from src.models.gptmini import GPTMini, device_pick
from src.tokenizer import CharTokenizer
from src.dataset import load_lyrics_text

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/gptmini.pt")
    ap.add_argument("--num_tokens", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--prompt", default="")
    ap.add_argument("--data", default="data/lyrics.txt")
    args = ap.parse_args()

    device = device_pick()
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt.get("cfg", {})
    text = load_lyrics_text(args.data)
    tok = CharTokenizer(text); 
    if "stoi" in ckpt and "itos" in ckpt:
        tok.stoi, tok.itos = ckpt["stoi"], ckpt["itos"]; tok.vocab_size = len(tok.stoi)

    model = GPTMini(tok.vocab_size, block_size=cfg.get("block_size",256),
                    n_layer=cfg.get("n_layer",4), n_head=cfg.get("n_head",4),
                    n_embd=cfg.get("n_embd",128)).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()

    if args.prompt:
        start = torch.tensor([tok.encode(args.prompt)], dtype=torch.long, device=device)
    else:
        start = torch.zeros((1,1), dtype=torch.long, device=device)
    out = model.generate(start, args.num_tokens, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)[0].tolist()
    print(tok.decode(out))

if __name__ == "__main__":
    main()
