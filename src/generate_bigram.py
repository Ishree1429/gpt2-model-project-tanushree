import argparse, torch
from src.dataset import load_lyrics_text
from src.tokenizer import CharTokenizer
from src.models.bigram import BigramLM

def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/bigram.pt")
    ap.add_argument("--num_tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--top_p", type=float, default=0.0)
    ap.add_argument("--data", default="data/lyrics.txt")
    args = ap.parse_args()

    device = pick_device()

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=device)

    # Build tokenizer from checkpoint if available; else rebuild from data
    if isinstance(ckpt, dict) and "stoi" in ckpt and "itos" in ckpt:
        tok = CharTokenizer("")
        tok.stoi = ckpt["stoi"]
        tok.itos = ckpt["itos"]
        tok.vocab_size = len(tok.stoi)
    else:
        text = load_lyrics_text(args.data)
        tok = CharTokenizer(text)

    # Resolve state dict
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    # Key rename support: table.weight -> token_embedding_table.weight
    if any(k.startswith("table.") for k in state.keys()):
        new_state = {}
        for k, v in state.items():
            nk = k.replace("table.", "token_embedding_table.")
            new_state[nk] = v
        state = new_state

    model = BigramLM(tok.vocab_size).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    ids = model.generate(context, args.num_tokens, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)[0].tolist()
    print(tok.decode(ids))

if __name__ == "__main__":
    main()
