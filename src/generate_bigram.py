import argparse, torch
from src.models.bigram import BigramLM

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/bigram.pt")
    ap.add_argument("--num_tokens", type=int, default=300)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    stoi, itos = ckpt["stoi"], ckpt["itos"]
    V = len(itos)

    model = BigramLM(V)
    model.load_state_dict(ckpt["state"])
    model.eval()

    start = stoi.get("\n", 0)
    idx = torch.tensor([[start]], dtype=torch.long)
    out = [start]

    for _ in range(args.num_tokens):
        logits = model(idx)[:, -1, :]
        probs  = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        out.append(int(next_id.item()))
        idx = torch.cat([idx, next_id], dim=1)

    print("".join(itos[i] for i in out))

if __name__ == "__main__":
    main()
