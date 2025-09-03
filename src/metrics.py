import math, torch

@torch.no_grad()
def evaluate_next_token_metrics(model, get_batch_fn, iters=100):
    model.eval()
    losses, correct, total = [], 0, 0
    for _ in range(iters):
        xb, yb = get_batch_fn("val")
        logits, loss = model(xb, yb)
        losses.append(loss.item())
        pred = logits[:, :-1, :].argmax(dim=-1)     # predict next token at t from t
        targ = yb[:, :-1]
        correct += (pred == targ).sum().item()
        total   += targ.numel()
    avg_loss = sum(losses)/len(losses)
    ppl = math.exp(min(20.0, avg_loss))            # clamp to avoid overflow
    acc = correct / max(1,total)
    return {"val_loss": avg_loss, "perplexity": ppl, "next_token_acc": acc}
