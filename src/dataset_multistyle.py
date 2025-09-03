import torch, random

class MultiStyleBatcher:
    def __init__(self, datasets, probs, block_size, batch_size, device):
        assert abs(sum(probs)-1.0) < 1e-6, "probs must sum to 1"
        self.datasets = [torch.tensor(d, dtype=torch.long, device=device) for d in datasets]
        self.probs = probs
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device

    def _choose(self):
        r = random.random(); cum = 0.0
        for i,p in enumerate(self.probs):
            cum += p
            if r <= cum: return i
        return len(self.probs)-1

    def get_batch(self, split="train"):
        xs, ys, ss = [], [], []
        for _ in range(self.batch_size):
            di = self._choose()
            data = self.datasets[di]
            i = torch.randint(0, len(data) - self.block_size - 1, (1,), device=self.device).item()
            x = data[i:i+self.block_size]
            y = data[i+1:i+self.block_size+1]
            xs.append(x); ys.append(y); ss.append(di)
        x = torch.stack(xs); y = torch.stack(ys)
        s = torch.tensor(ss, dtype=torch.long, device=self.device)
        return x, y, s
