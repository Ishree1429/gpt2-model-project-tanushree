# GPT Song Lyrics: Bigram (Day 3) → Mini-GPT (Day 4)

## How to run
### Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

## Samples
- Bigram: see console sample
- Mini-GPT base: saved after base train
- Fine-tuned Mini-GPT: see samples/style0.txt vs samples/style1.txt


## Results 
- Fine-tuned mini-GPT on lyrics (style token):
  - Perplexity: **4.80**
  - Next-token accuracy: **0.539** (char-level)
- See `samples/style0.txt` vs `samples/style1.txt` for style control.

