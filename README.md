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

**Day 3 — Bigram baseline**  
Prompt: `I miss you`

## Interesting Outputs

**Day 3 — Bigram baseline**  
Prompt: `I miss you`
(see `showcase/bigram.txt`)

**Day 4/5 — Mini-GPT base**  
Prompt: `I miss you`
(see `showcase/minigpt.txt`)

**Day 7 — Fine-tuned with style token**  
Prompt: `I miss you`

Style 0 — (see `showcase/style0.txt`)

Style 1 — (see `showcase/style1.txt`)

**Metrics (Day 6/7)**  
Perplexity: **4.80**  
Next-token accuracy: **0.539** (character-level)


**Fine-tuned with style token**  
Prompt: `I miss you`

Style 0 — see `showcase/style0.txt`  
Style 1 — see `showcase/style1.txt`

