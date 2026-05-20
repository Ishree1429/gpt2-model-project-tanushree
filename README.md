# Mini-GPT: Building a Transformer Language Model from Scratch

A small GPT-style language model built end-to-end — tokenization, attention, training loop, and inference — on song lyrics, with **style-token fine-tuning** for controllable generation.

I built this to understand what's actually happening *one layer below* the model APIs everyone calls. If you can build it from scratch, you understand why agents and LLM systems behave the way they do.

---

## Results

| Model | Perplexity | Next-token accuracy (char-level) |
|---|---|---|
| Bigram baseline | — | — |
| Mini-GPT (base) | — | — |
| Mini-GPT (fine-tuned with style token) | **4.80** | **0.539** |

Style control works: the same prompt produces noticeably different outputs depending on the style token. See `samples/style0.txt` vs `samples/style1.txt`.

---

## Sample output

**Prompt:** `I miss you`

- **Bigram baseline:** see `showcase/bigram.txt`
- **Mini-GPT base:** see `showcase/minigpt.txt`
- **Fine-tuned, Style 0:** see `showcase/style0.txt`
- **Fine-tuned, Style 1:** see `showcase/style1.txt`

---

## What's in herewcase/style1.txt`

├── src/          # Model, training loop, tokenizer
├── notebooks/    # Exploratory work, training runs
├── data/         # Lyric corpus, processed splits
├── samples/      # Generated text per style token
├── showcase/     # Curated example outputs
└── requirements.txt

---

## How to run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Then run the notebooks in `notebooks/` in order:
1. Day 3 — Bigram baseline
2. Day 4 — Mini-GPT (base train + fine-tune with style tokens)

---

## What I learned

- Why character-level perplexity ≠ word-level perplexity, and when each matters
- How attention masks and positional encodings actually shape generation
- Why style-token fine-tuning works better than I expected for small models
- The gap between "model trains" and "model generates coherently" — they are not the same milestone

---

## Next steps

- Move from character-level to BPE tokenization
- Add temperature / top-k / top-p sampling controls
- Try LoRA fine-tuning on a larger base model
- Build a small inference API around it

---

**Built by:** [Tanushree Sharma](https://github.com/Ishree1429) — Master's in Data Science, Illinois Tech
