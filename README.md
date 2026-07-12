# Build an LLM From Scratch

A from-scratch GPT implementation in PyTorch, following the path you've already
worked through in your notebooks: tokenizer -> BPE -> embeddings -> attention ->
GPT architecture -> pretraining -> loading real GPT-2 weights -> fine-tuning.

All architecture code here matches what's in your notebooks exactly — this just
turns it into clean, runnable `.py` files instead of one long notebook, and
finishes the parts you hadn't reached yet (top-k/temperature sampling, loading
real GPT-2 weights, classification fine-tuning, instruction fine-tuning).

## 1. Setup (run once)

```bash
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The first time you run anything that uses `tiktoken.get_encoding("gpt2")`, it
downloads the GPT-2 BPE vocab files (~2MB) from OpenAI's servers and caches
them locally — needs internet once, then works offline.

## 2. Files

| File | What it does |
|---|---|
| `model.py` | The GPT architecture itself: attention, transformer blocks, the full model |
| `dataset.py` | Sliding-window dataset/dataloader for next-token prediction |
| `generate.py` | Text generation: greedy decoding + temperature/top-k sampling |
| `train.py` | **Chapter 5** — pretrains a GPT model from random weights on `data/the-verdict.txt` |
| `load_pretrained.py` | **Chapter 5 (cont.)** — downloads real OpenAI GPT-2 weights and loads them into our architecture |
| `classify_finetune.py` | **Chapter 6** — fine-tunes GPT-2 as a spam/ham classifier (UCI SMS Spam Collection) |
| `instruction_finetune.py` | **Chapter 7** — fine-tunes GPT-2 to follow instructions, Alpaca-style |

## 3. Run it

### Pretrain from scratch (small, fast, just to see the mechanics work)
```bash
python train.py --epochs 10
```
Trains a 124M-parameter GPT with random initial weights on the 20KB short story
in `data/the-verdict.txt`. This is too little data to produce coherent text —
it's here to demonstrate the training loop, not to make a useful model. Takes
a few minutes on CPU.

### Load and use real pretrained GPT-2 (this is the model that actually writes English)
```bash
python load_pretrained.py --size "gpt2-small (124M)" --prompt "The future of AI is"
```
Downloads OpenAI's real GPT-2 weights via Hugging Face (~500MB, one time) and
loads them into *your own* `GPTModel` class — proving your from-scratch
architecture is bit-for-bit compatible with the real thing. Available sizes:
`"gpt2-small (124M)"`, `"gpt2-medium (355M)"`, `"gpt2-large (774M)"`, `"gpt2-xl (1558M)"`.

### Fine-tune as a spam classifier
```bash
python classify_finetune.py --epochs 5
```
Downloads the UCI SMS Spam Collection automatically, balances the classes,
swaps GPT-2's output head for a 2-class classifier, and fine-tunes just the
last transformer block + new head (fast, doesn't need much data). Then:
```bash
python classify_finetune.py --text "You won a free prize! Click here now!!!"
```

### Fine-tune to follow instructions (Alpaca-style)
```bash
python instruction_finetune.py --epochs 2
```
Downloads the 1,100-example instruction dataset automatically and fine-tunes
GPT-2 to follow instructions in the same `instruction/input/output` format
popularized by [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca).
Defaults to the 355M model since instruction-following needs more capacity
than 124M to work well — pass `--size "gpt2-small (124M)"` if you want it
faster/lighter. Then:
```bash
python instruction_finetune.py --prompt "Convert 10 miles to kilometers."
```

## 4. Notes on the datasets you linked

- **`instruction-data.json`** (rasbt's repo) — downloaded automatically by
  `instruction_finetune.py`, same Alpaca instruction/input/output format.
- **Stanford Alpaca** — the format `instruction_finetune.py` uses is exactly
  Alpaca's. If you want the *full* 52K-example Alpaca dataset instead of the
  1,100-example one, swap `DATA_URL` in `instruction_finetune.py` for
  Alpaca's `alpaca_data.json` — the rest of the code needs no changes.
- **UCI SMS Spam Collection** — downloaded automatically by `classify_finetune.py`.
- **OpenAI GPT-2 Weights (Kaggle)** — `load_pretrained.py` uses Hugging Face
  instead (simpler, no manual download/Kaggle login needed, no TensorFlow
  dependency). If you'd rather use the Kaggle mirror, download it, extract it,
  and tell me the folder structure — I'll add a loader for it.

## 5. Hardware

Everything here runs on CPU, just slowly. If you have a GPU:
```bash
python train.py --device cuda
python load_pretrained.py --device cuda --size "gpt2-medium (355M)"
```
On Apple Silicon, use `--device mps` instead of `--device cuda`.

## 6. What's deliberately *not* included

- Multi-GPU / distributed training (single-device only, matching the book's scope)
- DPO/RLHF-style preference tuning (chapter 7 of the book covers this as an
  extra — say the word if you want it added)
- A web UI — these are all CLI scripts by design, for transparency over what's
  actually happening
