# Build an LLM From Scratch (reference pipeline)

> **This is the original from-scratch GPT-2 pipeline, kept as a reference/learning
> exercise.** It is **not** the actively developed coding chatbot — that's the
> Qwen2.5-Coder + LoRA pipeline documented in
> [`README_HF_PIPELINE.md`](README_HF_PIPELINE.md), which is what
> `app_hf.py` / `autonomous_trainer_hf.py` / `lora_finetune.py` etc. belong to.
>
> **Why the switch:** after continued pretraining + instruction fine-tuning,
> this pipeline was tested against 5 real DSA problems (`eval_set.json`) by
> actually executing its generated code. It scored **0/5** — genuine bugs in
> every response (unterminated strings, indentation that didn't match any
> consistent level, mismatched function/variable names). This isn't a data or
> training-recipe problem; GPT-2's BPE tokenizer doesn't track Python's
> whitespace-sensitive syntax reliably, and 124M/355M parameters isn't enough
> capacity to compensate. This pipeline is genuinely useful for understanding
> how a GPT model is built and trained end-to-end — just don't expect
> reliable code generation from it.

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

`requirements.txt` is shared with the active Qwen2.5-Coder pipeline (both
live in this same project), so this installs more than this pipeline alone
strictly needs (`transformers`, `peft`, `accelerate`, `bitsandbytes`,
`datasets`, `fastapi`, `uvicorn` are for the other pipeline; `tiktoken`,
`numpy`, `matplotlib`, `pandas` are what this one actually uses).

The first time you run anything that uses `tiktoken.get_encoding("gpt2")`, it
downloads the GPT-2 BPE vocab files (~2MB) from OpenAI's servers and caches
them locally — needs internet once, then works offline.

## 2. Files

| File | What it does |
|---|---|
| `model.py` | The GPT architecture itself: attention, transformer blocks, the full model |
| `dataset.py` | Sliding-window dataset/dataloader for next-token prediction |
| `generate.py` | Text generation: greedy decoding + temperature/top-k sampling |
| `train.py` | **Chapter 5** — pretrains a GPT model from **random** weights on `data/the-verdict.txt`. Demonstrates training mechanics only; not for continued pretraining on real pretrained weights (see `continue_pretrain.py` for that). |
| `load_pretrained.py` | **Chapter 5 (cont.)** — downloads real OpenAI GPT-2 weights and loads them into our architecture |
| `continue_pretrain.py` | Continues training **real pretrained GPT-2 weights** on a custom corpus (e.g. code), unlike `train.py` which starts from scratch |
| `classify_finetune.py` | **Chapter 6** — fine-tunes GPT-2 as a spam/ham classifier (UCI SMS Spam Collection) |
| `diagnose_classifier.py` | Per-class accuracy/confusion-matrix breakdown for the spam classifier |
| `instruction_finetune.py` | **Chapter 7** — fine-tunes GPT-2 to follow instructions, Alpaca-style |
| `app.py` | Serves the fine-tuned model over a local chat API with hot-reload (`/api/reload`). **No conversation history, no save-to-project** — see the compatibility note above before pointing the shared `static/index.html` frontend at this. |
| `autonomous_trainer.py` | Scheduled learn-from-users loop for this pipeline (harvest chat logs -> filter via `sandbox.py` -> retrain -> auto-eval -> promote/reject). See `AUTONOMOUS_TRAINING_README.md`. |
| `eval_checkpoint.py` | Runs `eval_set.json` against a checkpoint and reports a real, executed pass rate |
| `sandbox.py`, `interaction_logger.py` | Shared with the Qwen pipeline: sandboxed code execution for the training filter, and chat-interaction logging |

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

### Continue pretraining on a custom corpus (e.g. code)
```bash
python continue_pretrain.py --data data/code_corpus.txt --epochs 3
```
Starts from real pretrained weights (not random init like `train.py`) and
keeps training on your own text/code corpus. This was the first step toward
a coding-focused version of this model, before testing showed the approach's
limits (see the note at the top of this file).

### Fine-tune as a spam classifier
```bash
python classify_finetune.py --epochs 5
```
Downloads the UCI SMS Spam Collection automatically, balances the classes,
swaps GPT-2's output head for a 2-class classifier, and fine-tunes just the
last transformer block + new head (fast, doesn't need much data). Then:
```bash
python classify_finetune.py --text "You won a free prize! Click here now!!!"
python diagnose_classifier.py   # per-class accuracy breakdown
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

### Serve it as a chatbot
```bash
python app.py --model checkpoints/instruction_model.pth
```
Serves over `http://localhost:8000`. Again: this shares `static/index.html`
with the Qwen pipeline's `app_hf.py`, but doesn't implement the
conversation-history or save-to-project endpoints that frontend expects —
expect the sidebar and save button to not work correctly here.

### Evaluate honestly
```bash
python eval_checkpoint.py --checkpoint checkpoints/instruction_model.pth --verbose
```
Actually executes the model's generated code against `eval_set.json` and
reports a real pass rate, rather than eyeballing output quality.

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
  dependency).

## 5. Hardware

Everything here runs on CPU, just slowly. If you have a GPU:
```bash
python train.py --device cuda
python load_pretrained.py --device cuda --size "gpt2-medium (355M)"
```
On Apple Silicon, use `--device mps` instead of `--device cuda`.

## 6. What's deliberately *not* included

- Multi-GPU / distributed training (single-device only, matching the book's scope)
- DPO/RLHF-style preference tuning
- Conversation memory / multi-turn chat, and save-to-project — these exist
  in the Qwen pipeline (`app_hf.py`) but were never built for this one, since
  development moved to Qwen before those features existed
