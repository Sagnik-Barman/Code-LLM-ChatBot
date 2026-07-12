# Code chatbot: Qwen2.5-Coder + LoRA pipeline

This replaces the from-scratch GPT-2 pipeline for the coding chatbot. GPT-2
scored 0/5 on `eval_set.json` with genuinely broken code (bad indentation,
undefined variables, mismatched function names) even after continued
pretraining + instruction fine-tuning -- a capacity/tokenizer ceiling, not a
data problem. Qwen2.5-Coder-1.5B-Instruct + LoRA fine-tuning on the same
CodeAlpaca data scored **5/5**, with clean, idiomatic, correct solutions.

Your original from-scratch GPT-2 pipeline (`model.py`, `train.py`,
`continue_pretrain.py`, `instruction_finetune.py`, `app.py`,
`autonomous_trainer.py`, `eval_checkpoint.py`) still exists and still works
-- it's just no longer the one to build on for actual coding quality.

## Files in this pipeline

| File | Role |
|---|---|
| `code_model.py` | Loads Qwen2.5-Coder-1.5B-Instruct (4-bit quantized) via `transformers`; `load_base_model()` + `attach_adapter()` are split so reloading only swaps the cheap LoRA adapter, not the whole model |
| `lora_finetune.py` | LoRA fine-tuning; `train_lora()` is reusable by both its own CLI and the autonomous trainer |
| `eval_checkpoint_hf.py` | Baseline evaluator -- executes generated code against `eval_set.json`'s real test assertions; `evaluate()` is reusable |
| `app_hf.py` | Chat server with `/api/reload` hot-reload (adapter-only swap, base model stays resident) |
| `autonomous_trainer_hf.py` | Learn-from-users loop: harvest -> filter (`sandbox.py`) -> mix with replay buffer -> LoRA fine-tune a candidate -> auto-eval -> promote (+ hot-reload) or reject |
| `sandbox.py`, `interaction_logger.py`, `eval_set.json` | Unchanged, shared with the original pipeline |

## Run order

**1. Fine-tune the first adapter** (already done):
```powershell
python lora_finetune.py --data data/code_alpaca_20k.json --out checkpoints/code_lora_v1 --epochs 2
```

**2. Check it** (already done -- got 5/5):
```powershell
python eval_checkpoint_hf.py --adapter checkpoints/code_lora_v1 --verbose
```

**3. Serve it:**
```powershell
python app_hf.py --adapter checkpoints/code_lora_v1
```
Open `http://localhost:8000` to chat with it.

**4. Turn on autonomous learning from users** (run as a separate long-lived
process, e.g. a second terminal or a background service, while `app_hf.py`
keeps running):
```powershell
python autonomous_trainer_hf.py --cycle_hours 6
```

## Before running Stage 4

1. **Seed `data/replay_buffer.json`.** If you skip this, the trainer
   automatically falls back to `data/code_alpaca_20k.json` (already on
   disk from Stage 1), so it's not strictly required -- but a smaller,
   more curated replay set is better than replaying the entire 20k-example
   set every cycle. Same JSON shape either way:
   `[{"instruction": ..., "input": ..., "output": ...}, ...]`
2. **`app_hf.py` must be running** for hot-reload to take effect. If it's
   not reachable, the trainer logs a warning, keeps the new adapter marked
   as "live" on disk, and moves on -- it won't crash the cycle, but the
   running app keeps serving the old adapter until restarted or reloaded
   manually (`curl -X POST http://localhost:8000/api/reload`).
3. **`g++` needed for C++ filtering**, same as before; Python-only works
   without it.

## What changed from the GPT-2 version, mechanically

- **Adapters, not full checkpoints.** Each promoted version is a ~20-50MB
  LoRA adapter, not a multi-GB model file -- cheaper to train, store, and
  hot-reload.
- **Base model loads once per cycle**, reused for the baseline eval,
  training, and candidate eval, instead of reloading a 3GB+ model three
  times.
- **No in-training eval loss.** An earlier version of `lora_finetune.py`
  computed held-out LM loss mid-training via `Trainer`'s built-in eval --
  this is what caused a real CUDA OOM crash partway through your first
  training run (different batch/sequence-length profile than training
  batches, on top of already-tight VRAM). Removed in favor of the
  functional `eval_set.json` pass rate, computed only after training
  finishes, which is the metric that actually matters anyway.
- **Custom label-masking collator.** `transformers`' built-in
  `DataCollatorForLanguageModeling(mlm=False)` unconditionally overwrites
  any custom `labels` field with an unmasked copy of `input_ids` --
  silently training on the prompt tokens, not just the response. Caught
  before it shipped; `lora_finetune.py` uses `MaskedLabelCollator` instead.
- **Periodic checkpointing during training** (`--save_steps`, default
  every 100 steps) so an interrupted run doesn't lose all its progress.

## Known limitations

- **`sandbox.py` is a correctness filter, not a hardened security
  sandbox** -- same caveat as before, fine for your own logged traffic,
  not for arbitrary untrusted code.
- **`eval_set.json` is intentionally tiny (5 problems)** -- good enough to
  catch a badly regressed adapter, not a comprehensive benchmark.
- **CodeAlpaca is Python-skewed.** If C++ quality matters as much as
  Python, that's the next data gap worth closing.
