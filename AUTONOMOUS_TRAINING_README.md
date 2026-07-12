# Autonomous training add-on

This adds two things to your existing GPT-2 chatbot: (1) actual training on
Python/C++/DSA code, and (2) "learns from user inputs" via a scheduled,
auto-filtered, hot-reloaded training loop.

## Run order (do this once, in this sequence)

**Stage 1 — continued pretraining on code**
```bash
python prepare_code_corpus.py                 # clones TheAlgorithms Python + C++ repos (MIT-licensed),
                                                # builds data/code_corpus.txt
python train.py --data data/code_corpus.txt --epochs 3 --out checkpoints/code_pretrained.pth
```
This exposes the model to real, clean DSA-style code before it ever sees an
instruction format — matches "Stage 1" in your training plan doc.

**Stage 2 — instruction fine-tuning on coding tasks**
```bash
# codealpaca is already registered in instruction_finetune.py's DATASETS dict,
# pointing at the real 20k-example CodeAlpaca dataset (confirmed live, same
# instruction/input/output JSON shape your pipeline already expects)
python instruction_finetune.py --dataset codealpaca --epochs 2 \
    --out checkpoints/instruction_model.pth
```
Note: CodeAlpaca is Python-skewed (~2,600 explicit Python examples vs. ~475
C++ out of 20k). If C++ quality matters as much as Python, supplement with a
C++-specific instruction set later, or weight `eval_set.json` results by
language to see where it's actually weak.

**Stage 3 — serve it**
```bash
python app.py --model checkpoints/instruction_model.pth
```

**Stage 4 — turn on autonomous learning from users**
```bash
python autonomous_trainer.py --cycle_hours 6
```

## How the pieces fit together

| File | Role |
|---|---|
| `prepare_code_corpus.py` | Stage 1: builds a code corpus from TheAlgorithms' Python/C++ repos for continued pretraining |
| `instruction_finetune.py` | *(unchanged code, new dataset)* Stage 2: now has a `codealpaca` entry in `DATASETS` for coding-instruction fine-tuning |
| `interaction_logger.py` | `app.py` calls this after every chat reply; appends to `logs/interactions.jsonl` |
| `app.py` | Serves chat, logs every exchange, and exposes `POST /api/reload` to hot-swap weights with **no restart** (guarded by `STATE_LOCK` so in-flight requests never see a half-swapped model) |
| `sandbox.py` | Runs/compiles code blocks in a subprocess with a timeout + memory limit — the automated safety gate that lets training run without a human reviewer |
| `eval_set.json` | 5 fixed DSA problems with real test assertions, used to auto-grade every candidate checkpoint |
| `autonomous_trainer.py` | Every `--cycle_hours`: harvest new interactions → filter via `sandbox.py` → mix with a replay buffer → fine-tune a candidate → auto-eval it → promote (and hot-reload the live app via `/api/reload`) or reject |

## Before running the autonomous loop (Stage 4)

1. **Seed the replay buffer.** Create `data/replay_buffer.json` — reuse a
   sample of `data/code_alpaca_20k.json` (downloaded automatically in Stage
   2) in the same `{"instruction", "input", "output"}` format. This anchors
   every training cycle so the model doesn't drift or forget after a few
   cycles of thin, recent-only chat data.
2. **`checkpoints/instruction_model.pth` must already exist** (i.e. Stage 2
   is done). The autonomous trainer always fine-tunes *from* the live
   checkpoint, never from scratch.
3. **`g++` must be installed** for C++ code to be eligible for the training
   filter; Python-only works without it.
4. **`app.py` must be running** for hot-reload to take effect — if it's not
   reachable, the trainer logs a warning and leaves the new weights on disk
   for a later manual/restart-triggered reload; it doesn't fail the cycle.

## Hot-reload, if you want to trigger it manually

```bash
curl -X POST http://localhost:8000/api/reload
curl -X POST http://localhost:8000/api/reload -H "Content-Type: application/json" -d '{"model": "checkpoints/some_other_checkpoint.pth"}'
curl http://localhost:8000/api/status   # confirm which checkpoint + when it was loaded
```

## What "autonomous" means here, precisely

No human looks at any individual chat before it's trained on. Two automated
checks replace that review, and both can veto:

- **Execution filter** (`sandbox.py`): an exchange only becomes training
  data if its code block actually runs/compiles. Prompts with no code, or
  with code that errors out, are silently dropped.
- **Auto-eval gate** (`evaluate_checkpoint` in `autonomous_trainer.py`):
  after fine-tuning, the candidate is graded on `eval_set.json` — problems
  it never trains on — by literally running its generated code against
  test assertions. If its pass rate drops more than `--tolerance` (default
  5 points) versus the live model, the candidate is discarded and the live
  model keeps serving unchanged. Every promoted checkpoint's predecessor is
  archived to `checkpoints/archive/` for manual rollback at any time.

## Known limitations, worth knowing going in

- **GPT-2 (124M) is a weak coding base model.** Stages 1–2 will meaningfully
  improve it on simple Python tasks; don't expect strong C++ or hard-DSA
  correctness — the training plan you uploaded is explicit that this needs
  a code-pretrained base model, not a from-scratch/GPT-2 one.
- **`sandbox.py` is a correctness filter, not a hardened security sandbox.**
  Fine for your own logged chatbot traffic on a machine you control; don't
  expose it to arbitrary untrusted code without real container/VM isolation.
- **Low-traffic cold start.** If fewer than `--min_new_examples` (default 5)
  pass the filter in a 6-hour window, that cycle is skipped — no training,
  nothing wasted. Lower the threshold for testing, raise it in production.
- **The eval set is intentionally tiny (5 problems)** — enough to catch a
  badly regressed checkpoint, not a real benchmark. A "pass" means "didn't
  obviously break," not "is good at DSA now."

