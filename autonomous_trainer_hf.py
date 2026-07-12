"""
Autonomous training loop for the code_model.py / lora_finetune.py pipeline
(Qwen2.5-Coder + LoRA). Same design as the original autonomous_trainer.py,
ported to this architecture -- with one real advantage: because adapters
are small, the base model loads ONCE per cycle and is reused for the
baseline eval, training, and candidate eval, rather than reloading a 3GB+
model repeatedly.

Every --cycle_hours (default 6):
  1. Harvest chat interactions logged by app_hf.py since the last cycle.
  2. Auto-filter via sandbox.py: keep only exchanges whose code actually runs.
  3. Mix filtered examples with a replay buffer (data/replay_buffer.json,
     falling back to data/code_alpaca_20k.json if not yet seeded).
  4. LoRA fine-tune a candidate adapter from the currently-live adapter.
  5. Auto-evaluate both live and candidate adapters on eval_set.json by
     actually executing the generated code.
  6. Promote (and hot-reload the live app_hf.py via /api/reload) only if
     the candidate didn't regress beyond --tolerance; otherwise discard it.

Usage:
    python autonomous_trainer_hf.py                     # run forever, every 6h
    python autonomous_trainer_hf.py --once               # single cycle (testing)
"""
import argparse
import json
import os
import random
import shutil
import time

import requests
import torch

from code_model import attach_adapter, get_live_adapter_path, load_base_model, set_live_adapter_path
from eval_checkpoint_hf import evaluate
from lora_finetune import load_instruction_json, train_lora
from sandbox import passes_execution_filter

LOG_PATH = "logs/interactions.jsonl"
STATE_PATH = "logs/trainer_state_hf.json"
REPLAY_BUFFER_PATH = "data/replay_buffer.json"
FALLBACK_REPLAY_PATH = "data/code_alpaca_20k.json"  # used if replay_buffer.json isn't seeded yet
EVAL_SET_PATH = "eval_set.json"
ADAPTER_DIR = "checkpoints/autonomous_adapters"


# ---------------------------------------------------------------------------
# State + harvesting
# ---------------------------------------------------------------------------
def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    return {"last_cycle_ts": 0, "processed_lines": 0, "cycle_count": 0}


def save_state(state):
    os.makedirs(os.path.dirname(STATE_PATH) or ".", exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def harvest_new_interactions(processed_lines):
    if not os.path.exists(LOG_PATH):
        return [], processed_lines
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
    new_lines = lines[processed_lines:]
    records = [json.loads(l) for l in new_lines if l.strip()]
    return records, len(lines)


# ---------------------------------------------------------------------------
# Filtering + dataset construction
# ---------------------------------------------------------------------------
def auto_filter(records):
    """The no-human-reviewer safety gate: execution correctness + basic hygiene."""
    kept, seen_prompts = [], set()
    for r in records:
        prompt = r.get("prompt", "").strip()
        response = r.get("response", "").strip()
        if len(prompt) < 8 or len(response) < 8:
            continue
        if prompt in seen_prompts:
            continue
        if not passes_execution_filter(response):
            continue
        seen_prompts.add(prompt)
        kept.append({"instruction": prompt, "input": "", "output": response})
    return kept


def build_training_set(new_examples, replay_fraction=0.7):
    replay_path = REPLAY_BUFFER_PATH if os.path.exists(REPLAY_BUFFER_PATH) else FALLBACK_REPLAY_PATH
    if not os.path.exists(replay_path):
        raise FileNotFoundError(
            f"Neither {REPLAY_BUFFER_PATH} nor the fallback {FALLBACK_REPLAY_PATH} exist. "
            f"Seed a replay buffer before running the autonomous trainer -- this anchors "
            f"every cycle so the model doesn't drift on thin, recent-only chat data."
        )
    replay_data = load_instruction_json(replay_path)
    print(f"Using replay data from {replay_path} ({len(replay_data)} examples)")

    if not new_examples:
        return replay_data

    n_new = len(new_examples)
    n_replay_min = int(n_new * replay_fraction / (1 - replay_fraction)) if replay_fraction < 1 else len(replay_data)
    n_replay = min(len(replay_data), max(n_replay_min, len(replay_data) // 10))

    random.seed(int(time.time()))
    replay_sample = random.sample(replay_data, n_replay) if n_replay < len(replay_data) else list(replay_data)

    # new_examples are already {"instruction", "input": "", "output"} shaped
    # (see auto_filter); load_instruction_json output has {"instruction", "output"}
    # (input already folded in) -- normalize new_examples to match before mixing.
    new_normalized = [{"instruction": ex["instruction"], "output": ex["output"]} for ex in new_examples]

    combined = replay_sample + new_normalized
    random.shuffle(combined)
    return combined


# ---------------------------------------------------------------------------
# One full cycle
# ---------------------------------------------------------------------------
def run_cycle(args):
    state = load_state()
    print(f"\n=== Autonomous training cycle #{state['cycle_count'] + 1} "
          f"at {time.strftime('%Y-%m-%d %H:%M:%S')} ===")

    records, new_processed_count = harvest_new_interactions(state["processed_lines"])
    print(f"Harvested {len(records)} raw interactions since last cycle.")

    filtered = auto_filter(records)
    print(f"{len(filtered)} passed the execution-correctness filter.")

    if len(filtered) < args.min_new_examples:
        print(f"Fewer than --min_new_examples ({args.min_new_examples}); "
              f"skipping training this cycle (live adapter unchanged).")
        state.update(processed_lines=new_processed_count,
                      cycle_count=state["cycle_count"] + 1, last_cycle_ts=time.time())
        save_state(state)
        return

    train_examples = build_training_set(filtered)
    print(f"Training on {len(train_examples)} examples (filtered + replay buffer).")

    print(f"Loading base model on {args.device} ...")
    base_model, tokenizer = load_base_model(device=args.device)

    live_adapter = get_live_adapter_path(args.initial_adapter)
    print(f"Live adapter: {live_adapter}")

    live_model = attach_adapter(base_model, live_adapter)
    base_score, _ = evaluate(live_model, tokenizer, EVAL_SET_PATH)
    print(f"Live adapter held-out pass rate:      {base_score*100:.1f}%")
    # IMPORTANT: attach_adapter() modifies base_model IN PLACE (injects LoRA
    # layers directly into its modules) -- deleting the `live_model` Python
    # reference does NOT undo that. Reusing base_model for training or the
    # candidate eval without properly unloading first causes the adapter to
    # linger, triggering PEFT's "already found a peft_config" warning and
    # risking stale adapter state accumulating across repeated cycles.
    # PeftModel.unload() is the real fix: it strips the LoRA layers back out
    # and returns the underlying base model in a genuinely clean state.
    base_model = live_model.unload()
    del live_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    candidate_path = os.path.join(ADAPTER_DIR, f"adapter_{int(time.time())}")
    _, candidate_model = train_lora(
        train_examples, base_model=base_model, tokenizer=tokenizer, out_path=candidate_path,
        resume_adapter=live_adapter, epochs=args.epochs, batch_size=args.batch_size,
        grad_accum=args.grad_accum, lr=args.lr, save_steps=args.save_steps,
        max_length=args.max_length, device=args.device,
    )
    candidate_model.eval()
    candidate_score, _ = evaluate(candidate_model, tokenizer, EVAL_SET_PATH)
    print(f"Candidate adapter held-out pass rate: {candidate_score*100:.1f}%")

    # Clean up in case this base_model reference gets reused later in a
    # future extension of this cycle (e.g. a retry-on-rejection path).
    base_model = candidate_model.unload()
    del candidate_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if candidate_score >= base_score - args.tolerance:
        set_live_adapter_path(candidate_path)
        print(f"PROMOTED candidate -> live adapter ({candidate_path}).")
        try:
            resp = requests.post(f"{args.app_url}/api/reload",
                                  json={"adapter_path": candidate_path}, timeout=60)
            if resp.ok:
                print(f"Live app hot-reloaded new adapter via {args.app_url}/api/reload.")
            else:
                print(f"Reload request to {args.app_url} failed ({resp.status_code}): {resp.text}")
        except Exception as e:
            print(f"Could not reach {args.app_url} to hot-reload ({e!r}). "
                  f"New adapter is saved and marked live, but the running app "
                  f"is still serving the old one until it's reloaded or restarted.")
    else:
        print(f"REJECTED candidate: pass rate dropped more than tolerance "
              f"({args.tolerance*100:.0f} pts). Live adapter left unchanged.")
        shutil.rmtree(candidate_path, ignore_errors=True)
        shutil.rmtree(candidate_path + "_trainer_tmp", ignore_errors=True)

    state.update(processed_lines=new_processed_count,
                  cycle_count=state["cycle_count"] + 1, last_cycle_ts=time.time())
    save_state(state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle_hours", type=float, default=6.0)
    parser.add_argument("--min_new_examples", type=int, default=5)
    parser.add_argument("--initial_adapter", default="checkpoints/code_lora_v1",
                         help="Adapter to start from if no live adapter has been promoted yet")
    parser.add_argument("--epochs", type=int, default=1,
                         help="Kept low for autonomous cycles -- this is incremental adaptation, "
                              "not a full fine-tuning pass")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--save_steps", type=int, default=25,
                         help="Lower than lora_finetune.py's CLI default -- autonomous cycles are "
                              "small (tens to low hundreds of steps), so checkpointing more often "
                              "costs little and means a crash loses less unattended progress.")
    parser.add_argument("--max_length", type=int, default=384,
                         help="Lower than lora_finetune.py's CLI default. Real logged chat responses "
                              "vary more in length than curated CodeAlpaca examples, so autonomous "
                              "cycles use a tighter cap to reduce the odds of a VRAM-related crash "
                              "on an unattended run (attention memory grows with the square of this).")
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--app_url", default="http://localhost:8000",
                         help="Base URL of the running app_hf.py, used to hot-reload after promotion")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    if args.once:
        try:
            run_cycle(args)
        except Exception as e:
            print(f"Cycle failed with error: {e!r}. Live adapter left untouched. "
                  f"Nothing was corrupted -- run_cycle() only saves progress after a cycle "
                  f"completes successfully, so the next scheduled run will simply retry.")
        return

    print(f"Autonomous trainer started. Cycle interval: {args.cycle_hours}h. Ctrl+C to stop.")
    while True:
        try:
            run_cycle(args)
        except Exception as e:
            print(f"Cycle failed with error: {e!r}. Live adapter left untouched.")
        time.sleep(args.cycle_hours * 3600)


if __name__ == "__main__":
    main()
