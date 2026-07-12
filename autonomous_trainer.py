"""
Autonomous training loop for the coding chatbot.

Every `--cycle_hours` (default 6), this script:
  1. Harvests chat interactions logged by app.py since the last cycle.
  2. Auto-filters them: an exchange is only kept if its code block actually
     runs (Python) or compiles+runs (C++). This is the no-human-required
     safety gate -- see sandbox.py.
  3. Mixes the filtered new examples with a fixed "replay buffer" of known-
     good instruction data, so the model doesn't drift/forget on a thin diet
     of only-recent chats.
  4. Fine-tunes a *candidate* checkpoint starting from the currently-live one.
  5. Auto-evaluates both the live and candidate checkpoints on a frozen,
     hand-written held-out problem set (eval_set.json) by actually running
     the generated code against test assertions.
  6. Promotes the candidate to live only if it didn't regress beyond
     `--tolerance`; otherwise discards it and leaves the live model untouched.

This intentionally never trains and promotes in one unconditional step --
"autonomous" here means "no human reviews each cycle," not "no safety check
before the model users talk to is changed."

Usage:
    python autonomous_trainer.py                      # run forever, every 6h
    python autonomous_trainer.py --once                # run a single cycle (testing/cron)
    python autonomous_trainer.py --cycle_hours 6 --min_new_examples 5
"""
import argparse
import json
import os
import random
import shutil
import time
from functools import partial

import tiktoken
import torch
import requests
from torch.utils.data import DataLoader

from generate import generate, text_to_token_ids, token_ids_to_text
from instruction_finetune import (
    InstructionDataset, custom_collate_fn, format_input, train_model_simple,
)
from load_pretrained import load_pretrained_gpt2
from sandbox import extract_code_blocks, passes_execution_filter, run_python

LOG_PATH = "logs/interactions.jsonl"
STATE_PATH = "logs/trainer_state.json"
REPLAY_BUFFER_PATH = "data/replay_buffer.json"
EVAL_SET_PATH = "eval_set.json"
LIVE_CHECKPOINT = "checkpoints/instruction_model.pth"
CANDIDATE_CHECKPOINT = "checkpoints/instruction_model_candidate.pth"
ARCHIVE_DIR = "checkpoints/archive"


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


def build_training_set(new_examples, replay_path, replay_fraction=0.7):
    """
    Mix new (filtered) examples with a fixed replay buffer of curated
    instruction data. replay_fraction is the approximate minimum share of
    replay data in the resulting mix, to resist drift/forgetting.
    """
    if not os.path.exists(replay_path):
        raise FileNotFoundError(
            f"{replay_path} not found. Seed it with a curated instruction "
            f"dataset (same format as data/instruction-data.json) before "
            f"running the autonomous trainer -- this is the anchor that "
            f"keeps the model from drifting on thin, recent-only data."
        )
    with open(replay_path) as f:
        replay_data = json.load(f)

    if not new_examples:
        return replay_data

    n_new = len(new_examples)
    n_replay_min = int(n_new * replay_fraction / (1 - replay_fraction)) if replay_fraction < 1 else len(replay_data)
    n_replay = min(len(replay_data), max(n_replay_min, len(replay_data) // 10))

    random.seed(int(time.time()))
    replay_sample = random.sample(replay_data, n_replay) if n_replay < len(replay_data) else list(replay_data)

    combined = replay_sample + new_examples
    random.shuffle(combined)
    return combined


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------
def fine_tune_candidate(train_examples, device, base_checkpoint, out_checkpoint,
                         size, epochs, batch_size, lr):
    tokenizer = tiktoken.get_encoding("gpt2")
    gpt, cfg = load_pretrained_gpt2(size, device)

    base_ckpt = torch.load(base_checkpoint, map_location=device)
    gpt.load_state_dict(base_ckpt["model_state_dict"])

    val_split = max(1, len(train_examples) // 10)
    val_data = train_examples[:val_split]
    train_data = train_examples[val_split:] or train_examples

    collate = partial(custom_collate_fn, device=device, allowed_max_length=512)
    train_loader = DataLoader(InstructionDataset(train_data, tokenizer), batch_size=batch_size,
                               collate_fn=collate, shuffle=True, drop_last=False)
    val_loader = DataLoader(InstructionDataset(val_data, tokenizer), batch_size=batch_size,
                             collate_fn=collate, shuffle=False, drop_last=False)

    optimizer = torch.optim.AdamW(gpt.parameters(), lr=lr, weight_decay=0.1)

    train_model_simple(
        gpt, train_loader, val_loader, optimizer, device,
        num_epochs=epochs, eval_freq=max(1, len(train_loader) // 2), eval_iter=2,
        start_text=format_input(val_data[0]), tokenizer=tokenizer,
        context_size=cfg["context_length"],
    )

    os.makedirs(os.path.dirname(out_checkpoint) or ".", exist_ok=True)
    torch.save({"model_state_dict": gpt.state_dict()}, out_checkpoint)
    return gpt, cfg, tokenizer


# ---------------------------------------------------------------------------
# Auto-evaluation (execution-based, no human grading)
# ---------------------------------------------------------------------------
def evaluate_checkpoint(gpt, cfg, tokenizer, device, eval_set_path):
    with open(eval_set_path) as f:
        problems = json.load(f)

    passed = 0
    for problem in problems:
        entry = {"instruction": problem["instruction"], "input": ""}
        input_text = format_input(entry)
        token_ids = generate(
            model=gpt, idx=text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens=200, context_size=cfg["context_length"], temperature=0.0,
            eos_id=50256, no_repeat_ngram_size=3,
        )
        full_text = token_ids_to_text(token_ids, tokenizer)
        response = full_text[len(input_text):].replace("### Response:", "").strip()

        blocks = extract_code_blocks(response, language={"python", "py"})
        if not blocks:
            continue
        _, code = blocks[0]
        ok, stdout, _ = run_python(code + "\n" + problem["test_code"])
        if ok and "OK" in stdout:
            passed += 1

    return passed / len(problems) if problems else 0.0


# ---------------------------------------------------------------------------
# One full cycle
# ---------------------------------------------------------------------------
def run_cycle(args):
    device = torch.device(args.device)
    state = load_state()

    print(f"\n=== Autonomous training cycle #{state['cycle_count'] + 1} "
          f"at {time.strftime('%Y-%m-%d %H:%M:%S')} ===")

    records, new_processed_count = harvest_new_interactions(state["processed_lines"])
    print(f"Harvested {len(records)} raw interactions since last cycle.")

    filtered = auto_filter(records)
    print(f"{len(filtered)} passed the execution-correctness filter.")

    if len(filtered) < args.min_new_examples:
        print(f"Fewer than --min_new_examples ({args.min_new_examples}); "
              f"skipping training this cycle (live checkpoint unchanged).")
        state.update(processed_lines=new_processed_count,
                      cycle_count=state["cycle_count"] + 1, last_cycle_ts=time.time())
        save_state(state)
        return

    train_examples = build_training_set(filtered, REPLAY_BUFFER_PATH)
    print(f"Training on {len(train_examples)} examples (filtered + replay buffer).")

    tokenizer = tiktoken.get_encoding("gpt2")
    base_gpt, cfg = load_pretrained_gpt2(args.size, device)
    base_gpt.load_state_dict(torch.load(LIVE_CHECKPOINT, map_location=device)["model_state_dict"])
    base_score = evaluate_checkpoint(base_gpt, cfg, tokenizer, device, EVAL_SET_PATH)
    print(f"Live checkpoint held-out pass rate:      {base_score*100:.1f}%")
    del base_gpt

    candidate_gpt, cfg, tokenizer = fine_tune_candidate(
        train_examples, device, LIVE_CHECKPOINT, CANDIDATE_CHECKPOINT,
        size=args.size, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
    )
    candidate_score = evaluate_checkpoint(candidate_gpt, cfg, tokenizer, device, EVAL_SET_PATH)
    print(f"Candidate checkpoint held-out pass rate: {candidate_score*100:.1f}%")

    if candidate_score >= base_score - args.tolerance:
        os.makedirs(ARCHIVE_DIR, exist_ok=True)
        archive_path = os.path.join(ARCHIVE_DIR, f"model_{int(time.time())}.pth")
        if os.path.exists(LIVE_CHECKPOINT):
            shutil.copy(LIVE_CHECKPOINT, archive_path)
        shutil.copy(CANDIDATE_CHECKPOINT, LIVE_CHECKPOINT)
        print(f"PROMOTED candidate -> live checkpoint. Previous version archived to {archive_path}.")

        try:
            resp = requests.post(f"{args.app_url}/api/reload", timeout=60)
            if resp.ok:
                print(f"Live app hot-reloaded new weights via {args.app_url}/api/reload -- "
                      f"no restart needed.")
            else:
                print(f"Reload request to {args.app_url} failed ({resp.status_code}): {resp.text}")
        except Exception as e:
            print(f"Could not reach {args.app_url} to hot-reload ({e!r}). "
                  f"The new weights are saved but app.py is still serving the old ones "
                  f"until it's restarted or reloaded manually.")
    else:
        print(f"REJECTED candidate: pass rate dropped more than tolerance "
              f"({args.tolerance*100:.0f} pts). Live checkpoint left unchanged.")

    state.update(processed_lines=new_processed_count,
                  cycle_count=state["cycle_count"] + 1, last_cycle_ts=time.time())
    save_state(state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle_hours", type=float, default=6.0)
    parser.add_argument("--min_new_examples", type=int, default=5,
                         help="Skip a cycle rather than fine-tune on too little filtered data")
    parser.add_argument("--size", default="gpt2-small (124M)")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--tolerance", type=float, default=0.05,
                         help="Allowed drop in held-out pass rate before a candidate is rejected")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--app_url", default="http://localhost:8000",
                         help="Base URL of the running app.py, used to hot-reload weights after promotion")
    parser.add_argument("--once", action="store_true", help="Run a single cycle and exit")
    args = parser.parse_args()

    if args.once:
        run_cycle(args)
        return

    print(f"Autonomous trainer started. Cycle interval: {args.cycle_hours}h. Ctrl+C to stop.")
    while True:
        try:
            run_cycle(args)
        except Exception as e:
            print(f"Cycle failed with error: {e!r}. Live checkpoint left untouched.")
        time.sleep(args.cycle_hours * 3600)


if __name__ == "__main__":
    main()
