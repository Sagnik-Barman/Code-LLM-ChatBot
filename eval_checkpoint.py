"""
Standalone baseline evaluator. Runs the eval_set.json held-out DSA problems
against any checkpoint and reports an actual pass rate (by executing the
generated code against real test assertions) -- not a vibe check.

Run this once after Stage 2, before turning on autonomous_trainer.py, so you
have a real baseline number to compare future checkpoints against.

Usage:
    python eval_checkpoint.py --checkpoint checkpoints/instruction_model.pth
    python eval_checkpoint.py --checkpoint checkpoints/instruction_model.pth --verbose
"""
import argparse
import json

import tiktoken
import torch

from generate import generate, text_to_token_ids, token_ids_to_text
from instruction_finetune import format_input
from load_pretrained import load_pretrained_gpt2
from sandbox import extract_code_blocks, run_python


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval_set", default="eval_set.json")
    parser.add_argument("--size", default="gpt2-small (124M)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--verbose", action="store_true",
                         help="Print each problem's generated code and result, not just the score")
    args = parser.parse_args()

    device = torch.device(args.device)
    tokenizer = tiktoken.get_encoding("gpt2")

    print(f"Loading base {args.size} architecture ...")
    gpt, cfg = load_pretrained_gpt2(args.size, device)

    print(f"Loading fine-tuned weights from {args.checkpoint} ...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    gpt.load_state_dict(checkpoint["model_state_dict"])
    gpt.eval()

    with open(args.eval_set) as f:
        problems = json.load(f)

    passed = 0
    for problem in problems:
        entry = {"instruction": problem["instruction"], "input": ""}
        input_text = format_input(entry)
        token_ids = generate(
            model=gpt, idx=text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens=args.max_new_tokens, context_size=cfg["context_length"],
            temperature=0.0, eos_id=50256, no_repeat_ngram_size=3,
        )
        full_text = token_ids_to_text(token_ids, tokenizer)
        response = full_text[len(input_text):].replace("### Response:", "").strip()

        blocks = extract_code_blocks(response, language={"python", "py"})
        result = "NO CODE BLOCK FOUND"
        ok = False
        if blocks:
            _, code = blocks[0]
            ok, stdout, stderr = run_python(code + "\n" + problem["test_code"])
            result = "PASS" if (ok and "OK" in stdout) else f"FAIL ({stderr.strip() or 'assertion failed'})"
            ok = ok and "OK" in stdout

        if ok:
            passed += 1

        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {problem['id']}" + (f" -- {result}" if not ok else ""))
        if args.verbose:
            print("  Generated response:")
            print("  " + response.replace("\n", "\n  "))
            print()

    print(f"\nPass rate: {passed}/{len(problems)} ({passed/len(problems)*100:.0f}%)")


if __name__ == "__main__":
    main()
