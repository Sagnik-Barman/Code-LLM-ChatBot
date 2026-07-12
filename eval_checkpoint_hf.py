"""
Standalone baseline evaluator for the code_model.py / lora_finetune.py
pipeline (Qwen2.5-Coder + LoRA). Runs eval_set.json against the base model
plus your trained adapter, executes the generated code for real, and
reports an honest pass rate -- the HF-pipeline equivalent of the original
eval_checkpoint.py, which only works with the from-scratch GPT-2 model.

Usage:
    python eval_checkpoint_hf.py --adapter checkpoints/code_lora_v1
    python eval_checkpoint_hf.py --adapter checkpoints/code_lora_v1 --verbose
    python eval_checkpoint_hf.py --no_adapter   # evaluate the raw base model, for comparison
"""
import argparse
import json

from code_model import load_model, generate_response
from sandbox import extract_code_blocks, run_python


def evaluate(model, tokenizer, eval_set_path, max_new_tokens=400, verbose=False):
    """
    Runs eval_set.json problems against an already-loaded model, executes
    the generated code for real, and returns (pass_rate, details).
    Reusable by both this script's CLI and autonomous_trainer_hf.py, which
    needs to score both the live and candidate adapters each cycle without
    reloading the model from scratch.
    """
    with open(eval_set_path) as f:
        problems = json.load(f)

    passed = 0
    details = []
    for problem in problems:
        response = generate_response(model, tokenizer, problem["instruction"],
                                      max_new_tokens=max_new_tokens, temperature=0.0)

        blocks = extract_code_blocks(response, language={"python", "py"})
        ok, result = False, "NO CODE FOUND"
        if blocks:
            _, code = blocks[0]
            run_ok, stdout, stderr = run_python(code + "\n" + problem["test_code"])
            ok = run_ok and "OK" in stdout
            result = "PASS" if ok else f"FAIL ({stderr.strip()[:150] or 'assertion failed'})"

        if ok:
            passed += 1
        details.append({"id": problem["id"], "passed": ok, "response": response, "result": result})

        print(f"[{'PASS' if ok else 'FAIL'}] {problem['id']}" + ("" if ok else f" -- {result}"))
        if verbose:
            print("  Generated response:")
            print("  " + response.replace("\n", "\n  "))
            print()

    pass_rate = passed / len(problems) if problems else 0.0
    return pass_rate, details


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default=None, help="Path to a trained LoRA adapter dir")
    parser.add_argument("--no_adapter", action="store_true",
                         help="Evaluate the raw base model with no adapter, for comparison")
    parser.add_argument("--eval_set", default="eval_set.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=400)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    adapter_path = None if args.no_adapter else args.adapter
    print(f"Loading model{' with adapter ' + adapter_path if adapter_path else ' (base only)'} ...")
    model, tokenizer = load_model(adapter_path=adapter_path, device=args.device)

    pass_rate, _ = evaluate(model, tokenizer, args.eval_set,
                             max_new_tokens=args.max_new_tokens, verbose=args.verbose)
    n = len(json.load(open(args.eval_set)))
    print(f"\nPass rate: {round(pass_rate*n)}/{n} ({pass_rate*100:.0f}%)")


if __name__ == "__main__":
    main()
