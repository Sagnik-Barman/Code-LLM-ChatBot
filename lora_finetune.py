"""
LoRA fine-tunes Qwen2.5-Coder-1.5B-Instruct on an instruction dataset (same
{"instruction", "input", "output"} JSON shape as your original pipeline's
CodeAlpaca data -- you can reuse data/code_alpaca_20k.json directly).

Only a small adapter trains; the base model's weights stay frozen. This is
what makes repeated fine-tuning cycles (e.g. from autonomous_trainer_v2.py)
cheap: each cycle saves a ~20-50MB adapter, not a multi-GB checkpoint.

Usage:
    python lora_finetune.py --data data/code_alpaca_20k.json --out checkpoints/code_lora_v1 --epochs 2
    python lora_finetune.py --data data/code_alpaca_20k.json --resume_adapter checkpoints/code_lora_v1 --out checkpoints/code_lora_v2
"""
import argparse
import json

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments

from code_model import BASE_MODEL, format_chat_prompt


class MaskedLabelCollator:
    """
    Pads input_ids/attention_mask/labels for a batch.

    IMPORTANT: transformers' built-in DataCollatorForLanguageModeling(mlm=False)
    always OVERWRITES the "labels" field with an unmasked clone of input_ids,
    discarding any custom prompt-masking. That would train the model to also
    predict the instruction/prompt tokens, not just the response -- silently
    diluting the training signal. This collator instead pads the "labels"
    field we already built (with -100 over the prompt) and leaves it alone.
    """
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        max_len = max(len(ex["input_ids"]) for ex in examples)
        input_ids, attention_mask, labels = [], [], []
        for ex in examples:
            pad_len = max_len - len(ex["input_ids"])
            input_ids.append(ex["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append(ex["attention_mask"] + [0] * pad_len)
            labels.append(ex["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels),
        }


def load_instruction_json(path):
    with open(path) as f:
        data = json.load(f)
    # Fold "input" into the instruction text, same convention as the
    # original pipeline's Alpaca-style formatting.
    examples = []
    for entry in data:
        instruction = entry["instruction"]
        if entry.get("input"):
            instruction = f"{instruction}\n\nInput: {entry['input']}"
        examples.append({"instruction": instruction, "output": entry["output"]})
    return examples


def build_dataset(examples, tokenizer, max_length=512):
    def tokenize(example):
        prompt = format_chat_prompt(tokenizer, example["instruction"])
        full_text = prompt + example["output"] + tokenizer.eos_token
        tokenized = tokenizer(full_text, truncation=True, max_length=max_length)
        # Mask the prompt portion out of the loss -- only train on the
        # response, same principle as custom_collate_fn in the original
        # instruction_finetune.py.
        prompt_len = len(tokenizer(prompt, truncation=True, max_length=max_length)["input_ids"])
        labels = list(tokenized["input_ids"])
        labels[:prompt_len] = [-100] * prompt_len
        tokenized["labels"] = labels
        return tokenized

    ds = Dataset.from_list(examples)
    return ds.map(tokenize, remove_columns=ds.column_names)


def train_lora(examples, base_model, tokenizer, out_path, resume_adapter=None,
                epochs=2, batch_size=2, grad_accum=8, lr=1e-4, save_steps=100,
                max_length=512, resume_from_checkpoint=None, device="cuda"):
    """
    Core LoRA training routine, reusable by both the CLI below and
    autonomous_trainer_hf.py (which loads the base model once and calls
    this repeatedly across cycles, rather than reloading a 3GB+ model
    every time).

    base_model: an already-loaded AutoModelForCausalLM (see code_model.load_base_model).
    examples: list of {"instruction", "input", "output"} dicts (same shape
              loaded by load_instruction_json).
    resume_adapter: path to an existing adapter to continue training from,
              or None to start a fresh adapter from the base model.
    Returns the path the adapter was saved to (== out_path).
    """
    if resume_adapter:
        print(f"Continuing from existing adapter: {resume_adapter}")
        model = PeftModel.from_pretrained(base_model, resume_adapter, is_trainable=True)
    else:
        lora_config = LoraConfig(
            task_type="CAUSAL_LM", r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    dataset = build_dataset(examples, tokenizer, max_length=max_length)

    training_args = TrainingArguments(
        output_dir=out_path + "_trainer_tmp",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        logging_steps=20,
        # No in-training eval: the metric that actually matters is functional
        # pass rate on eval_set.json (via eval_checkpoint_hf.py), not LM loss
        # on held-out text -- and running an eval pass mid-training was the
        # exact point that triggered a CUDA OOM (different batch/seq-length
        # profile than training batches, on top of already-tight VRAM).
        eval_strategy="no",
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,  # keep last 2 checkpoints so a crash doesn't cost the whole run
        bf16=(device == "cuda"),
        report_to="none",
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=dataset,
        data_collator=MaskedLabelCollator(pad_token_id=tokenizer.pad_token_id),
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(out_path)
    tokenizer.save_pretrained(out_path)
    print(f"Saved LoRA adapter to {out_path}")
    return out_path, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--base_model", default=BASE_MODEL)
    parser.add_argument("--resume_adapter", default=None,
                         help="Path to an existing LoRA adapter to continue training from")
    parser.add_argument("--out", default="checkpoints/code_lora_v1")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_length", type=int, default=512,
                         help="Token cap per example. Lower this if you hit CUDA errors mid-training "
                              "(cuBLAS failures, OOM) on longer examples -- attention memory grows "
                              "with the square of this number.")
    parser.add_argument("--save_steps", type=int, default=100,
                         help="Save a checkpoint every N steps so a crash doesn't lose the whole run")
    parser.add_argument("--resume_from_checkpoint", default=None,
                         help="Path to a trainer checkpoint dir (e.g. checkpoints/code_lora_v1_trainer_tmp/checkpoint-100) to resume an interrupted run")
    parser.add_argument("--use_4bit", action="store_true", default=True)
    parser.add_argument("--no_4bit", dest="use_4bit", action="store_false")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if args.use_4bit and args.device == "cuda":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=quant_config,
        dtype=torch.bfloat16 if args.device == "cuda" else torch.float32,
        device_map=args.device if args.device == "cuda" else None,
    )

    examples = load_instruction_json(args.data)
    print(f"Loaded {len(examples)} training examples from {args.data}")

    train_lora(
        examples, base_model=model, tokenizer=tokenizer, out_path=args.out,
        resume_adapter=args.resume_adapter, epochs=args.epochs, batch_size=args.batch_size,
        grad_accum=args.grad_accum, lr=args.lr, save_steps=args.save_steps, max_length=args.max_length,
        resume_from_checkpoint=args.resume_from_checkpoint, device=args.device,
    )


if __name__ == "__main__":
    main()
