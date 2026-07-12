"""
Instruction fine-tuning (Alpaca-style), chapter 7.

Uses the same instruction/input/output JSON format popularized by Stanford
Alpaca (https://github.com/tatsu-lab/stanford_alpaca) and used in rasbt's
instruction-data.json (https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/01_main-chapter-code/instruction-data.json).

Usage:
    python instruction_finetune.py --dataset alpaca --epochs 3
    python instruction_finetune.py --prompt "Convert 10 miles to kilometers."
    python instruction_finetune.py --dataset alpaca --resume checkpoints/instruction_model_step12000.pth
"""
import argparse
import json
import os
import time
import urllib.request

import tiktoken
import torch
from torch.amp import autocast, GradScaler
from functools import partial
from torch.utils.data import Dataset, DataLoader

from generate import generate, text_to_token_ids, token_ids_to_text
from load_pretrained import load_pretrained_gpt2

DATASETS = {
    "rasbt": {
        "path": "data/instruction-data.json",
        "url": (
            "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/"
            "ch07/01_main-chapter-code/instruction-data.json"
        ),
    },
    "alpaca": {
        "path": "data/alpaca_data.json",
        "url": "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json",
    },
    "codealpaca": {
        "path": "data/code_alpaca_20k.json",
        "url": "https://raw.githubusercontent.com/sahil280114/codealpaca/master/data/code_alpaca_20k.json",
        # 20k coding instruction/input/output triples -- same JSON shape as the
        # datasets above, so no code changes are needed to use it here.
    },
}


# ---------------------------------------------------------------------------
# Data download + Alpaca-style prompt formatting
# ---------------------------------------------------------------------------
def download_instruction_data(dataset_path, dataset_url):
    if os.path.exists(dataset_path):
        print(f"{dataset_path} already exists, skipping download.")
        return
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    print(f"Downloading {dataset_url} ...")
    urllib.request.urlretrieve(dataset_url, dataset_path)


def format_input(entry):
    """Alpaca-style prompt template."""
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry.get("input") else ""
    return instruction_text + input_text


def load_instruction_data(dataset_path, train_frac=0.85, val_frac=0.05):
    with open(dataset_path, "r") as f:
        data = json.load(f)
    train_portion = int(len(data) * train_frac)
    val_portion = int(len(data) * val_frac)
    train_data = data[:train_portion]
    val_data = data[train_portion:train_portion + val_portion]
    test_data = data[train_portion + val_portion:]
    return train_data, val_data, test_data


# ---------------------------------------------------------------------------
# Dataset + collate function (dynamic padding per batch, masked loss)
# ---------------------------------------------------------------------------
class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(tokenizer.encode(full_text))

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


def custom_collate_fn(batch, pad_token_id=50256, ignore_index=-100,
                       allowed_max_length=None, device="cpu"):
    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]  # append one EOS token

        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        # Mask all but the first padding token in the loss
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor


# ---------------------------------------------------------------------------
# Training / evaluation (reuses the same loss machinery as pretraining, but
# the collate function already encodes -100 ignore_index for padding)
# ---------------------------------------------------------------------------
def calc_loss_batch(input_batch, target_batch, model, device, use_amp=False):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    with autocast(device_type=device.type, enabled=use_amp):
        logits = model(input_batch)
        loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), target_batch.flatten(), ignore_index=-100
        )
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None, use_amp=False):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    num_batches = len(data_loader) if num_batches is None else min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        loss = calc_loss_batch(input_batch, target_batch, model, device, use_amp=use_amp)
        total_loss += loss.item()
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter, use_amp=False):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter, use_amp=use_amp)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter, use_amp=use_amp)
    model.train()
    return train_loss, val_loss


def train_model_simple(model, train_loader, val_loader, optimizer, device,
                        num_epochs, eval_freq, eval_iter, start_text, tokenizer, context_size,
                        scaler=None, use_amp=False, save_every=0, save_path=None,
                        start_step=0, start_epoch=0):
    train_losses, val_losses = [], []
    tokens_seen, global_step = 0, start_step - 1
    total_steps = num_epochs * len(train_loader)
    t_start = time.time()

    for epoch in range(start_epoch, num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device, use_amp=use_amp)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter, use_amp=use_amp)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                elapsed = time.time() - t_start
                steps_done = global_step - start_step + 1
                steps_per_sec = steps_done / elapsed if elapsed > 0 else 0
                remaining = total_steps - global_step
                eta_min = (remaining / steps_per_sec / 60) if steps_per_sec > 0 else float("nan")
                print(f"Ep {epoch+1} (Step {global_step:06d}/{total_steps}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f} "
                      f"| ETA {eta_min:.0f} min")

            if save_every > 0 and save_path and global_step > 0 and global_step % save_every == 0:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "global_step": global_step,
                    "epoch": epoch,
                }, save_path)
                print(f"  [checkpoint saved at step {global_step} -> {save_path}]")

        model.eval()
        token_ids = generate(
            model=model, idx=text_to_token_ids(start_text, tokenizer).to(device),
            max_new_tokens=50, context_size=context_size, temperature=0.0,
            eos_id=50256, no_repeat_ngram_size=3,
        )
        print(token_ids_to_text(token_ids, tokenizer).replace("\n", " "))
        model.train()

        # End-of-epoch checkpoint regardless of save_every, so a resume is
        # always possible at an epoch boundary at minimum.
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
                "epoch": epoch + 1,
            }, save_path)
            print(f"  [end-of-epoch checkpoint saved -> {save_path}]")

    return train_losses, val_losses, tokens_seen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--size", default="gpt2-small (124M)",
                         help="124M fits comfortably on 6-8GB GPUs fully fine-tuned; "
                              "355M+ needs more VRAM or gradient checkpointing")
    parser.add_argument("--max_length", type=int, default=512,
                         help="Caps tokens per example. Attention memory grows with the "
                              "square of this number, so lowering it is the single biggest VRAM lever.")
    parser.add_argument("--dataset", default="rasbt", choices=list(DATASETS.keys()),
                         help="'rasbt' = 1,100 curated examples (fast). "
                              "'alpaca' = full 52,002-example Stanford Alpaca set (slow, better quality).")
    parser.add_argument("--eval_freq", type=int, default=20,
                         help="How often (in steps) to print loss + ETA. Raise this for large datasets "
                              "so logging doesn't dominate runtime.")
    parser.add_argument("--save_every", type=int, default=0,
                         help="Save a mid-epoch checkpoint every N steps (0 = only save at epoch end). "
                              "Recommended for long runs (e.g. --dataset alpaca) in case of interruption.")
    parser.add_argument("--resume", default=None,
                         help="Path to a checkpoint to resume training from (e.g. checkpoints/instruction_model.pth)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="checkpoints/instruction_model.pth")
    parser.add_argument("--prompt", default=None, help="Skip training; run inference with a saved checkpoint")
    args = parser.parse_args()

    device = torch.device(args.device)
    tokenizer = tiktoken.get_encoding("gpt2")

    gpt, cfg = load_pretrained_gpt2(args.size, device)

    if args.prompt is not None:
        checkpoint = torch.load(args.out, map_location=device)
        gpt.load_state_dict(checkpoint["model_state_dict"])
        gpt.eval()
        entry = {"instruction": args.prompt, "input": ""}
        input_text = format_input(entry)
        token_ids = generate(
            model=gpt, idx=text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens=150, context_size=cfg["context_length"], temperature=0.0,
            eos_id=50256, no_repeat_ngram_size=3,
        )
        full_text = token_ids_to_text(token_ids, tokenizer)
        response = full_text[len(input_text):].replace("### Response:", "").strip()
        print(response)
        return

    download_instruction_data(DATASETS[args.dataset]["path"], DATASETS[args.dataset]["url"])
    train_data, val_data, test_data = load_instruction_data(DATASETS[args.dataset]["path"])
    print(f"Dataset: {args.dataset} | Train/val/test sizes: {len(train_data)}/{len(val_data)}/{len(test_data)}")

    customized_collate_fn = partial(custom_collate_fn, device=device, allowed_max_length=args.max_length)

    train_loader = DataLoader(
        InstructionDataset(train_data, tokenizer), batch_size=args.batch_size,
        collate_fn=customized_collate_fn, shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        InstructionDataset(val_data, tokenizer), batch_size=args.batch_size,
        collate_fn=customized_collate_fn, shuffle=False, drop_last=False,
    )

    optimizer = torch.optim.AdamW(gpt.parameters(), lr=args.lr, weight_decay=0.1)
    use_amp = (device.type == "cuda")
    scaler = GradScaler(device=device.type, enabled=use_amp)
    if use_amp:
        print("Mixed precision (AMP) enabled for faster, lower-memory training.")

    start_epoch, start_step = 0, 0
    if args.resume is not None:
        print(f"Resuming from {args.resume} ...")
        checkpoint = torch.load(args.resume, map_location=device)
        gpt.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0)
        start_step = checkpoint.get("global_step", 0)
        print(f"Resumed at epoch {start_epoch}, step {start_step}. "
              f"(Resume restarts the in-progress epoch from its beginning if it wasn't fully completed.)")

    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    print(f"Steps per epoch: {steps_per_epoch} | Total planned steps: {total_steps}")

    train_model_simple(
        gpt, train_loader, val_loader, optimizer, device,
        num_epochs=args.epochs, eval_freq=args.eval_freq, eval_iter=5,
        start_text=format_input(val_data[0]), tokenizer=tokenizer,
        context_size=cfg["context_length"],
        scaler=scaler, use_amp=use_amp,
        save_every=args.save_every, save_path=args.out,
        start_step=start_step, start_epoch=start_epoch,
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"model_state_dict": gpt.state_dict()}, args.out)
    print(f"Saved instruction-tuned model to {args.out}")


if __name__ == "__main__":
    main()
