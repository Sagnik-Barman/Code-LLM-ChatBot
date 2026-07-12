"""
Continues pretraining REAL, already-pretrained GPT-2 weights on a custom code
corpus (e.g. data/code_corpus.txt from prepare_code_corpus.py).

This is NOT the same as train.py: train.py initializes a brand-new GPTModel
with random weights (useful for seeing the training loop mechanics work on a
tiny file, per the book's Chapter 5 exercise, but the wrong tool here).
This script instead loads OpenAI's actual GPT-2 weights via load_pretrained.py
and keeps training THOSE -- so the model doesn't have to relearn English
before it can learn code. This is Stage 1 in your training plan.

Usage:
    python continue_pretrain.py --data data/code_corpus.txt --epochs 3
    python continue_pretrain.py --data data/code_corpus.txt --device cuda --lr 1e-5
"""
import argparse
import os

import tiktoken
import torch
from torch.amp import GradScaler

from dataset import create_dataloader_v1
from generate import generate_text_simple, text_to_token_ids, token_ids_to_text
from load_pretrained import load_pretrained_gpt2
from train import calc_loss_batch, evaluate_model, plot_losses


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                        eval_freq, eval_iter, start_context, tokenizer,
                        scaler=None, use_amp=False, save_every=0, save_path=None):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
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
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device,
                                                        eval_iter, use_amp=use_amp)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

            if save_every > 0 and save_path and global_step > 0 and global_step % save_every == 0:
                os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
                torch.save({"model_state_dict": model.state_dict()}, save_path)
                print(f"  [checkpoint saved at step {global_step} -> {save_path}]")

        model.eval()
        context_size = model.pos_emb.weight.shape[0]
        encoded = text_to_token_ids(start_context, tokenizer).to(device)
        with torch.no_grad():
            token_ids = generate_text_simple(model=model, idx=encoded, max_new_tokens=60,
                                              context_size=context_size)
        print(token_ids_to_text(token_ids, tokenizer).replace("\n", " "))
        model.train()

    return train_losses, val_losses, track_tokens_seen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to the code corpus text file")
    parser.add_argument("--size", default="gpt2-small (124M)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5,
                         help="Deliberately low -- this fine-tunes real pretrained weights, "
                              "not a from-scratch model. A high LR here degrades the model's "
                              "existing language ability fast.")
    parser.add_argument("--stride", type=int, default=512, help="Sliding-window stride, in tokens")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="checkpoints/code_pretrained.pth")
    parser.add_argument("--save_every", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    with open(args.data, "r", encoding="utf-8", errors="ignore") as f:
        text_data = f.read()
    print(f"Corpus size: {len(text_data)/1e6:.1f}M characters")

    print(f"Loading pretrained {args.size} weights ...")
    model, cfg = load_pretrained_gpt2(args.size, device)
    tokenizer = tiktoken.get_encoding("gpt2")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    use_amp = (device.type == "cuda")
    scaler = GradScaler(device=device.type, enabled=use_amp)
    if use_amp:
        print("Mixed precision (AMP) enabled.")

    split_idx = int(0.95 * len(text_data))
    train_data, val_data = text_data[:split_idx], text_data[split_idx:]
    context_length = cfg["context_length"]  # 1024 -- matches the real pretrained weights

    train_loader = create_dataloader_v1(train_data, batch_size=args.batch_size,
                                         max_length=context_length, stride=args.stride,
                                         drop_last=True, shuffle=True)
    val_loader = create_dataloader_v1(val_data, batch_size=args.batch_size,
                                       max_length=context_length, stride=args.stride,
                                       drop_last=False, shuffle=False)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    if len(train_loader) == 0:
        raise SystemExit("Corpus too small for this context_length/batch_size/stride -- "
                          "build a bigger corpus or lower --stride.")

    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=args.epochs, eval_freq=20, eval_iter=5,
        start_context="def quicksort(arr):", tokenizer=tokenizer,
        scaler=scaler, use_amp=use_amp,
        save_every=args.save_every, save_path=args.out,
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, args.out)
    print(f"Saved continued-pretrained checkpoint to {args.out}")

    if train_losses:
        epochs_seen = torch.linspace(0, args.epochs, len(train_losses))
        plot_losses(epochs_seen, tokens_seen, train_losses, val_losses,
                    "checkpoints/code_pretrain_loss_plot.png")
        print("Saved loss plot to checkpoints/code_pretrain_loss_plot.png")


if __name__ == "__main__":
    main()
