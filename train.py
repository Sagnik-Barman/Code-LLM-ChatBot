"""
Pretrain the GPT model on the-verdict.txt (or any text file you point it at).

Usage:
    python train.py
    python train.py --data data/the-verdict.txt --epochs 10 --device cuda

Saves a checkpoint to checkpoints/model.pth when done, and a loss plot to
checkpoints/loss_plot.png.
"""
import argparse
import os

import matplotlib.pyplot as plt
import tiktoken
import torch
from torch.amp import autocast, GradScaler

from dataset import create_dataloader_v1
from generate import generate_text_simple, text_to_token_ids, token_ids_to_text
from model import GPT_CONFIG_124M, GPTModel


def calc_loss_batch(input_batch, target_batch, model, device, use_amp=False):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    with autocast(device_type=device.type, enabled=use_amp):
        logits = model(input_batch)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
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


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded, max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                        eval_freq, eval_iter, start_context, tokenizer,
                        scaler=None, use_amp=False):
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
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter, use_amp=use_amp)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses, out_path):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)  # invisible plot for second x-axis
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/the-verdict.txt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="checkpoints/model.pth")
    parser.add_argument("--start_context", default="Every effort moves you")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    with open(args.data, "r", encoding="utf-8") as f:
        text_data = f.read()

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    tokenizer = tiktoken.get_encoding("gpt2")

    use_amp = (device.type == "cuda")
    scaler = GradScaler(device=device.type, enabled=use_amp)
    if use_amp:
        print("Mixed precision (AMP) enabled for faster, lower-memory training.")

    # 90/10 train/val split
    split_idx = int(0.9 * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    train_loader = create_dataloader_v1(
        train_data, batch_size=args.batch_size,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True, shuffle=True, num_workers=0,
    )
    val_loader = create_dataloader_v1(
        val_data, batch_size=args.batch_size,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=False, shuffle=False, num_workers=0,
    )

    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=args.epochs, eval_freq=5, eval_iter=5,
        start_context=args.start_context, tokenizer=tokenizer,
        scaler=scaler, use_amp=use_amp,
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()}, args.out)
    print(f"Saved checkpoint to {args.out}")

    epochs_seen = torch.linspace(0, args.epochs, len(train_losses))
    plot_losses(epochs_seen, tokens_seen, train_losses, val_losses,
                "checkpoints/loss_plot.png")
    print("Saved loss plot to checkpoints/loss_plot.png")


if __name__ == "__main__":
    main()
