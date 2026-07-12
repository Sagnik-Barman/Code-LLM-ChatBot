"""
Fine-tune a pretrained GPT-2 model as a binary spam/ham text classifier,
using the UCI SMS Spam Collection dataset.
https://archive.ics.uci.edu/dataset/228/sms+spam+collection

Usage:
    python classify_finetune.py --epochs 5
    python classify_finetune.py --text "You won a free prize! Click here now!!!"
"""
import argparse
import os
import urllib.request
import zipfile

import pandas as pd
import tiktoken
import torch
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader

from load_pretrained import load_pretrained_gpt2

DATA_DIR = "data/sms_spam"
ZIP_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
ZIP_PATH = os.path.join(DATA_DIR, "sms_spam_collection.zip")
EXTRACTED_FILE = os.path.join(DATA_DIR, "SMSSpamCollection")


# ---------------------------------------------------------------------------
# Data download + preprocessing
# ---------------------------------------------------------------------------
def download_and_unzip_spam_data():
    if os.path.exists(EXTRACTED_FILE):
        print(f"{EXTRACTED_FILE} already exists, skipping download.")
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Downloading {ZIP_URL} ...")
    urllib.request.urlretrieve(ZIP_URL, ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)
    # The archive extracts to a file literally named "SMSSpamCollection" (no extension)
    print(f"Extracted to {DATA_DIR}")


def load_spam_dataframe():
    df = pd.read_csv(EXTRACTED_FILE, sep="\t", header=None, names=["Label", "Text"])
    return df


def create_balanced_dataset(df):
    """Undersample the majority class (ham) so both classes are equally represented."""
    num_spam = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])
    return balanced_df


def random_split(df, train_frac=0.7, validation_frac=0.1):
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]
    return train_df, validation_df, test_df


def prepare_spam_csvs():
    download_and_unzip_spam_data()
    df = load_spam_dataframe()
    balanced_df = create_balanced_dataset(df)
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})
    train_df, val_df, test_df = random_split(balanced_df)
    train_df.to_csv(os.path.join(DATA_DIR, "train.csv"), index=None)
    val_df.to_csv(os.path.join(DATA_DIR, "validation.csv"), index=None)
    test_df.to_csv(os.path.join(DATA_DIR, "test.csv"), index=None)
    print(f"Train/val/test sizes: {len(train_df)}/{len(val_df)}/{len(test_df)}")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)
        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]

        if max_length is None:
            self.max_length = max(len(t) for t in self.encoded_texts)
        else:
            self.max_length = max_length
            self.encoded_texts = [t[:self.max_length] for t in self.encoded_texts]

        self.encoded_texts = [
            t + [pad_token_id] * (self.max_length - len(t)) for t in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.data)


# ---------------------------------------------------------------------------
# Model setup: swap the LM head for a 2-class classification head
# ---------------------------------------------------------------------------
def setup_classifier(gpt, cfg, num_classes=2, freeze_base=True):
    if freeze_base:
        for param in gpt.parameters():
            param.requires_grad = False

    gpt.out_head = torch.nn.Linear(cfg["emb_dim"], num_classes)

    # Following the book: unfreeze the last transformer block + final norm
    # so the model can still adapt, while the rest stays frozen (fast + less data needed)
    for param in gpt.trf_blocks[-1].parameters():
        param.requires_grad = True
    for param in gpt.final_norm.parameters():
        param.requires_grad = True

    return gpt


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------
def calc_accuracy_loader(data_loader, model, device, num_batches=None, use_amp=False):
    model.eval()
    correct_predictions, num_examples = 0, 0
    num_batches = len(data_loader) if num_batches is None else min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)
        with torch.no_grad(), autocast(device_type=device.type, enabled=use_amp):
            logits = model(input_batch)[:, -1, :]  # last token's logits
        predicted_labels = torch.argmax(logits, dim=-1)
        num_examples += predicted_labels.shape[0]
        correct_predictions += (predicted_labels == target_batch).sum().item()
    return correct_predictions / num_examples


def calc_loss_batch(input_batch, target_batch, model, device, use_amp=False):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    with autocast(device_type=device.type, enabled=use_amp):
        logits = model(input_batch)[:, -1, :]
        loss = torch.nn.functional.cross_entropy(logits, target_batch)
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


def train_classifier_simple(model, train_loader, val_loader, optimizer, device,
                             num_epochs, eval_freq, eval_iter, scaler=None, use_amp=False):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

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
            examples_seen += input_batch.shape[0]
            global_step += 1

            if global_step % eval_freq == 0:
                model.eval()
                with torch.no_grad():
                    train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter, use_amp=use_amp)
                    val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter, use_amp=use_amp)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
                model.train()

        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter, use_amp=use_amp)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter, use_amp=use_amp)
        print(f"Training accuracy: {train_accuracy*100:.2f}% | Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen


def classify_review(text, model, tokenizer, device, max_length, pad_token_id=50256):
    model.eval()
    input_ids = tokenizer.encode(text)[:max_length]
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)
    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]
    predicted_label = torch.argmax(logits, dim=-1).item()
    return "spam" if predicted_label == 1 else "ham"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--size", default="gpt2-small (124M)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="checkpoints/spam_classifier.pth")
    parser.add_argument("--text", default=None, help="Skip training; classify this text using a saved checkpoint")
    args = parser.parse_args()

    device = torch.device(args.device)
    tokenizer = tiktoken.get_encoding("gpt2")

    gpt, cfg = load_pretrained_gpt2(args.size, device)
    gpt = setup_classifier(gpt, cfg)
    gpt.to(device)

    if args.text is not None:
        checkpoint = torch.load(args.out, map_location=device)
        gpt.load_state_dict(checkpoint["model_state_dict"])
        max_length = checkpoint["max_length"]
        label = classify_review(args.text, gpt, tokenizer, device, max_length)
        print(f"Prediction: {label}")
        return

    prepare_spam_csvs()

    train_ds = SpamDataset(os.path.join(DATA_DIR, "train.csv"), tokenizer)
    val_ds = SpamDataset(os.path.join(DATA_DIR, "validation.csv"), tokenizer, max_length=train_ds.max_length)
    test_ds = SpamDataset(os.path.join(DATA_DIR, "test.csv"), tokenizer, max_length=train_ds.max_length)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    optimizer = torch.optim.AdamW(gpt.parameters(), lr=args.lr, weight_decay=0.1)
    use_amp = (device.type == "cuda")
    scaler = GradScaler(device=device.type, enabled=use_amp)
    if use_amp:
        print("Mixed precision (AMP) enabled for faster, lower-memory training.")
    train_classifier_simple(gpt, train_loader, val_loader, optimizer, device,
                             num_epochs=args.epochs, eval_freq=50, eval_iter=5,
                             scaler=scaler, use_amp=use_amp)

    test_accuracy = calc_accuracy_loader(test_loader, gpt, device, use_amp=use_amp)
    print(f"Test accuracy: {test_accuracy*100:.2f}%")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"model_state_dict": gpt.state_dict(), "max_length": train_ds.max_length}, args.out)
    print(f"Saved classifier checkpoint to {args.out}")


if __name__ == "__main__":
    main()
