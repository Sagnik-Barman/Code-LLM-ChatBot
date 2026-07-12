"""
Diagnostic: evaluate the saved spam classifier on the full test set with a
per-class breakdown, so we can tell apart "generally imperfect" from
"systematically biased toward one class" (the latter hides behind a
deceptively OK-looking overall accuracy number).

Usage:
    python diagnose_classifier.py --device cuda
"""
import argparse

import tiktoken
import torch
from torch.utils.data import DataLoader

from classify_finetune import DATA_DIR, SpamDataset, setup_classifier
from load_pretrained import load_pretrained_gpt2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", default="gpt2-small (124M)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", default="checkpoints/spam_classifier.pth")
    args = parser.parse_args()

    device = torch.device(args.device)
    tokenizer = tiktoken.get_encoding("gpt2")

    gpt, cfg = load_pretrained_gpt2(args.size, device)
    gpt = setup_classifier(gpt, cfg)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    gpt.load_state_dict(checkpoint["model_state_dict"])
    max_length = checkpoint["max_length"]
    gpt.to(device)
    gpt.eval()

    test_ds = SpamDataset(f"{DATA_DIR}/test.csv", tokenizer, max_length=max_length)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

    # Confusion matrix: rows = actual, cols = predicted. 0 = ham, 1 = spam
    tp = tn = fp = fn = 0

    with torch.no_grad():
        for input_batch, target_batch in test_loader:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            logits = gpt(input_batch)[:, -1, :]
            preds = torch.argmax(logits, dim=-1)

            for pred, actual in zip(preds.tolist(), target_batch.tolist()):
                if actual == 1 and pred == 1:
                    tp += 1
                elif actual == 0 and pred == 0:
                    tn += 1
                elif actual == 0 and pred == 1:
                    fp += 1
                elif actual == 1 and pred == 0:
                    fn += 1

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total
    spam_recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spam_precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    ham_recall = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    print(f"Total test examples: {total}")
    print(f"Overall accuracy:    {accuracy*100:.2f}%")
    print()
    print("Confusion matrix:")
    print(f"                 Predicted ham   Predicted spam")
    print(f"  Actual ham:       {tn:4d}             {fp:4d}")
    print(f"  Actual spam:      {fn:4d}             {tp:4d}")
    print()
    print(f"Spam recall (% of real spam caught):     {spam_recall*100:.2f}%")
    print(f"Spam precision (% flagged-spam correct): {spam_precision*100:.2f}%")
    print(f"Ham recall (% of real ham left alone):   {ham_recall*100:.2f}%")

    if spam_recall < 0.85:
        print("\n=> Spam recall is notably weaker than ham recall: the model is biased")
        print("   toward predicting 'ham'. This is the bias behind misses like the")
        print("   obvious gift-card spam example. Worth retraining with more epochs")
        print("   and/or unfreezing more layers.")


if __name__ == "__main__":
    main()
