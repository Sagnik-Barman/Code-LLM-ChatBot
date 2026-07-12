"""
Text generation utilities: token<->text conversion, greedy decoding, and the
combined temperature-scaling + top-k sampling decoder.
"""
import torch


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """Greedy decoding (always picks the highest-probability next token)."""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def _banned_ngram_tokens(idx, n):
    """For each sequence in the batch, find which next-tokens would recreate
    an n-gram that's already appeared earlier in the sequence."""
    banned = []
    for seq in idx.tolist():
        seq_banned = set()
        if len(seq) >= n:
            prefix = tuple(seq[-(n - 1):])
            for i in range(len(seq) - n + 1):
                ngram = tuple(seq[i:i + n])
                if ngram[:-1] == prefix:
                    seq_banned.add(ngram[-1])
        banned.append(seq_banned)
    return banned


def generate(model, idx, max_new_tokens, context_size, temperature=0.0,
             top_k=None, eos_id=None, no_repeat_ngram_size=0):
    """
    Full decoding function with temperature scaling and top-k sampling.

    - temperature=0.0  -> falls back to greedy (argmax) decoding
    - temperature>0.0  -> samples from the (optionally top-k filtered) distribution;
                          higher temperature = more random, lower = more confident
    - top_k=k          -> restricts sampling to the k most likely tokens at each step
    - eos_id           -> if set, generation stops early once this token is produced
    - no_repeat_ngram_size=n -> if set, blocks any token that would recreate an
                          n-token sequence already generated earlier. This is what
                          stops greedy decoding from looping on a repeated phrase
                          forever (a common failure mode at temperature=0).
    """
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # --- Block repeated n-grams (prevents "stuck in a loop" generations) ---
        if no_repeat_ngram_size > 0:
            banned = _banned_ngram_tokens(idx, no_repeat_ngram_size)
            for b, banned_tokens in enumerate(banned):
                if banned_tokens:
                    logits[b, list(banned_tokens)] = float("-inf")

        # --- Top-k filtering ---
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float("-inf")).to(logits.device),
                logits,
            )

        # --- Temperature scaling + sampling, or greedy ---
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if eos_id is not None and (idx_next == eos_id).all():
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx


if __name__ == "__main__":
    # Sanity check with the small illustrative vocab from the notebook
    vocab = {"closer": 0, "every": 1, "effort": 2, "forward": 3, "inches": 4,
              "moves": 5, "pizza": 6, "toward": 7, "you": 8}
    inverse_vocab = {v: k for k, v in vocab.items()}

    class TinyModel(torch.nn.Module):
        """Fake 'model' that always returns the same fixed logits, for testing."""
        def forward(self, idx):
            logits = torch.tensor([4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79])
            return logits.repeat(idx.shape[0], idx.shape[1], 1)

    model = TinyModel()
    idx = torch.tensor([[1, 2, 5, 8]])  # "every effort moves you"

    torch.manual_seed(123)
    greedy = generate(model, idx, max_new_tokens=3, context_size=4, temperature=0.0)
    print("Greedy:", [inverse_vocab[i] for i in greedy[0, 4:].tolist()])

    torch.manual_seed(123)
    sampled = generate(model, idx, max_new_tokens=3, context_size=4, temperature=1.4, top_k=3)
    print("Temp=1.4, top_k=3:", [inverse_vocab[i] for i in sampled[0, 4:].tolist()])
