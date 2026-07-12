"""
Load OpenAI's real pretrained GPT-2 weights into our from-scratch GPTModel.

Uses Hugging Face `transformers` to fetch the weights (no TensorFlow needed,
unlike the original book's gpt_download.py). This downloads ~500MB on first
run and caches locally afterward.

Usage:
    python load_pretrained.py --size "gpt2-small (124M)"
    python load_pretrained.py --size "gpt2-medium (355M)" --prompt "The future of AI is"

If you instead downloaded weights manually from the Kaggle mirror
(https://www.kaggle.com/datasets/xhlulu/openai-gpt2-weights), point
--local_dir at the extracted folder; see load_from_kaggle_dir() below.
"""
import argparse

import numpy as np
import tiktoken
import torch

from generate import generate, text_to_token_ids, token_ids_to_text
from model import GPT_CONFIG_124M_PRETRAINED, MODEL_CONFIGS, GPTModel


def assign(left, right):
    if not isinstance(right, torch.Tensor):
        right = torch.tensor(right)
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch: {left.shape} vs {right.shape}")
    return torch.nn.Parameter(right.clone().detach())


def load_weights_into_gpt_from_hf(gpt, hf_model):
    """Copy weights from a Hugging Face GPT2LMHeadModel into our GPTModel."""
    sd = hf_model.state_dict()

    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, sd["transformer.wpe.weight"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, sd["transformer.wte.weight"])

    for b in range(len(gpt.trf_blocks)):
        prefix = f"transformer.h.{b}."

        # Combined qkv weight in HF -> split into q, k, v for our model
        qkv_w = sd[prefix + "attn.c_attn.weight"].T  # HF stores transposed (Conv1D)
        q_w, k_w, v_w = np.split(qkv_w.numpy(), 3, axis=0)
        gpt.trf_blocks[b].att.W_query.weight = assign(gpt.trf_blocks[b].att.W_query.weight, q_w)
        gpt.trf_blocks[b].att.W_key.weight = assign(gpt.trf_blocks[b].att.W_key.weight, k_w)
        gpt.trf_blocks[b].att.W_value.weight = assign(gpt.trf_blocks[b].att.W_value.weight, v_w)

        qkv_b = sd[prefix + "attn.c_attn.bias"]
        q_b, k_b, v_b = np.split(qkv_b.numpy(), 3, axis=0)
        gpt.trf_blocks[b].att.W_query.bias = assign(gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(gpt.trf_blocks[b].att.W_value.bias, v_b)

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight, sd[prefix + "attn.c_proj.weight"].T.numpy()
        )
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias, sd[prefix + "attn.c_proj.bias"].numpy()
        )

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight, sd[prefix + "mlp.c_fc.weight"].T.numpy()
        )
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias, sd[prefix + "mlp.c_fc.bias"].numpy()
        )
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight, sd[prefix + "mlp.c_proj.weight"].T.numpy()
        )
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias, sd[prefix + "mlp.c_proj.bias"].numpy()
        )

        gpt.trf_blocks[b].norm1.scale = assign(gpt.trf_blocks[b].norm1.scale, sd[prefix + "ln_1.weight"].numpy())
        gpt.trf_blocks[b].norm1.shift = assign(gpt.trf_blocks[b].norm1.shift, sd[prefix + "ln_1.bias"].numpy())
        gpt.trf_blocks[b].norm2.scale = assign(gpt.trf_blocks[b].norm2.scale, sd[prefix + "ln_2.weight"].numpy())
        gpt.trf_blocks[b].norm2.shift = assign(gpt.trf_blocks[b].norm2.shift, sd[prefix + "ln_2.bias"].numpy())

    gpt.final_norm.scale = assign(gpt.final_norm.scale, sd["transformer.ln_f.weight"].numpy())
    gpt.final_norm.shift = assign(gpt.final_norm.shift, sd["transformer.ln_f.bias"].numpy())
    # GPT-2 ties the output head to the token embedding weights
    gpt.out_head.weight = assign(gpt.out_head.weight, sd["transformer.wte.weight"])

    return gpt


def load_pretrained_gpt2(size="gpt2-small (124M)", device="cpu"):
    """
    Downloads pretrained GPT-2 weights via Hugging Face and returns a ready-to-use
    GPTModel (our own architecture) with those weights loaded in.
    """
    from transformers import GPT2LMHeadModel

    hf_name_map = {
        "gpt2-small (124M)": "gpt2",
        "gpt2-medium (355M)": "gpt2-medium",
        "gpt2-large (774M)": "gpt2-large",
        "gpt2-xl (1558M)": "gpt2-xl",
    }
    hf_name = hf_name_map[size]
    print(f"Downloading/loading '{hf_name}' from Hugging Face (cached after first run)...")
    hf_model = GPT2LMHeadModel.from_pretrained(hf_name)
    hf_model.eval()

    cfg = {**GPT_CONFIG_124M_PRETRAINED, **MODEL_CONFIGS[size]}
    gpt = GPTModel(cfg)
    gpt = load_weights_into_gpt_from_hf(gpt, hf_model)
    gpt.to(device)
    gpt.eval()
    return gpt, cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", default="gpt2-small (124M)", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--prompt", default="Every effort moves you")
    parser.add_argument("--max_new_tokens", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    gpt, cfg = load_pretrained_gpt2(args.size, device)

    tokenizer = tiktoken.get_encoding("gpt2")
    torch.manual_seed(123)
    token_ids = generate(
        model=gpt,
        idx=text_to_token_ids(args.prompt, tokenizer).to(device),
        max_new_tokens=args.max_new_tokens,
        context_size=cfg["context_length"],
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(token_ids_to_text(token_ids, tokenizer))


if __name__ == "__main__":
    main()
