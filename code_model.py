"""
Loads Qwen2.5-Coder-1.5B-Instruct (or another causal-LM coding model) via
Hugging Face `transformers`, optionally 4-bit quantized, with an optional
LoRA adapter layered on top. This replaces model.py/load_pretrained.py/
generate.py for the code-model pipeline -- those files are specific to the
from-scratch GPT-2 architecture and don't apply here.

Why Qwen2.5-Coder-1.5B-Instruct: Apache-2.0 licensed, strong Python/C++
performance for its size, and small enough to LoRA fine-tune on a laptop
GPU (RTX 3050-4060 class) using 4-bit quantization -- full weights stay
frozen, only a small adapter (tens of MB) trains and gets saved.

Usage as a library:
    from code_model import load_model, generate_response
    model, tokenizer = load_model(adapter_path="checkpoints/code_lora_v1")
    reply = generate_response(model, tokenizer, "Write a function that reverses a list.")
"""
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
LIVE_ADAPTER_POINTER = "checkpoints/live_adapter_path.txt"


def get_live_adapter_path(default):
    """
    Resolves to whatever autonomous_trainer_hf.py last promoted, if
    anything -- falls back to `default` if no cycle has promoted yet, or
    if the pointed-to adapter no longer exists on disk.
    """
    if os.path.exists(LIVE_ADAPTER_POINTER):
        with open(LIVE_ADAPTER_POINTER) as f:
            path = f.read().strip()
        if path and os.path.exists(path):
            return path
    return default


def set_live_adapter_path(path):
    os.makedirs(os.path.dirname(LIVE_ADAPTER_POINTER) or ".", exist_ok=True)
    with open(LIVE_ADAPTER_POINTER, "w") as f:
        f.write(path)


def load_base_model(base_model=BASE_MODEL, use_4bit=True, device="cuda"):
    """Loads just the base model + tokenizer, no adapter. Load this once and
    reuse it across multiple adapter attach/reload cycles -- the base model
    is the expensive part (3GB+ download, VRAM-resident weights); attaching
    a LoRA adapter on top is cheap (tens of MB) by comparison."""
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if use_4bit and device == "cuda":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant_config,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map=device if device == "cuda" else None,
    )
    if device != "cuda":
        model = model.to(device)
    model.eval()
    return model, tokenizer


def attach_adapter(base_model, adapter_path):
    """Wraps an already-loaded base model with a LoRA adapter. Cheap: only
    the small adapter weights get loaded, the frozen base is reused as-is."""
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return model


def load_model(base_model=BASE_MODEL, adapter_path=None, use_4bit=True, device="cuda"):
    """Convenience one-shot loader (base + optional adapter together)."""
    model, tokenizer = load_base_model(base_model=base_model, use_4bit=use_4bit, device=device)
    if adapter_path:
        model = attach_adapter(model, adapter_path)
    return model, tokenizer


def format_chat_prompt(tokenizer, instruction, history=None, system=None):
    """
    Uses the model's own chat template (Qwen2.5-Coder is chat-tuned).
    history, if given, is a list of {"role": "user"|"assistant", "content": str}
    prior turns, fed back in so the model has real conversational memory
    instead of treating every message as a fresh, standalone request.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    if history:
        messages.extend({"role": m["role"], "content": m["content"]} for m in history)
    messages.append({"role": "user", "content": instruction})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate_response(model, tokenizer, instruction, history=None, system=None,
                       max_new_tokens=400, temperature=0.0):
    prompt = format_chat_prompt(tokenizer, instruction, history=history, system=system)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    gen_kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
    if temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature)
    else:
        gen_kwargs.update(do_sample=False)

    output_ids = model.generate(**inputs, **gen_kwargs)
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
