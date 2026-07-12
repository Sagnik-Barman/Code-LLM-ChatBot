"""
Local chatbot web app: serves your fine-tuned GPT model through a browser UI.

Usage:
    python app.py
    python app.py --model checkpoints/instruction_model_alpaca.pth --port 8000

Then open http://localhost:8000 in your browser.
"""
import argparse
import os
import threading
import time

import tiktoken
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from generate import generate, text_to_token_ids, token_ids_to_text
from instruction_finetune import format_input
from interaction_logger import log_interaction
from load_pretrained import load_pretrained_gpt2

app = FastAPI(title="Local GPT Chatbot")

# Populated at startup by main(); swapped in-place by /api/reload
STATE = {
    "model": None,
    "tokenizer": None,
    "cfg": None,
    "device": None,
    "model_path": None,
    "size": None,
    "loaded_at": None,
}
STATE_LOCK = threading.Lock()  # guards STATE during a reload swap


class ChatRequest(BaseModel):
    message: str
    temperature: float = 0.0
    top_k: int | None = None
    max_new_tokens: int = 150


class ChatResponse(BaseModel):
    response: str
    gen_seconds: float


class ReloadRequest(BaseModel):
    model: str | None = None  # defaults to the currently-configured checkpoint path


@app.get("/api/status")
def status():
    with STATE_LOCK:
        return {
            "model_path": STATE["model_path"],
            "size": STATE["size"],
            "device": str(STATE["device"]),
            "context_length": STATE["cfg"]["context_length"] if STATE["cfg"] else None,
            "loaded_at": STATE["loaded_at"],
        }


@app.post("/api/reload")
def reload_model(req: ReloadRequest = ReloadRequest()):
    """
    Hot-swaps the served model weights without restarting the process.
    Called by autonomous_trainer.py after it promotes a new checkpoint;
    can also be called manually, e.g. `curl -X POST localhost:8000/api/reload`.
    """
    with STATE_LOCK:
        model_path = req.model or STATE["model_path"]
        device = STATE["device"]
        size = STATE["size"]

    if not model_path or not os.path.exists(model_path):
        return {"status": "error", "detail": f"Checkpoint not found at '{model_path}'"}

    print(f"[reload] Loading weights from {model_path} ...")
    gpt, cfg = load_pretrained_gpt2(size, device)
    checkpoint = torch.load(model_path, map_location=device)
    gpt.load_state_dict(checkpoint["model_state_dict"])
    gpt.eval()

    # Build the new state fully before taking the lock, so in-flight chat
    # requests never see a half-swapped model.
    with STATE_LOCK:
        STATE["model"] = gpt
        STATE["cfg"] = cfg
        STATE["model_path"] = model_path
        STATE["loaded_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    print(f"[reload] Now serving {model_path}")
    return {"status": "ok", "model_path": model_path, "loaded_at": STATE["loaded_at"]}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    with STATE_LOCK:
        model = STATE["model"]
        tokenizer = STATE["tokenizer"]
        cfg = STATE["cfg"]
        device = STATE["device"]
        model_path = STATE["model_path"]

    entry = {"instruction": req.message, "input": ""}
    input_text = format_input(entry)

    t0 = time.time()
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=req.max_new_tokens,
        context_size=cfg["context_length"],
        temperature=req.temperature,
        top_k=req.top_k,
        eos_id=50256,
        no_repeat_ngram_size=3,
    )
    gen_seconds = time.time() - t0

    full_text = token_ids_to_text(token_ids, tokenizer)
    response_text = full_text[len(input_text):].replace("### Response:", "").strip()

    log_interaction(
        prompt=req.message,
        response=response_text,
        meta={"model_path": model_path, "temperature": req.temperature},
    )

    return ChatResponse(response=response_text, gen_seconds=gen_seconds)


# Serve the frontend
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root():
    return FileResponse("static/index.html")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="checkpoints/instruction_model.pth",
                         help="Path to the instruction-tuned checkpoint to serve")
    parser.add_argument("--size", default="gpt2-small (124M)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit(
            f"Checkpoint not found at '{args.model}'. Train one first "
            f"(python instruction_finetune.py ...) or pass --model <path>."
        )

    device = torch.device(args.device)
    print(f"Loading base model ({args.size}) on {device} ...")
    gpt, cfg = load_pretrained_gpt2(args.size, device)

    print(f"Loading fine-tuned weights from {args.model} ...")
    checkpoint = torch.load(args.model, map_location=device)
    gpt.load_state_dict(checkpoint["model_state_dict"])
    gpt.eval()

    STATE["model"] = gpt
    STATE["tokenizer"] = tiktoken.get_encoding("gpt2")
    STATE["cfg"] = cfg
    STATE["device"] = device
    STATE["model_path"] = args.model
    STATE["size"] = args.size
    STATE["loaded_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    print(f"Ready. Open http://localhost:{args.port} in your browser.")
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
