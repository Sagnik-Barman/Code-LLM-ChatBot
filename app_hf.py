"""
Local chatbot web app serving the Qwen2.5-Coder + LoRA pipeline (code_model.py).
This is the HF-pipeline equivalent of app.py, which only works with the
from-scratch GPT-2 architecture.

Usage:
    python app_hf.py
    python app_hf.py --adapter checkpoints/code_lora_v1 --port 8000
"""
import argparse
import ntpath
import os
import posixpath
import threading
import time

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from code_model import attach_adapter, generate_response, get_live_adapter_path, load_base_model
from conversation_store import append_message, create_conversation, delete_conversation, get_conversation, list_conversations
from interaction_logger import log_interaction

app = FastAPI(title="Local Code Chatbot (Qwen2.5-Coder)")

# Guards model reads (inference) against concurrent writes (hot-reload) so a
# reload can never swap the model/adapter out from under an in-flight generation.
STATE_LOCK = threading.Lock()

STATE = {
    "base_model": None,   # the actual loaded model object -- stays resident across reloads
    "model": None,        # base_model, or base_model wrapped with an attached adapter
    "tokenizer": None,
    "adapter_path": None,
    "device": None,
    "loaded_at": None,
}


class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None  # None starts a new conversation
    temperature: float = 0.0
    max_new_tokens: int = 400


class ChatResponse(BaseModel):
    response: str
    gen_seconds: float
    conversation_id: str


class ReloadRequest(BaseModel):
    adapter_path: str | None = None  # None -> reload with whatever's currently configured


class SaveFileRequest(BaseModel):
    path: str    # relative path, e.g. "two_sum.py" or "utils/helpers.py"
    content: str


FILES_ROOT = "generated_files"  # overridden by --files_root in main()


def _resolve_safe_path(relative_path):
    """
    Resolves relative_path against FILES_ROOT and rejects anything that
    would escape it. Checks Windows AND POSIX absolute-path/drive-letter
    semantics explicitly, and manually rejects ".." components after
    normalizing both separator styles -- os.path.join/abspath only collapse
    ".." according to the CURRENT platform's separator, so a backslash-style
    "..\\..\\x" traversal would silently fail to get caught when this is
    tested on Linux, even though backslash IS the separator (and this WOULD
    be a real escape) on the Windows machine this actually deploys to.
    """
    if ntpath.isabs(relative_path) or posixpath.isabs(relative_path):
        return None
    if ntpath.splitdrive(relative_path)[0]:  # catches "C:foo.py" (drive-relative, no slash)
        return None

    normalized = relative_path.replace("\\", "/")
    parts = [p for p in normalized.split("/") if p not in ("", ".")]
    if any(p == ".." for p in parts):
        return None

    root = os.path.abspath(FILES_ROOT)
    candidate = os.path.abspath(os.path.join(root, *parts)) if parts else root
    if os.path.commonpath([root, candidate]) != root:
        return None
    return candidate


@app.post("/api/save_file")
def save_file(req: SaveFileRequest):
    """
    Writes a file to disk -- only ever called when the person explicitly
    clicks "Save to project" in the UI and confirms a filename, never
    automatically after a response is generated.
    """
    safe_path = _resolve_safe_path(req.path)
    if safe_path is None:
        return {"status": "error", "detail": f"'{req.path}' resolves outside {FILES_ROOT}/ -- refusing to write there."}

    os.makedirs(os.path.dirname(safe_path), exist_ok=True)
    with open(safe_path, "w", encoding="utf-8") as f:
        f.write(req.content)

    return {"status": "saved", "path": os.path.relpath(safe_path, os.getcwd())}


@app.get("/api/status")
def status():
    return {
        "adapter_path": STATE["adapter_path"],
        "device": STATE["device"],
        "loaded_at": STATE["loaded_at"],
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    conversation_id = req.conversation_id or create_conversation()
    conv = get_conversation(conversation_id)
    if conv is None:
        # client sent an id that doesn't exist (e.g. after a fresh log wipe) -- start clean
        conversation_id = create_conversation()
        conv = get_conversation(conversation_id)

    history = [{"role": m["role"], "content": m["content"]} for m in conv["messages"]]

    t0 = time.time()
    with STATE_LOCK:
        response_text = generate_response(
            STATE["model"], STATE["tokenizer"], req.message, history=history,
            max_new_tokens=req.max_new_tokens, temperature=req.temperature,
        )
    gen_seconds = time.time() - t0

    append_message(conversation_id, "user", req.message)
    append_message(conversation_id, "assistant", response_text)

    log_interaction(
        prompt=req.message, response=response_text,
        meta={"adapter_path": STATE["adapter_path"], "temperature": req.temperature,
              "conversation_id": conversation_id},
    )

    return ChatResponse(response=response_text, gen_seconds=gen_seconds, conversation_id=conversation_id)


@app.get("/api/conversations")
def get_conversations():
    """Summaries for the history sidebar. Translates conversation_store's
    internal id/messages naming to the conversation_id/message_count shape
    the frontend expects."""
    return [
        {
            "conversation_id": s["id"],
            "title": s["title"],
            "created_at": s["created_at"],
            "updated_at": s["updated_at"],
            "message_count": s["message_count"],
        }
        for s in list_conversations()
    ]


@app.get("/api/conversations/{conversation_id}")
def get_conversation_detail(conversation_id: str):
    conv = get_conversation(conversation_id)
    if conv is None:
        return {"error": "not found"}
    return {
        "conversation_id": conv["id"],
        "title": conv["title"],
        "created_at": conv["created_at"],
        "updated_at": conv["updated_at"],
        "turns": [{"role": m["role"], "content": m["content"]} for m in conv["messages"]],
    }


@app.delete("/api/conversations/{conversation_id}")
def remove_conversation(conversation_id: str):
    deleted = delete_conversation(conversation_id)
    return {"status": "deleted" if deleted else "not found"}


@app.post("/api/reload")
def reload_model(req: ReloadRequest = ReloadRequest()):
    """
    Hot-reloads with a new (or refreshed) LoRA adapter, with no process
    restart. Called by autonomous_trainer_hf.py after it promotes a new
    adapter. Reuses the already-resident base model -- only the small
    adapter gets loaded, not the full 3GB+ base model again.
    """
    new_adapter_path = req.adapter_path or STATE["adapter_path"]
    if not new_adapter_path or not os.path.exists(new_adapter_path):
        return {"status": "error", "detail": f"Adapter not found at {new_adapter_path}"}

    with STATE_LOCK:
        try:
            new_model = attach_adapter(STATE["base_model"], new_adapter_path)
        except Exception as e:
            return {"status": "error", "detail": f"Failed to attach adapter: {e}"}
        STATE["model"] = new_model
        STATE["adapter_path"] = new_adapter_path
        STATE["loaded_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {"status": "reloaded", "adapter_path": new_adapter_path, "loaded_at": STATE["loaded_at"]}


# Serve the frontend (reuses the same static/ dir as the original app.py)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root():
    return FileResponse("static/index.html")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="checkpoints/code_lora_v1",
                         help="Path to a LoRA adapter dir, or omit --adapter with an empty string to serve the raw base model")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no_4bit", action="store_true")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--files_root", default="generated_files",
                         help="Directory (relative to cwd) that 'Save to project' writes into -- can't escape this")
    args = parser.parse_args()

    global FILES_ROOT
    FILES_ROOT = args.files_root
    os.makedirs(FILES_ROOT, exist_ok=True)

    print(f"Loading base model on {args.device} ...")
    base_model, tokenizer = load_base_model(use_4bit=not args.no_4bit, device=args.device)
    STATE["base_model"] = base_model
    STATE["tokenizer"] = tokenizer
    STATE["device"] = args.device

    adapter_path = args.adapter or None
    if adapter_path:
        adapter_path = get_live_adapter_path(adapter_path)  # picks up the latest promoted adapter, if any
    if adapter_path and os.path.exists(adapter_path):
        print(f"Attaching adapter from {adapter_path} ...")
        STATE["model"] = attach_adapter(base_model, adapter_path)
        STATE["adapter_path"] = adapter_path
    else:
        print("No adapter found -- serving the base model with no fine-tuning applied.")
        STATE["model"] = base_model
        STATE["adapter_path"] = None

    STATE["loaded_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"Ready. Open http://localhost:{args.port} in your browser.")
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
