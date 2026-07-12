"""
Appends every chat exchange to a JSONL log. The autonomous trainer reads
from this file; it never talks to the live app directly.
"""
import json
import os
import threading
import time

LOG_PATH = os.environ.get("INTERACTION_LOG_PATH", "logs/interactions.jsonl")
_lock = threading.Lock()


def log_interaction(prompt: str, response: str, meta: dict = None):
    os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)
    record = {
        "timestamp": time.time(),
        "prompt": prompt,
        "response": response,
        "meta": meta or {},
    }
    with _lock:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
