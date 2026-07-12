"""
Persistent conversation storage: one JSON file per conversation under
logs/conversations/. Powers both multi-turn memory (feeding prior turns
back into generation) and the history viewer (listing/reopening past
conversations).

Kept separate from interaction_logger.py's logs/interactions.jsonl on
purpose: that file is training data (one flat prompt/response pair per
line, consumed by autonomous_trainer_hf.py's auto-filter). This module is
product data (full conversations, browsable, editable) -- different shape,
different consumers, shouldn't be coupled.
"""
import json
import os
import threading
import time
import uuid

CONV_DIR = "logs/conversations"
_lock = threading.Lock()


def _path(conversation_id):
    return os.path.join(CONV_DIR, f"{conversation_id}.json")


def create_conversation():
    conversation_id = uuid.uuid4().hex[:12]
    os.makedirs(CONV_DIR, exist_ok=True)
    record = {
        "id": conversation_id,
        "title": None,  # set from the first user message once one arrives
        "created_at": time.time(),
        "updated_at": time.time(),
        "messages": [],  # [{"role": "user"|"assistant", "content": str, "ts": float}]
    }
    with _lock:
        with open(_path(conversation_id), "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
    return conversation_id


def get_conversation(conversation_id):
    path = _path(conversation_id)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_message(conversation_id, role, content):
    """Appends a message and returns the updated message list."""
    with _lock:
        record = get_conversation(conversation_id)
        if record is None:
            raise FileNotFoundError(f"No conversation with id {conversation_id}")

        record["messages"].append({"role": role, "content": content, "ts": time.time()})
        record["updated_at"] = time.time()
        if record["title"] is None and role == "user":
            record["title"] = content[:60] + ("..." if len(content) > 60 else "")

        with open(_path(conversation_id), "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        return record["messages"]


def list_conversations():
    """Returns conversation summaries (no message bodies), newest first."""
    if not os.path.exists(CONV_DIR):
        return []
    summaries = []
    for fname in os.listdir(CONV_DIR):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(CONV_DIR, fname), "r", encoding="utf-8") as f:
                record = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        summaries.append({
            "id": record["id"],
            "title": record["title"] or "New conversation",
            "created_at": record["created_at"],
            "updated_at": record["updated_at"],
            "message_count": len(record["messages"]),
        })
    summaries.sort(key=lambda s: s["updated_at"], reverse=True)
    return summaries


def delete_conversation(conversation_id):
    path = _path(conversation_id)
    if os.path.exists(path):
        os.remove(path)
        return True
    return False
