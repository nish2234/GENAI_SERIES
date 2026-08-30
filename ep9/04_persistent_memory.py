"""
Episode 09 — File 4: Persistent Memory (JSON File Storage)
============================================================
Run: python 04_persistent_memory.py

What this file does:
  - Stores conversation history + summary to a JSON file
  - On startup, loads existing memory — the agent remembers past sessions
  - On each turn, auto-saves to disk
  - Exit, restart the script, and it picks up right where you left off
  - Supports 'reset' command to wipe memory and start fresh
  - Combines summarised memory (file 3) with persistence

This is the "killer demo" — memory that survives restarts.
"""

import json
import os
from datetime import datetime

from gemini_client import generate, count_tokens

SYSTEM_PROMPT = (
    "You are a friendly, helpful assistant with persistent memory. "
    "You remember conversations across sessions. When the user returns, "
    "greet them warmly and reference what you know. Keep answers concise."
)

MEMORY_FILE = "memory_store.json"
RECENT_TURNS_TO_KEEP = 4
COMPRESS_THRESHOLD = 5


def load_memory(filepath: str) -> dict:
    """Load memory from JSON file, or return empty structure."""
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            data = json.load(f)
        print(f"  [Loaded memory from {filepath}]")
        print(f"  [Sessions: {data.get('session_count', 0)} | "
              f"Total turns: {data.get('total_turns', 0)} | "
              f"Summary: {'Yes' if data.get('summary') else 'No'}]")
        return data
    return {
        "summary": None,
        "recent_history": [],
        "session_count": 0,
        "total_turns": 0,
        "created_at": datetime.now().isoformat(),
        "last_active": None,
    }


def save_memory(filepath: str, memory: dict):
    """Save memory to JSON file."""
    memory["last_active"] = datetime.now().isoformat()
    with open(filepath, "w") as f:
        json.dump(memory, f, indent=2)


def summarise_turns(turns: list[dict], existing_summary: str | None = None) -> str:
    """Compress conversation turns into a summary."""
    parts = []
    if existing_summary:
        parts.append(f"Previous summary: {existing_summary}")
    parts.append("Recent conversation:")
    for t in turns:
        parts.append(f"  {t['role'].capitalize()}: {t['content']}")

    prompt = (
        "Summarise this conversation in 2-3 sentences. Capture ALL key facts "
        "the user shared (names, preferences, locations, jobs, pets, hobbies, etc). "
        "Write in third person. Preserve facts from the previous summary.\n\n"
        + "\n".join(parts)
        + "\n\nUpdated summary:"
    )
    return generate(prompt, temperature=0.3, max_tokens=256)


def build_prompt(memory: dict, user_message: str) -> str:
    """Build the full prompt from persistent memory + current message."""
    parts = [f"System: {SYSTEM_PROMPT}", ""]

    if memory["summary"]:
        parts.append(f"[Memory from previous sessions: {memory['summary']}]")
        parts.append("")

    recent = memory["recent_history"][-RECENT_TURNS_TO_KEEP:]
    for turn in recent:
        role = turn["role"].capitalize()
        parts.append(f"{role}: {turn['content']}")

    parts.append(f"User: {user_message}")
    parts.append("Assistant:")
    return "\n".join(parts)


def compress_if_needed(memory: dict) -> bool:
    """Compress old turns into summary if threshold exceeded."""
    if len(memory["recent_history"]) < COMPRESS_THRESHOLD:
        return False

    old_turns = memory["recent_history"][:-RECENT_TURNS_TO_KEEP]
    new_summary = summarise_turns(old_turns, memory["summary"])
    memory["summary"] = new_summary
    memory["recent_history"] = memory["recent_history"][-RECENT_TURNS_TO_KEEP:]
    return True


def chat_persistent():
    """Interactive chat with persistent, file-backed memory."""
    print("=" * 60)
    print("PERSISTENT MEMORY CHATBOT")
    print(f"Memory file: {MEMORY_FILE}")
    print("Your conversation survives restarts!")
    print()
    print("Commands:")
    print("  quit    — exit (memory is saved)")
    print("=" * 60)
    print()

    memory = load_memory(MEMORY_FILE)
    memory["session_count"] += 1
    session_num = memory["session_count"]
    print()

    if memory["summary"] or memory["recent_history"]:
        welcome_prompt = build_prompt(
            memory,
            "I just came back. Welcome me and briefly mention what you "
            "remember about me from our previous conversations."
        )
        welcome = generate(welcome_prompt, temperature=0.7)
        print(f"Bot: {welcome}")
        print(f"  [Session #{session_num} | Loaded from disk]")
        print()
    else:
        print(f"Bot: Hi there! I'm your assistant with persistent memory.")
        print(f"     Tell me about yourself — I'll remember it even after you close this app.")
        print(f"  [Session #{session_num} | Fresh start]")
        print()

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            save_memory(MEMORY_FILE, memory)
            print(f"\n  [Memory saved to {MEMORY_FILE}]")
            print(f"  [Total turns this session: {memory['total_turns']}]")
            print(f"  [Run this script again — I'll remember everything!]")
            break

        if not user_input:
            continue


        compressed = compress_if_needed(memory)
        if compressed:
            print(f"  [Old turns compressed into summary to save tokens]")

        prompt = build_prompt(memory, user_input)
        token_count = count_tokens(prompt)

        response = generate(prompt, temperature=0.7)

        memory["recent_history"].append({"role": "user", "content": user_input})
        memory["recent_history"].append({"role": "assistant", "content": response})
        memory["total_turns"] += 1

        save_memory(MEMORY_FILE, memory)

        print(f"Bot: {response}")
        print(f"  [{token_count} tokens | turn #{memory['total_turns']} | auto-saved]")
        print()





if __name__ == "__main__":

    print(" Interactive chat (real persistent memory)")
    chat_persistent()

