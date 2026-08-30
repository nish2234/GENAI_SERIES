"""
Episode 09 — File 1: The Problem — A Stateless Chatbot
========================================================
Run: python 01_no_memory.py

What this file demonstrates:
  - A chatbot with ZERO memory
  - Each turn is a brand-new, independent API call
  - The model has no idea what you said 10 seconds ago
  - Tell it your name, then ask "what is my name?" — it can't answer

This is the PROBLEM that motivates the entire episode.
"""

from gemini_client import generate

SYSTEM_PROMPT = (
    "You are a friendly assistant. Answer questions concisely. "
    "If you don't know something, say so honestly."
)


def chat_no_memory():
    """Each user message is sent in isolation — no history, no context."""
    print("=" * 60)
    print("STATELESS CHATBOT (no memory)")
    print("Every message is sent in complete isolation.")
    print("Type 'quit' to exit.")
    print("=" * 60)
    print()

    turn = 0
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("\nGoodbye!")
            break
        if not user_input:
            continue

        turn += 1

        response = generate(user_input, system=SYSTEM_PROMPT)
        print(f"Bot: {response}\n")


if __name__ == "__main__":

    print("  Interactive chat (try it yourself)")
    print()
    chat_no_memory()

