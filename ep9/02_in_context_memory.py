"""
Episode 09 — File 2: In-Context Memory (Full Conversation History)
===================================================================
Run: python 02_in_context_memory.py

What this file does:
  - Keeps EVERY user/assistant turn in a Python list
  - Formats the entire conversation into the prompt each time
  - The model now remembers your name, preferences, everything
  - Also shows the PROBLEM: token count grows with every turn

This is the simplest form of agent memory — and the most common.
"""

from gemini_client import generate, count_tokens

SYSTEM_PROMPT = (
    "You are a friendly, helpful assistant. You remember everything "
    "the user has told you in this conversation. Refer back to earlier "
    "details when relevant. Keep responses concise."
)


def format_history(history: list[dict]) -> str:
    """Turn the conversation list into a single prompt string."""
    lines = []
    for turn in history:
        role = turn["role"].capitalize()
        lines.append(f"{role}: {turn['content']}")
    return "\n".join(lines)


def build_prompt(history: list[dict], user_message: str) -> str:
    """Build the full prompt: system + history + latest user message."""
    parts = [f"System: {SYSTEM_PROMPT}", ""]

    if history:
        parts.append(format_history(history))

    parts.append(f"User: {user_message}")
    parts.append("Assistant:")
    return "\n".join(parts)


def chat_with_memory():
    """Interactive chat that keeps full conversation history."""
    print("=" * 60)
    print("IN-CONTEXT MEMORY CHATBOT")
    print("Full conversation history in every prompt.")
    print("Type 'quit' to exit.")
    print("=" * 60)
    print()

    history: list[dict] = []

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print(f"\nFinal history: {len(history)} turns")
            break
        if not user_input:
            continue

        prompt = build_prompt(history, user_input)
        token_count = count_tokens(prompt)

        response = generate(prompt, temperature=0.7)

        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})

        print(f"Bot: {response}")
        print(f"  [{token_count} tokens | {len(history) // 2} turns]")
        print()



if __name__ == "__main__":

    print("Interactive chat (try it yourself)")
    print()

    chat_with_memory()

