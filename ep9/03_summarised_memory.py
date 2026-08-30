"""
Episode 09 — File 3: Summarised Memory (Compress Old Turns)
=============================================================
Run: python 03_summarised_memory.py

What this file does:
  - Keeps the most recent N turns in full
  - When conversation exceeds a threshold, compresses old turns
    into a 2-3 sentence summary using generate()
  - The summary replaces the old turns in the prompt
  - Token count stays manageable even in long conversations

This is the sweet spot between full history and no history.
"""

from gemini_client import generate, count_tokens

SYSTEM_PROMPT = (
    "You are a friendly, helpful assistant. You have access to a summary "
    "of earlier conversation and the most recent messages. Use both to "
    "give contextual, helpful responses. Keep your answers concise."
)

RECENT_TURNS_TO_KEEP = 4   # keep last 6 messages (3 user + 3 assistant) in full
COMPRESS_THRESHOLD = 5     # compress when total messages exceed this


def summarise_turns(turns: list[dict]) -> str:
    """Use the LLM to compress a list of conversation turns into a summary."""
    formatted = "\n".join(
        f"{t['role'].capitalize()}: {t['content']}" for t in turns
    )
    prompt = (
        "Summarise this conversation in 2-3 sentences. Capture all key facts "
        "the user shared (names, preferences, locations, jobs, pets, etc). "
        "Write in third person (e.g., 'The user said their name is Alex').\n\n"
        f"{formatted}\n\nSummary:"
    )
    return generate(prompt, temperature=0.3, max_tokens=256)


def build_prompt(summary: str | None, recent: list[dict], user_message: str) -> str:
    """Build prompt with optional summary + recent turns + new message."""
    parts = [f"System: {SYSTEM_PROMPT}", ""]

    if summary:
        parts.append(f"[Summary of earlier conversation: {summary}]")
        parts.append("")

    if recent:
        for turn in recent:
            role = turn["role"].capitalize()
            parts.append(f"{role}: {turn['content']}")

    parts.append(f"User: {user_message}")
    parts.append("Assistant:")
    return "\n".join(parts)


def chat_with_summarised_memory():
    """Interactive chat with automatic memory compression."""
    print("=" * 60)
    print("SUMMARISED MEMORY CHATBOT")
    print(f"Keeps last {RECENT_TURNS_TO_KEEP} messages in full.")
    print(f"Compresses older messages when total > {COMPRESS_THRESHOLD}.")
    print("Type 'quit' to exit, 'tokens' to see prompt size.")
    print("=" * 60)
    print()

    history: list[dict] = []
    summary: str | None = None
    compressions = 0

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print(f"\nFinal: {len(history)} turns, {compressions} compressions")
            break
        if not user_input:
            continue


        if len(history) >= COMPRESS_THRESHOLD:
            old_turns = history[:-RECENT_TURNS_TO_KEEP]
            old_context = old_turns

            if summary:
                old_context = [{"role": "system", "content": f"Previous summary: {summary}"}] + old_context

            new_summary = summarise_turns(old_context)
            summary = new_summary
            history = history[-RECENT_TURNS_TO_KEEP:]
            compressions += 1
            print(f"  [Memory compressed! Summary updated. Compression #{compressions}]")

        prompt = build_prompt(summary, history[-RECENT_TURNS_TO_KEEP:], user_input)
        token_count = count_tokens(prompt)

        response = generate(prompt, temperature=0.7)

        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})

        print(f"Bot: {response}")
        print(f"  [{token_count} tokens | {len(history)} turns in buffer]")
        print()




if __name__ == "__main__":
    print(" Interactive chat (try it yourself)")
    print()

    chat_with_summarised_memory()
  
