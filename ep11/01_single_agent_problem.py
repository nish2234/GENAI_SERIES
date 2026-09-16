"""
01 — The Single Agent Problem
==============================
One agent tries to do everything: research AND write a blog post.
Result: mediocre at both. This motivates splitting into specialists.
"""

from gemini_client import generate

TOPIC = "quantum computing"

PROMPT = f"""You are a helpful AI assistant. The user wants you to:
1. Research the topic "{TOPIC}" thoroughly — find key facts, recent breakthroughs, and important concepts.
2. Write a polished, engaging blog post about it.

Do both in a single response. First show your research, then write the blog post."""


def single_agent_attempt(topic: str) -> str:
    prompt = f"""You are a helpful AI assistant. The user wants you to:
1. Research the topic "{topic}" thoroughly — find key facts, recent breakthroughs, and important concepts.
2. Write a polished, engaging blog post about it.

Do both in a single response. First show your research, then write the blog post."""

    return generate(prompt, max_tokens=2048, temperature=0.7)


if __name__ == "__main__":
    print("=" * 60)
    print("THE SINGLE AGENT PROBLEM")
    print("=" * 60)
    print(f"\nTask: Research '{TOPIC}' AND write a blog post about it.")
    print("One agent. One prompt. Doing everything.\n")
    print("-" * 60)

    result = single_agent_attempt(TOPIC)
    print(result)

