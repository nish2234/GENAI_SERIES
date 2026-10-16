"""
Episode 15 — File 1: Hello ADK
================================
Run: python 01_hello_adk.py

What this file does:
  - Creates the simplest possible ADK agent (no tools)
  - Uses InMemoryRunner to run it programmatically
  - Sends a single question and prints the response
  - Then sends a multi-turn conversation to show session memory

This is your first ADK agent — the "hello world" of Google ADK.
"""

import asyncio

from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner

from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

agent = Agent(
    name="hello_agent",
    model="gemini-3.1-flash-lite",
    instruction="You are a friendly assistant. Be helpful, concise, and clear.",
)


async def main():
    runner = InMemoryRunner(agent=agent)

    # --- Single question ---
    print("=" * 60)
    print("HELLO ADK — Single Question")
    print("=" * 60)

    events = await runner.run_debug(
        "What is the capital of France?",
        user_id="user1",
    )

    print(events)
    for event in events:
        if event.is_final_response():
            print(f"\nAgent: {event.content.parts[0].text}")

    # --- Multi-turn conversation ---
    print("\n" + "=" * 60)
    print("HELLO ADK — Multi-Turn Conversation")
    print("=" * 60)

    events = await runner.run_debug(
        [
            "What is the capital of France?",
            "And what about Germany?",
            "Which one has a larger population?",
        ],
        user_id="user2",
    )

    for event in events:
        if event.is_final_response():
            print(f"\nAgent: {event.content.parts[0].text}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
