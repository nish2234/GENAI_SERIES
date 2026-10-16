"""
Episode 15 — File 2: ADK Agent with Tools
==========================================
Run: python 02_adk_with_tool.py

What this file does:
  - Defines two custom function tools (get_current_time, calculate)
  - Creates an ADK Agent with tools=[...] — schema auto-generated from docstrings
  - Runs a multi-turn conversation that triggers different tools
  - Shows how ADK handles tool dispatch automatically

Compare this to Episode 8's manual tool dispatch — same result, fraction of the code.
"""

import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo

from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file
# ─── Tools ────────────────────────────────────────────────────────────────────
# ADK reads the docstring + type hints to auto-generate the function schema.
# No manual FunctionDeclaration needed — just write a normal Python function.


def get_current_time(city: str) -> dict:
    """Get the current time in a specified city.

    Args:
        city: The name of the city (e.g., 'London', 'Tokyo', 'New York').

    Returns:
        A dictionary with the city, timezone, and current time.
    """
    timezone_map = {
        "london": "Europe/London",
        "tokyo": "Asia/Tokyo",
        "new york": "America/New_York",
        "paris": "Europe/Paris",
        "sydney": "Australia/Sydney",
        "mumbai": "Asia/Kolkata",
        "berlin": "Europe/Berlin",
        "los angeles": "America/Los_Angeles",
    }

    tz_name = timezone_map.get(city.lower())
    if not tz_name:
        return {"error": f"Unknown city: {city}. Try: {', '.join(timezone_map.keys())}"}

    now = datetime.now(ZoneInfo(tz_name))
    return {
        "city": city,
        "timezone": tz_name,
        "current_time": now.strftime("%Y-%m-%d %H:%M:%S %Z"),
    }


def calculate(expression: str) -> dict:
    """Evaluate a mathematical expression.

    Args:
        expression: A math expression to evaluate (e.g., '15 * 340 / 100').

    Returns:
        A dictionary with the expression and its result.
    """
    try:
        allowed = {"abs": abs, "round": round, "min": min, "max": max, "pow": pow}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return {"expression": expression, "result": str(result)}
    except Exception as e:
        return {"expression": expression, "error": str(e)}


# ─── Agent ────────────────────────────────────────────────────────────────────

agent = Agent(
    name="time_agent",
    model="gemini-3.1-flash-lite",
    instruction=(
        "You are a helpful assistant that can tell the current time in cities "
        "and evaluate math expressions. Use the appropriate tool when needed. "
        "Be friendly and concise."
    ),
    tools=[get_current_time, calculate],
)


# ─── Run ──────────────────────────────────────────────────────────────────────

async def main():
    runner = InMemoryRunner(agent=agent)

    print("=" * 60)
    print("ADK AGENT WITH TOOLS")
    print("=" * 60)

    prompts = [
        "What time is it in Tokyo right now?",
        "And in New York?",
        "What is the time difference between them in hours?",
        "Now calculate: 1920 * 1080",
    ]

    events = await runner.run_debug(prompts, user_id="user1")

    for event in events:
        if event.is_final_response():
            print(f"\nAgent: {event.content.parts[0].text}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
