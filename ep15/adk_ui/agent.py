"""
Episode 15 — File 3: ADK Web App Agent
========================================
Run: cd code && adk web       (then open http://localhost:8000)
 or: cd code && adk run 03_adk_web_app

What this file does:
  - Defines a root_agent for use with `adk web` and `adk run`
  - Follows the standard ADK project structure:
      03_adk_web_app/
      ├── __init__.py      (from . import agent)
      ├── agent.py          (this file — defines root_agent)
      └── .env              (GOOGLE_API_KEY=...)
  - Two tools: get_current_time and calculate
  - The variable MUST be named root_agent — ADK looks for this name
"""

from datetime import datetime
from zoneinfo import ZoneInfo

from google.adk.agents import Agent


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
        "chicago": "America/Chicago",
        "dubai": "Asia/Dubai",
        "singapore": "Asia/Singapore",
        "hong kong": "Asia/Hong_Kong",
        "toronto": "America/Toronto",
        "sao paulo": "America/Sao_Paulo",
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
    """Evaluate a mathematical expression safely.

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


root_agent = Agent(
    name="smart_assistant",
    model="gemini-3.1-flash-lite",
    description="A helpful assistant that can tell time and do math.",
    instruction=(
        "You are a helpful, friendly assistant. "
        "You can look up the current time in cities using the get_current_time tool, "
        "and you can evaluate math expressions using the calculate tool. "
        "Be concise and clear in your responses."
    ),
    tools=[get_current_time, calculate],
)
