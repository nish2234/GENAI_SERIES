"""
Episode 16 — App 03: State Prefixes Profile
============================================
Run: cd code && adk web       (then pick "03_state_prefixes_profile" from the dropdown)

What this app does:
  - A dinner-ideas agent that uses all four state scopes:
      (no prefix)  session_topic           -> this session only
      user:        user:name, user:diet... -> every session of this user (same app)
      app:         app:profile_updates     -> every user and session of this app
      temp:        temp:last_saved_field   -> this turn only, never saved
  - With adk web's default SQLite storage, user: and app: state also survive
    a server restart

Try in ADK Web:
  - Session 1: "Hi, I'm Priya. I'm vegetarian and I love Thai food."
  - Session 1: "Tonight I'm planning a dinner party."      (session_topic)
  - Click "New Session", then: "What should I cook tonight?"
      -> it still knows your name, diet and cuisine (user: state)
      -> it does NOT know about the dinner party (no-prefix state)
  - "Actually I'm vegan now."   then switch back to session 1 and check the State tab.

Look at: the State tab in BOTH sessions — the user: and app: keys appear in
each, session_topic only in the first, and temp:last_saved_field in neither.
"""

from google.adk.agents import Agent
from google.adk.tools import ToolContext

MODEL = "gemini-3.1-flash-lite"

PROFILE_FIELDS = ("name", "diet", "favorite_cuisine", "allergies")


def save_user_profile(field: str, value: str, tool_context: ToolContext) -> dict:
    """Save one piece of the user's profile so it is remembered in every future session.

    Args:
        field: Which profile field to save. One of: 'name', 'diet',
            'favorite_cuisine', 'allergies'.
        value: The value to store (e.g., 'Priya', 'vegetarian', 'Thai', 'peanuts').

    Returns:
        A dict with status, the saved state key and value — or an error_message
        if the field is not supported.
    """
    field = field.strip().lower()
    if field not in PROFILE_FIELDS:
        return {
            "status": "error",
            "error_message": f"Unknown field '{field}'. Use one of: {', '.join(PROFILE_FIELDS)}.",
        }

    tool_context.state[f"user:{field}"] = value
    tool_context.state["app:profile_updates"] = tool_context.state.get("app:profile_updates", 0) + 1
    tool_context.state["temp:last_saved_field"] = field
    return {"status": "success", "saved_key": f"user:{field}", "value": value}


def get_user_profile(tool_context: ToolContext) -> dict:
    """Get everything saved about the user, from any previous session.

    Returns:
        A dict with status, the user's profile fields (missing ones are omitted),
        and how many profile updates this app has seen across all users.
    """
    profile = {
        field: tool_context.state.get(f"user:{field}")
        for field in PROFILE_FIELDS
        if tool_context.state.get(f"user:{field}")
    }
    return {
        "status": "success",
        "profile": profile,
        "app_profile_updates": tool_context.state.get("app:profile_updates", 0),
    }


def set_session_topic(topic: str, tool_context: ToolContext) -> dict:
    """Remember what THIS conversation is about (e.g., 'dinner party for 6').

    Args:
        topic: A short description of the current conversation's goal.

    Returns:
        A dict with status and the saved session_topic.
    """
    tool_context.state["session_topic"] = topic
    return {"status": "success", "session_topic": topic}


root_agent = Agent(
    name="dinner_ideas_agent",
    model=MODEL,
    description="Suggests dinner ideas using a profile remembered across sessions.",
    instruction=(
        "You are a warm, practical home-cooking assistant.\n"
        "- At the start of every conversation, call get_user_profile and greet the "
        "user by name if you know it.\n"
        "- Whenever the user tells you their name, diet, favourite cuisine or "
        "allergies, save each one with save_user_profile.\n"
        "- When the user describes what this particular conversation is for "
        "(an occasion, a number of guests), save it with set_session_topic.\n"
        "Current session topic: {session_topic?}\n"
        "Always respect the user's diet and allergies. Suggest 2-3 dishes, briefly."
    ),
    tools=[save_user_profile, get_user_profile, set_session_topic],
)
