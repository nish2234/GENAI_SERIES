"""
Episode 16 — App 01: Session Basics
====================================
Run: cd code && adk web       (then pick "01_session_basics" from the dropdown)

What this app does:
  - A trip-planning chat agent with NO tools — everything it "remembers" comes
    from the session's event history
  - output_key="last_reply" copies every final answer into session state, so
    you can watch state change in the State tab and see it survive a restart
  - Used to demo where adk web stores sessions:
      default                         -> code/01_session_basics/.adk/session.db (SQLite)
      --no_use_local_storage          -> in memory, gone when the server stops
      --session_service_uri sqlite:///sessions.db -> a SQLite file you choose

Try in ADK Web:
  - "I'm planning 5 days in Japan in April. I love food and temples."
  - "Suggest a day-by-day outline."
  - Stop adk web (Ctrl+C), start it again, click the same session in the
    Sessions list, then ask: "Remind me — how many days was my trip?"
  - Restart with `adk web --no_use_local_storage` — the Sessions list is empty.

Look at: the Sessions list (new / switch / resume after restart), the Events
tab (the whole conversation is just a list of events), the State tab (last_reply).
"""

from google.adk.agents import Agent

MODEL = "gemini-3.1-flash-lite"

root_agent = Agent(
    name="trip_planner",
    model=MODEL,
    description="A friendly trip-planning assistant.",
    instruction=(
        "You are a friendly trip-planning assistant. Use everything the user has "
        "told you earlier in this conversation (destination, dates, interests) "
        "when you answer. Keep answers short and practical — bullet points are good."
    ),
    output_key="last_reply",
)
