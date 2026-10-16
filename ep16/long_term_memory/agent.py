"""
Episode 16 — App 04: Long-Term Memory
======================================
Run: cd code && adk web       (then pick "04_long_term_memory" from the dropdown)

What this app does:
  - A personal assistant that can recall facts from PAST sessions
  - load_memory (built-in tool) searches the memory service when the model
    decides past context would help
  - save_conversation_to_memory (our tool) calls
    tool_context.add_session_to_memory() to copy this session into memory
  - adk web uses an in-memory MemoryService by default: memory is shared
    across sessions while the server runs, and is lost when it restarts.
    For persistent memory, start adk web with --memory_service_uri
    (e.g. agentengine://<agent_engine_id> for Agent Platform Memory Bank).
  - Swap load_memory for preload_memory to search memory automatically at
    the start of every turn instead of on demand.

Try in ADK Web:
  - Session 1: "My sister Ana's birthday is March 14 and she loves orchids."
  - Session 1: "Please remember this conversation."
  - Click "New Session", then: "When is Ana's birthday, and what should I get her?"
  - Restart adk web, open a new session and ask again — memory is gone,
    even though the old sessions are still in the Sessions list.

Look at: the Events tab in session 2 — click the load_memory function_call
(the search query) and its function_response (the memories it found).
"""

from google.adk.agents import Agent
from google.adk.tools import ToolContext, load_memory

MODEL = "gemini-3.1-flash-lite"


async def save_conversation_to_memory(tool_context: ToolContext) -> dict:
    """Save the current conversation into long-term memory so it can be recalled in future sessions.

    Returns:
        A dict with status and a short confirmation message.
    """
    await tool_context.add_session_to_memory()
    return {"status": "success", "message": "This conversation is now in long-term memory."}


root_agent = Agent(
    name="personal_memory_agent",
    model=MODEL,
    description="A personal assistant that remembers facts across sessions.",
    instruction=(
        "You are a thoughtful personal assistant with long-term memory.\n"
        "- When the user asks about something that might have come up in a past "
        "conversation (people, dates, preferences, plans), call load_memory with "
        "a short keyword query before answering.\n"
        "- When the user shares personal facts they want kept, or asks you to "
        "remember the conversation, call save_conversation_to_memory.\n"
        "- If memory has nothing relevant, say so honestly — never invent facts.\n"
        "Be concise and warm."
    ),
    tools=[load_memory, save_conversation_to_memory],
)
