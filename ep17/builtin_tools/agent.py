"""
Episode 17 — App 3: Built-in Tools (Google Search + Code Execution)
====================================================================
Run: cd code && adk web       (then pick "03_builtin_tools" in the dropdown)

What this app shows:
  - google_search: Gemini grounds its answer with live Google Search results.
  - BuiltInCodeExecutor: Gemini writes Python and runs it in Google's sandbox.
  - The documented limitation: Code Execution (and Google Search, unless you
    opt out) must be used BY ITSELF in an agent — you can't drop them into a
    tools list next to your own functions.
  - The documented workaround: put each built-in in its own specialist agent
    and give those agents to the root agent as tools with AgentTool.

    NOT supported (one agent mixing a code executor with function tools):
        Agent(tools=[get_today], code_executor=BuiltInCodeExecutor())

    Supported (this file):
        root_agent  -> tools=[AgentTool(search_agent), AgentTool(code_agent), get_today]

What to try in ADK Web:
  - "What's the latest news about Google ADK?"                   (search)
  - "What is the 30th Fibonacci number? Run code to be sure."    (code)
  - "How many days until December 31st?"                         (function tool + code)
  - "Find the current population of Japan and Canada, then compute the ratio."

Panel to watch:
  - Events tab: the root agent's function_call targets "search_agent" or "code_agent"
  - Click into the code_agent events to see executable_code and code_execution_result
"""

from datetime import date

from google.adk.agents import Agent
from google.adk.code_executors import BuiltInCodeExecutor
from google.adk.tools import google_search
from google.adk.tools.agent_tool import AgentTool


MODEL = "gemini-3.1-flash-lite"


search_agent = Agent(
    name="search_agent",
    model=MODEL,
    description="Searches Google for fresh, real-world facts and news.",
    instruction=(
        "You are a Google Search specialist. Search for the request, then answer "
        "concisely with the key facts and mention your sources."
    ),
    tools=[google_search],
)

code_agent = Agent(
    name="code_agent",
    model=MODEL,
    description="Writes and runs Python code for exact calculations and data work.",
    instruction=(
        "You are a Python specialist. For every request, write Python code, run it, "
        "and report the exact result. Never do arithmetic in your head."
    ),
    code_executor=BuiltInCodeExecutor(),
)


def get_today() -> dict:
    """Get today's date. Use this whenever a question depends on the current date.

    Returns:
        A dict with today's date in ISO format and the weekday name.
    """
    today = date.today()
    return {"status": "success", "date": today.isoformat(), "weekday": today.strftime("%A")}


root_agent = Agent(
    name="research_assistant",
    model=MODEL,
    description="A research assistant that can search the web and run code.",
    instruction=(
        "You are a research assistant with three tools:\n"
        "- search_agent: for anything current, factual, or newsworthy.\n"
        "- code_agent: for any calculation, counting, or data crunching.\n"
        "- get_today: whenever the answer depends on today's date.\n"
        "Combine them when needed: e.g. get facts with search_agent, then pass "
        "the numbers to code_agent. Give a short, clear final answer."
    ),
    tools=[
        AgentTool(agent=search_agent),
        AgentTool(agent=code_agent),
        get_today,
    ],
)
