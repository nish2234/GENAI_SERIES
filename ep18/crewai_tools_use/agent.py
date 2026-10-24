"""
Episode 18 — App 1: CrewAI Tools inside ADK
============================================
Run: cd code && adk web       (then pick "01_crewai_tools" in the dropdown)

What this app shows:
  - Any CrewAI tool becomes an ADK tool with CrewaiTool(tool, name=..., description=...).
  - `name` is REQUIRED: CrewAI names have spaces ("Read website content"),
    ADK tool names can't. Pick a snake_case name.
  - Three CrewAI tools that need NO API key:
        DirectoryReadTool  -> list_research_notes
        FileReadTool       -> read_note
        ScrapeWebsiteTool  -> scrape_website
  - One keyed tool, added only if SERPER_API_KEY is in your .env:
        SerperDevTool      -> search_web

What to try in ADK Web:
  - "What notes do I have?"
  - "Read my story ideas and summarize them in 3 bullets"
  - "Scrape https://adk.dev and tell me what ADK is in two sentences"
  - "Search the web for this week's news about Google ADK"   (needs SERPER_API_KEY)

Panel to watch:
  - Events tab: function_call names are YOUR snake_case names (list_research_notes, scrape_website...)
  - Click a function_response: it's CrewAI's raw text output, wrapped by ADK
"""

import os
from pathlib import Path

from crewai_tools import DirectoryReadTool, FileReadTool, ScrapeWebsiteTool, SerperDevTool
from google.adk.agents import Agent
from google.adk.integrations.crewai import CrewaiTool
from dotenv import load_dotenv
load_dotenv()

MODEL = "gemini-3.1-flash-lite"
NOTES_DIR = str(Path(__file__).parent / "research_notes")


list_notes_tool = CrewaiTool(
    DirectoryReadTool(directory=NOTES_DIR),
    name="list_research_notes",
    description="List every file in the user's research notes folder. Returns full file paths.",
)

read_note_tool = CrewaiTool(
    FileReadTool(base_dir=NOTES_DIR),
    name="read_note",
    description="Read the text of one research note. Pass the full file_path from list_research_notes.",
)

scrape_tool = CrewaiTool(
    ScrapeWebsiteTool(),
    name="scrape_website",
    description="Download a web page and return its readable text. Pass a full https:// URL.",
)

tools = [list_notes_tool, read_note_tool, scrape_tool]

if os.getenv("SERPER_API_KEY"):
    tools.append(
        CrewaiTool(
            SerperDevTool(n_results=5),
            name="search_web",
            description="Search Google (via Serper) for current web results and news. Pass a search_query.",
        )
    )


root_agent = Agent(
    name="news_scout",
    model=MODEL,
    description="A research scout that reads local notes, scrapes pages, and searches the web.",
    instruction=(
        "You are a research scout for a tech newsletter.\n"
        "- Use list_research_notes, then read_note, for anything about the user's notes.\n"
        "- Use scrape_website to read a specific URL.\n"
        "- If search_web is available, use it for current news; include words like "
        "'news' and the current month in the query. If it is not available, say that "
        "web search needs a SERPER_API_KEY.\n"
        "Keep answers short and cite the file or URL you used."
    ),
    tools=tools,
)
