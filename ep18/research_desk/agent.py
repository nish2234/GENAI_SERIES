"""
Episode 18 — App 5: The Research Desk (CrewAI + LangChain + OpenAPI + AgentTool)
=================================================================================
Run: cd code && adk web       (then pick "05_research_desk" in the dropdown)

What this app shows — one desk chief, three specialist reporters, one file writer:

    research_desk (root)
      ├── AgentTool(web_reporter)        CrewAI:    ScrapeWebsiteTool (+ SerperDevTool if SERPER_API_KEY)
      ├── AgentTool(library_reporter)    LangChain: Wikipedia + arXiv  (+ TavilySearch if TAVILY_API_KEY)
      ├── AgentTool(community_reporter)  OpenAPI:   Hacker News search API (hn_api.yaml)
      └── save_report                    CrewAI:    FileWriterTool -> 05_research_desk/reports/

  Each specialist keeps its own tools, so the root agent sees just 4 clean tools
  instead of 10+ raw ones.

What to try in ADK Web:
  - "Brief me on the Model Context Protocol: background, key papers, and what HN thinks"
  - "Research Google ADK using https://adk.dev plus community reaction, then save the brief"
  - "What's the latest on AI coding agents?"   (richer with SERPER_API_KEY / TAVILY_API_KEY)

Panel to watch:
  - Events tab: the root calls web_reporter, library_reporter and community_reporter
    (often in the same turn), then save_report
  - Trace view: three reporter branches running under one desk turn
  - Your editor: a new markdown file appears in 05_research_desk/reports/
"""

import os
from pathlib import Path

from crewai_tools import FileWriterTool, ScrapeWebsiteTool, SerperDevTool
from google.adk.agents import Agent
from google.adk.integrations.crewai import CrewaiTool
from google.adk.integrations.langchain import LangchainTool
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.openapi_tool.openapi_spec_parser.openapi_toolset import OpenAPIToolset
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper

MODEL = "gemini-3.1-flash-lite"

APP_DIR = Path(__file__).parent
REPORTS_DIR = str(APP_DIR / "reports")

REPORTER_RULES = (
    "Return 4-8 bullet points of facts, each with its source (URL, paper title, or HN link). "
    "No intro, no conclusion."
)


# --- Web reporter: CrewAI tools ------------------------------------------------
web_tools = [
    CrewaiTool(
        ScrapeWebsiteTool(),
        name="scrape_website",
        description="Download a web page and return its readable text. Pass a full https:// URL.",
    ),
]
if os.getenv("SERPER_API_KEY"):
    web_tools.append(
        CrewaiTool(
            SerperDevTool(n_results=5),
            name="search_web",
            description="Search Google (via Serper) for current web results and news. Pass a search_query.",
        )
    )

web_reporter = Agent(
    name="web_reporter",
    model=MODEL,
    description="Reads official websites and (if configured) searches the live web. Give it a topic and any URLs.",
    instruction=(
        "You cover the open web. Scrape any URL you are given. If search_web is available, "
        "use it to find current coverage. " + REPORTER_RULES
    ),
    tools=web_tools,
)


# --- Library reporter: LangChain tools -----------------------------------------
library_tools = [
    LangchainTool(
        WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=3000))
    ),
    LangchainTool(ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=3, doc_content_chars_max=3000))),
]
if os.getenv("TAVILY_API_KEY"):
    from langchain_tavily import TavilySearch

    library_tools.append(LangchainTool(TavilySearch(max_results=5)))

library_reporter = Agent(
    name="library_reporter",
    model=MODEL,
    description="Finds background (Wikipedia) and research papers (arXiv) on a topic.",
    instruction=(
        "You cover background and research. Use wikipedia for context and arxiv for papers "
        "(title, authors, year). If tavily_search is available, use it for recent news. "
        + REPORTER_RULES
    ),
    tools=library_tools,
)


# --- Community reporter: OpenAPI tools -----------------------------------------
hn_toolset = OpenAPIToolset(
    spec_str=(APP_DIR / "hn_api.yaml").read_text(),
    spec_str_type="yaml",
)

community_reporter = Agent(
    name="community_reporter",
    model=MODEL,
    description="Finds what the developer community on Hacker News is saying about a topic.",
    instruction=(
        "You cover developer community reaction on Hacker News. Use search_stories with "
        "tags='story' and hits_per_page=5 for popular discussions, and search_stories_by_date "
        "for the latest. "
        "Link stories as https://news.ycombinator.com/item?id=<objectID> and include points. "
        + REPORTER_RULES
    ),
    tools=[hn_toolset],
)


# --- The desk ------------------------------------------------------------------
save_report_tool = CrewaiTool(
    FileWriterTool(base_dir=REPORTS_DIR),
    name="save_report",
    description=(
        "Save a finished research brief as a markdown file in the reports folder. "
        "Pass filename (e.g. 'mcp-brief.md') and content. Set overwrite to true to replace a file."
    ),
)

root_agent = Agent(
    name="research_desk",
    model=MODEL,
    description="A research desk that briefs you on any tech topic using three specialist reporters.",
    instruction=(
        "You run a research desk. For any topic:\n"
        "1. Call web_reporter, library_reporter and community_reporter — in parallel when "
        "you can. Pass each one the topic plus any URLs the user gave.\n"
        "2. Combine their facts into a brief with these sections: Summary (3 sentences), "
        "Background, Key Research, Community Pulse, Sources.\n"
        "3. Only call save_report when the user asks to save; use a short kebab-case .md filename.\n"
        "Never invent facts that no reporter returned."
    ),
    tools=[
        AgentTool(agent=web_reporter),
        AgentTool(agent=library_reporter),
        AgentTool(agent=community_reporter),
        save_report_tool,
    ],
)
