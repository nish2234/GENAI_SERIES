"""
Episode 18 — App 2: LangChain Tools inside ADK
===============================================
Run: cd code && adk web       (then pick "02_langchain_tools" in the dropdown)

What this app shows:
  - Any LangChain tool becomes an ADK tool with LangchainTool(tool).
  - Unlike CrewaiTool, name/description are OPTIONAL — LangChain names are
    already valid ("wikipedia", "arxiv"). You can still override them.
  - Two LangChain community tools that need NO API key:
        WikipediaQueryRun  -> wikipedia
        ArxivQueryRun      -> arxiv
  - One keyed tool, added only if TAVILY_API_KEY is in your .env:
        TavilySearch       -> tavily_search

What to try in ADK Web:
  - "Who created the Python programming language? Use Wikipedia."
  - "Find 3 recent arXiv papers about tool use in LLM agents"
  - "Compare what Wikipedia says about retrieval-augmented generation with the newest arXiv paper on it"
  - "What happened in AI news today?"   (needs TAVILY_API_KEY)

Panel to watch:
  - Events tab: function_call "wikipedia" with args {"query": ...} — the schema came from LangChain
  - Trace view: when the model calls wikipedia AND arxiv in one turn
"""

import os

from google.adk.agents import Agent
from google.adk.integrations.langchain import LangchainTool
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper

MODEL = "gemini-3.1-flash-lite"


wikipedia_tool = LangchainTool(
    WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=3000))
)

arxiv_tool = LangchainTool(
    ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=3, doc_content_chars_max=3000))
)

tools = [wikipedia_tool, arxiv_tool]

if os.getenv("TAVILY_API_KEY"):
    from langchain_tavily import TavilySearch

    tools.append(LangchainTool(TavilySearch(max_results=5)))


root_agent = Agent(
    name="knowledge_explorer",
    model=MODEL,
    description="Answers questions using Wikipedia, arXiv papers, and (optionally) Tavily web search.",
    instruction=(
        "You are a careful research librarian.\n"
        "- Use wikipedia for background facts about people, places, concepts and history.\n"
        "- Use arxiv for scientific papers; list title, authors and year.\n"
        "- If tavily_search is available, use it for today's news. If it is not, say "
        "live news search needs a TAVILY_API_KEY.\n"
        "When a question needs two sources, call both tools in the same turn."
    ),
    tools=tools,
)
