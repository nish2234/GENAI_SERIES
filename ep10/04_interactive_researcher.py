"""
Episode 10 — File 4: Interactive Research Agent
================================================
Run: python 04_interactive_researcher.py

What this file does:
  - Same research agent as 03_research_agent.py
  - But with an interactive input loop
  - The viewer types their own research topics
  - Agent researches and writes reports for each one
  - Type 'quit' or 'exit' to stop

Perfect for viewers to experiment with their own topics.
"""

import importlib
from datetime import datetime
from gemini_client import generate, generate_structured

search_tool = importlib.import_module("01_search_tool")
scrape_tool = importlib.import_module("02_scrape_tool")
search_web = search_tool.search_web
format_results = search_tool.format_results
scrape_url = scrape_tool.scrape_url


SYSTEM_PROMPT = """\
You are a research agent. Your job is to research a topic thoroughly and produce a structured report.

You have access to these tools:
1. search_web(query) — Search the web. Returns titles, URLs, and snippets.
2. scrape_url(url) — Read the full content of a web page.
3. write_report(content) — Write your final research report.

Follow this process:
1. SEARCH: Start by searching for the topic. Do 2-3 different searches with varied queries.
2. READ: Pick the most promising URLs from search results and scrape them for details.
3. SYNTHESISE: Once you have enough information (after 2-4 search+read cycles), write a report.

Your report should be structured with:
- Title
- Executive Summary (2-3 sentences)
- Key Findings (bullet points)
- Detailed Analysis (paragraphs)
- Sources (list URLs you used)

IMPORTANT RULES:
- Do 2-4 research cycles before writing the report.
- Be specific in your search queries — don't be too broad.
- When scraping, pick URLs that look most relevant and authoritative.
- Your final report should be comprehensive but concise (300-500 words).
"""

REACT_SCHEMA = {
    "type": "object",
    "properties": {
        "thought": {
            "type": "string",
            "description": "Your reasoning about what to do next",
        },
        "action": {
            "type": "string",
            "enum": ["search_web", "scrape_url", "write_report"],
            "description": "Tool to call",
        },
        "input": {
            "type": "string",
            "description": "Input for the tool (search query, URL, or full report content)",
        },
    },
    "required": ["thought", "action", "input"],
}


def execute_tool(action: str, tool_input: str) -> str:
    """Execute a tool and return the observation."""
    if action == "search_web":
        results = search_web(tool_input, max_results=5)
        return format_results(results)
    elif action == "scrape_url":
        return scrape_url(tool_input)
    elif action == "write_report":
        return save_report(tool_input)
    else:
        return f"Unknown tool: {action}. Available tools: search_web, scrape_url, write_report"


def save_report(content: str) -> str:
    """Save the research report to a file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"research_report_{timestamp}.md"

    with open(filename, "w") as f:
        f.write(content)

    return f"Report saved to {filename}"


def research_agent(topic: str, max_steps: int = 10) -> str:
    """Run the research agent on a topic."""
    print(f"\n{'='*60}")
    print(f"  RESEARCHING: {topic}")
    print(f"{'='*60}\n")
    print("  The agent will search, read, and write a report...")
    print("  This may take 30-60 seconds.\n")

    conversation = (
        f"Research this topic thoroughly: {topic}\n\n"
        f"Available tools:\n"
        f"  - search_web(query): Search the web\n"
        f"  - scrape_url(url): Read a web page\n"
        f"  - write_report(content): Write the final report\n\n"
        f"Begin your research. Remember: search first, then read promising pages, "
        f"then write a comprehensive report after 2-4 research cycles."
    )

    report_content = None

    for step in range(1, max_steps + 1):
        print(f"\n  {'─'*50}")
        print(f"  Step {step}/{max_steps}")
        print(f"  {'─'*50}")

        parsed = generate_structured(
            conversation,
            response_schema=REACT_SCHEMA,
            temperature=0.3,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
        )

        thought = parsed.get("thought", "")
        action = parsed.get("action", "")
        tool_input = parsed.get("input", "")

        if thought:
            print(f"\n  Thought: {thought[:150]}")
        if action:
            print(f"  Action:  {action}")
        if tool_input and action != "write_report":
            print(f"  Input:   {tool_input[:80]}")
        elif action == "write_report":
            print(f"  Input:   [Writing report... {len(tool_input)} chars]")

        observation = execute_tool(action, tool_input)

        if action == "write_report":
            report_content = tool_input
            print(f"\n  {observation}")
            break

        obs_preview = observation[:150] + "..." if len(observation) > 150 else observation
        print(f"  Result:  {obs_preview}")

        conversation += (
            f"\n\nAssistant action: {action}({tool_input})"
            f"\n\nObservation: {observation}"
            f"\n\nContinue your research. If you have gathered enough information "
            f"(at least 2-3 searches and 2+ pages read), write the report. "
            f"Otherwise, keep searching and reading."
        )

    if report_content:
        print(f"\n{'='*60}")
        print(f"  RESEARCH REPORT")
        print(f"{'='*60}\n")
        print(report_content)
        print(f"\n{'='*60}")
    else:
        print(f"\n  Agent did not produce a report within {max_steps} steps.")

    return report_content or ""


def main():
    """Interactive research agent loop."""
    print("\n" + "=" * 60)
    print("  INTERACTIVE RESEARCH AGENT")
    print("  Powered by: DuckDuckGo + Web Scraping + Gemini")
    print("=" * 60)
    print("\n  Type a research topic and the agent will:")
    print("  1. Search the web for information")
    print("  2. Read promising web pages")
    print("  3. Synthesise a structured research report")
    print("\n  Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            topic = input("  Research topic: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n  Goodbye!")
            break

        if not topic:
            continue
        if topic.lower() in ("quit", "exit", "q"):
            print("\n  Goodbye! Happy researching.")
            break

        research_agent(topic)
        print()


if __name__ == "__main__":
    main()
