"""
Episode 10 — File 1: Web Search Tool (DuckDuckGo)
===================================================
Run: python 01_search_tool.py

What this file does:
  - Builds a web search tool using the duckduckgo-search package
  - Function search_web(query, max_results) returns structured results
  - Each result has: title, url, snippet
  - Completely free — no API key needed
  - Demo searches at the bottom to verify it works

This is the first tool our research agent will use.
"""

from duckduckgo_search import DDGS


def search_web(query: str, max_results: int = 5) -> list[dict]:
    """
    Search the web using DuckDuckGo.

    Returns a list of dicts, each with:
      - title: page title
      - url: page URL
      - snippet: short description/excerpt
    """
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                })
    except Exception as e:
        print(f"  [Search Error] {e}")
        return []

    return results


def format_results(results: list[dict]) -> str:
    """Format search results into a readable string for the LLM."""
    if not results:
        return "No results found."

    formatted = []
    for i, r in enumerate(results, 1):
        formatted.append(
            f"{i}. {r['title']}\n"
            f"   URL: {r['url']}\n"
            f"   {r['snippet']}"
        )
    return "\n\n".join(formatted)


# --- Demo ---
if __name__ == "__main__":
    print("=" * 60)
    print("WEB SEARCH TOOL — DuckDuckGo (free, no API key)")
    print("=" * 60)

    query1 = "latest developments in quantum computing 2026"
    print(f"\nSearch: '{query1}'")
    print("-" * 60)
    results1 = search_web(query1, max_results=5)
    print(format_results(results1))

    print("\n" + "=" * 60)

    query2 = "how does CRISPR gene editing work"
    print(f"\nSearch: '{query2}'")
    print("-" * 60)
    results2 = search_web(query2, max_results=3)
    print(format_results(results2))

    print("\n" + "=" * 60)

    query3 = "Python async programming best practices"
    print(f"\nSearch: '{query3}'")
    print("-" * 60)
    results3 = search_web(query3, max_results=3)
    print(format_results(results3))

    print(f"\nDone! Returned {len(results1) + len(results2) + len(results3)} total results.")
