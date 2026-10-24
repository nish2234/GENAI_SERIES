"""
Episode 18 — App 3: OpenAPI Tools (any REST API -> agent tools)
================================================================
Run: cd code && adk web       (then pick "03_openapi_tools" in the dropdown)

What this app shows:
  - OpenAPIToolset reads an OpenAPI 3 spec and creates one RestApiTool per operation.
      operationId 'searchStories'       -> tool 'search_stories'
      operationId 'searchStoriesByDate' -> tool 'search_stories_by_date'
      operationId 'getUser'             -> tool 'get_user'
  - Tool descriptions come from each operation's summary/description;
    parameters come from the spec's parameters (also snake_cased:
    hitsPerPage -> hits_per_page, numericFilters -> numeric_filters).
  - The API: Hacker News search by Algolia — free, no key.
    The spec lives next to this file in hn_api.yaml.

What to try in ADK Web:
  - "What are people on Hacker News saying about Google ADK?"
  - "Show me the 5 newest Show HN posts about AI agents"
  - "Find stories about MCP with more than 100 points"
  - "How much karma does the user pg have?"

Panel to watch:
  - Events tab: function_call 'search_stories' with args like {"query": ..., "tags": "story"}
  - Click the function_response: that's the raw JSON from the real API
"""

from pathlib import Path

from google.adk.agents import Agent
from google.adk.tools.openapi_tool.openapi_spec_parser.openapi_toolset import OpenAPIToolset

MODEL = "gemini-3.1-flash-lite"

SPEC_PATH = Path(__file__).parent / "hn_api.yaml"

hn_toolset = OpenAPIToolset(
    spec_str=SPEC_PATH.read_text(),
    spec_str_type="yaml",
)


root_agent = Agent(
    name="hn_analyst",
    model=MODEL,
    description="Searches Hacker News stories and user profiles through its public REST API.",
    instruction=(
        "You analyse Hacker News using the API tools.\n"
        "- Use search_stories for 'what are people saying' or 'most popular' questions "
        "(tags='story', hits_per_page=5 unless asked otherwise).\n"
        "- Use search_stories_by_date for 'latest' or 'newest' questions.\n"
        "- Use numeric_filters like 'points>100' when the user wants popular items.\n"
        "- Use get_user for questions about a specific HN user.\n"
        "For each story give the title, points, comment count and the HN link "
        "https://news.ycombinator.com/item?id=<objectID>."
    ),
    tools=[hn_toolset],
)
