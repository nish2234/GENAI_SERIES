"""
Episode 17 — App 5: Multi-Tool Travel Desk (everything together)
=================================================================
Run: cd code && adk web       (then pick "05_multi_tool_assistant" in the dropdown)

What this app shows:
  - async tools: ADK runs the model's function calls IN PARALLEL when the
    tools are `async def` (3 cities x 2s each ~= 2s total, not 6s).
  - A real HTTP API tool (Frankfurter exchange rates, free, no key) with the
    ADK 2.0 error rule:
        * EXPECTED problem (unknown currency)   -> return {"status": "error", ...}
        * UNEXPECTED problem (network down, 500) -> let the exception propagate
          so ADK 2.0 can see it (and retry it under a RetryConfig).
        * NO broad `except Exception` anywhere.
  - Google Search next to our own function tools, using the documented
    GoogleSearchTool(bypass_multi_tools_limit=True) workaround.
  - ToolContext state ("user:trips") + a confirmation before booking.

What to try in ADK Web:
  - "Compare the weather in Tokyo, Lisbon and Cape Town"     (3 parallel calls)
  - "How many Japanese yen is 500 euros?"                    (real API)
  - "Convert 100 dollars to Klingon darseks"                 (expected error dict)
  - "Any big events happening in Lisbon next month?"         (Google Search)
  - "Book me 4 nights in Lisbon from 2026-11-12"             (confirmation prompt)
  - "What trips do I have booked?"

Panel to watch:
  - Events tab: ONE model event containing THREE function_calls for the weather prompt
  - Trace view: the three get_weather spans overlap (parallel), not stack up
  - State tab: 'user:trips' appears only after you approve the booking
"""

import asyncio

import httpx
from google.adk.agents import Agent
from google.adk.tools import FunctionTool, ToolContext
from google.adk.tools.google_search_tool import GoogleSearchTool

MODEL = "gemini-3.5-flash-lite"

FAKE_WEATHER = {
    "tokyo": {"temp_c": 18, "condition": "light rain"},
    "lisbon": {"temp_c": 21, "condition": "sunny"},
    "cape town": {"temp_c": 24, "condition": "clear and breezy"},
    "reykjavik": {"temp_c": 3, "condition": "snow showers"},
    "bangalore": {"temp_c": 27, "condition": "partly cloudy"},
}


async def get_weather(city: str) -> dict:
    """Get the current weather for ONE city.

    This tool is optimized for parallel execution: to compare several cities,
    call it once per city in the same turn.

    Args:
        city: The city name, for example 'Lisbon'.

    Returns:
        A dict with status, city, temperature in Celsius, and condition.
    """
    await asyncio.sleep(2)  # simulate a slow weather API
    data = FAKE_WEATHER.get(city.lower())
    if data is None:
        return {"status": "error", "error_message": f"No weather data for '{city}'."}
    return {"status": "success", "city": city, **data}


async def get_exchange_rate(from_currency: str, to_currency: str) -> dict:
    """Get the live exchange rate between two currencies.

    Args:
        from_currency: Three-letter ISO code to convert from, for example 'EUR'.
        to_currency: Three-letter ISO code to convert to, for example 'JPY'.

    Returns:
        A dict with status, the rate, and the date of the rate.
    """
    src, dst = from_currency.upper(), to_currency.upper()
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(
            "https://api.frankfurter.dev/v1/latest",
            params={"base": src, "symbols": dst},
        )

    if response.status_code == 404:
        return {
            "status": "error",
            "error_message": f"Unknown currency code in '{src}' -> '{dst}'. Use ISO codes like USD, EUR, JPY, INR.",
        }
    response.raise_for_status()

    data = response.json()
    return {"status": "success", "from": src, "to": dst, "rate": data["rates"][dst], "date": data["date"]}


def book_hotel(city: str, check_in: str, nights: int, tool_context: ToolContext) -> dict:
    """Book a hotel stay for the user.

    Args:
        city: The destination city.
        check_in: Check-in date in YYYY-MM-DD format.
        nights: Number of nights to stay.

    Returns:
        A dict with status and the booking details.
    """
    if nights < 1:
        return {"status": "error", "error_message": "Nights must be at least 1."}

    trips = tool_context.state.get("user:trips", [])
    booking = {
        "booking_id": f"BK-{len(trips) + 1:03d}",
        "city": city,
        "check_in": check_in,
        "nights": nights,
    }
    trips.append(booking)
    tool_context.state["user:trips"] = trips
    return {"status": "success", **booking}


def list_my_trips(tool_context: ToolContext) -> dict:
    """List all hotel bookings the user has made.

    Returns:
        A dict with status and the list of bookings.
    """
    return {"status": "success", "trips": tool_context.state.get("user:trips", [])}


root_agent = Agent(
    name="travel_desk",
    model=MODEL,
    description="A travel desk that checks weather, money, events and books hotels.",
    instruction=(
        "You are a friendly travel desk agent.\n"
        "- When users ask about several cities, call get_weather for each city in parallel.\n"
        "- Use get_exchange_rate for currency questions, then do the multiplication yourself.\n"
        "- Use google_search for events, opening hours, news, or anything current.\n"
        "- Use book_hotel to book; it needs the user's approval. Use list_my_trips to show bookings.\n"
        "If a tool returns status 'error', explain the problem and suggest a fix."
    ),
    tools=[
        get_weather,
        get_exchange_rate,
        GoogleSearchTool(bypass_multi_tools_limit=True),
        FunctionTool(book_hotel, require_confirmation=True),
        list_my_trips,
    ],
)
