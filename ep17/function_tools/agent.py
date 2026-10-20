"""
Episode 17 — App 1: Function Tools (good vs bad docstrings)
============================================================
Run: cd code && adk web --reload_agents   (then pick "01_function_tools" in the dropdown)
     --reload_agents makes ADK Web pick up your edits when you save this file.

What this app shows:
  - ADK turns a plain Python function into a tool automatically:
      function name   -> tool name
      docstring       -> tool description + parameter descriptions
      type hints      -> parameter types
      default values  -> optional parameters
  - Two versions of the same currency tool: a GOOD one and a BAD one.
    Swap ACTIVE_TOOLS below to see how much the docstring matters.

What to try in ADK Web:
  - "What's the weather in Tokyo?"
  - "What's the weather in Paris in fahrenheit?"
  - "How much is 250 US dollars in euros?"
  - "Convert 5000 rupees to yen" (then swap to the BAD tools and ask again)

Panel to watch:
  - Events tab -> click the function_call event: see the arguments Gemini chose
  - Events tab -> click the function_response event: see the dict you returned
"""

from google.adk.agents import Agent

MODEL = "gemini-3.5-flash-lite"

FAKE_WEATHER = {
    "tokyo": {"temp_c": 18, "condition": "light rain"},
    "paris": {"temp_c": 14, "condition": "cloudy"},
    "mumbai": {"temp_c": 31, "condition": "humid and sunny"},
    "new york": {"temp_c": 9, "condition": "windy"},
    "london": {"temp_c": 11, "condition": "drizzle"},
}

FAKE_RATES_TO_USD = {
    "USD": 1.0,
    "EUR": 1.12,
    "GBP": 1.31,
    "INR": 0.0103,
    "JPY": 0.0068,
}


# ---------------------------------------------------------------------------
# GOOD tools: clear names, type hints, Google-style docstrings, dict returns
# ---------------------------------------------------------------------------

def get_weather(city: str, unit: str = "celsius") -> dict:
    """Get the current weather for a city.

    Args:
        city: The city name, for example 'Tokyo' or 'New York'.
        unit: Temperature unit, either 'celsius' or 'fahrenheit'. Defaults to 'celsius'.

    Returns:
        A dict with status, city, temperature, unit and condition.
    """
    data = FAKE_WEATHER.get(city.lower())
    if data is None:
        return {
            "status": "error",
            "error_message": f"No weather data for '{city}'. Known cities: {', '.join(FAKE_WEATHER)}.",
        }

    temp = data["temp_c"]
    if unit.lower() == "fahrenheit":
        temp = round(temp * 9 / 5 + 32, 1)

    return {
        "status": "success",
        "city": city,
        "temperature": temp,
        "unit": unit.lower(),
        "condition": data["condition"],
    }


def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    """Convert an amount of money from one currency to another.

    Args:
        amount: The amount of money to convert, for example 250.
        from_currency: Three-letter ISO code of the source currency, for example 'USD'.
        to_currency: Three-letter ISO code of the target currency, for example 'EUR'.

    Returns:
        A dict with status, the original amount, and the converted amount.
    """
    src, dst = from_currency.upper(), to_currency.upper()
    if src not in FAKE_RATES_TO_USD or dst not in FAKE_RATES_TO_USD:
        return {
            "status": "error",
            "error_message": f"Unsupported currency. Supported: {', '.join(FAKE_RATES_TO_USD)}.",
        }

    converted = amount * FAKE_RATES_TO_USD[src] / FAKE_RATES_TO_USD[dst]
    return {
        "status": "success",
        "amount": amount,
        "from_currency": src,
        "to_currency": dst,
        "converted_amount": round(converted, 2),
    }


# ---------------------------------------------------------------------------
# BAD tool: same logic, terrible interface. The model has to guess.
# ---------------------------------------------------------------------------

def cc(a: float, b: str, c: str) -> dict:
    """Does conversion."""
    return convert_currency(a, b, c)


GOOD_TOOLS = [get_weather, convert_currency]
BAD_TOOLS = [get_weather, cc]

# On camera: switch this to BAD_TOOLS, save, click New Session in ADK Web.
ACTIVE_TOOLS = GOOD_TOOLS


root_agent = Agent(
    name="travel_helper",
    model=MODEL,
    description="Answers weather and currency questions for travellers.",
    instruction=(
        "You help travellers with weather and money questions. "
        "Use the available tools to get real numbers instead of guessing. "
        "If a tool returns status 'error', tell the user what went wrong "
        "and what they can ask instead."
    ),
    tools=BAD_TOOLS ,
)
