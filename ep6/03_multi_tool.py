"""
Episode 06 — File 5: Parallel Function Calling
================================================
Run: python 05_parallel_calls.py

What this file does:
  - Sends questions that require MULTIPLE tool calls at once
  - Shows how Gemini returns several function_calls in a single response
  - Dispatches all calls, collects all results, feeds them back together
  - Demonstrates both same-tool parallel (3 cities) and cross-tool parallel

Examples:
  "What's the weather in Tokyo, London, and Sydney?"
    → 3 parallel get_weather calls

  "What's the weather in Paris and convert 100 miles to km?"
    → get_weather + convert_units in parallel
"""

import os
import math
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL = "gemini-3.1-flash-lite"


# ── Tool implementations (same as 04) ────────────────────────────────

def calculate(expression: str) -> str:
    allowed = {"__builtins__": {}, "math": math, "abs": abs, "round": round}
    try:
        return str(eval(expression, allowed))
    except Exception as e:
        return f"Error: {e}"


def get_weather(city: str) -> str:
    weather_data = {
        "tokyo": "22°C, partly cloudy, humidity 65%",
        "london": "14°C, light rain, humidity 80%",
        "new york": "28°C, sunny, humidity 55%",
        "paris": "18°C, overcast, humidity 70%",
        "sydney": "16°C, clear skies, humidity 50%",
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")


def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    conversions = {
        ("miles", "km"): 1.60934, ("km", "miles"): 0.621371,
        ("fahrenheit", "celsius"): lambda v: (v - 32) * 5 / 9,
        ("celsius", "fahrenheit"): lambda v: v * 9 / 5 + 32,
        ("pounds", "kg"): 0.453592, ("kg", "pounds"): 2.20462,
    }
    key = (from_unit.lower(), to_unit.lower())
    factor = conversions.get(key)
    if factor is None:
        return f"Cannot convert from {from_unit} to {to_unit}"
    result = factor(value) if callable(factor) else value * factor
    return f"{value} {from_unit} = {result:.2f} {to_unit}"


def search_web(query: str) -> str:
    return (
        f"Search results for '{query}':\n"
        f"1. Wikipedia article about {query}\n"
        f"2. Recent news: Latest developments in {query}\n"
        f"3. Tutorial: Getting started with {query}"
    )


TOOL_FUNCTIONS = {
    "calculate": calculate,
    "get_weather": get_weather,
    "convert_units": convert_units,
    "search_web": search_web,
}


# ── Tool declarations ────────────────────────────────────────────────

tool_declarations = types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="calculate",
        description="Evaluate a mathematical expression and return the numeric result.",
        parameters=types.Schema(type=types.Type.OBJECT, properties={
            "expression": types.Schema(type=types.Type.STRING, description="Math expression"),
        }, required=["expression"]),
    ),
    types.FunctionDeclaration(
        name="get_weather",
        description="Get the current weather conditions for a given city.",
        parameters=types.Schema(type=types.Type.OBJECT, properties={
            "city": types.Schema(type=types.Type.STRING, description="City name"),
        }, required=["city"]),
    ),
    types.FunctionDeclaration(
        name="convert_units",
        description="Convert a value from one unit to another (miles/km, fahrenheit/celsius, etc.).",
        parameters=types.Schema(type=types.Type.OBJECT, properties={
            "value": types.Schema(type=types.Type.NUMBER, description="Numeric value"),
            "from_unit": types.Schema(type=types.Type.STRING, description="Source unit"),
            "to_unit": types.Schema(type=types.Type.STRING, description="Target unit"),
        }, required=["value", "from_unit", "to_unit"]),
    ),
    types.FunctionDeclaration(
        name="search_web",
        description="Search the web for information about a topic.",
        parameters=types.Schema(type=types.Type.OBJECT, properties={
            "query": types.Schema(type=types.Type.STRING, description="Search query"),
        }, required=["query"]),
    ),
])

config = types.GenerateContentConfig(tools=[tool_declarations])


# ── Run a question through the full tool-call loop ────────────────────

def ask(question: str) -> str:
    """Send a question, handle all tool calls (including parallel), return final answer."""
    print(f"\nUser: {question}")
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=question)])]

    response = client.models.generate_content(
        model=MODEL, contents=contents, config=config,
    )

    while True:
        print(response.candidates[0].content.parts )
        tool_calls = [p for p in response.candidates[0].content.parts if p.function_call]
        if not tool_calls:
            break

        print(f"  → Gemini requested {len(tool_calls)} tool call(s):")
        contents.append(types.Content(role="model", parts=response.candidates[0].content.parts))

        function_responses = []
        for part in tool_calls:
            fc = part.function_call
            args = dict(fc.args) if fc.args else {}
            fn = TOOL_FUNCTIONS.get(fc.name)
            result = fn(**args) if fn else f"Unknown tool: {fc.name}"
            print(f"    🔧 {fc.name}({args}) → {result}")

            function_responses.append(types.Part.from_function_response(
                name=fc.name, response={"result": result},
            ))

        contents.append(types.Content(role="user", parts=function_responses))

        response = client.models.generate_content(
            model=MODEL, contents=contents, config=config,
        )

    print(f"\nAssistant: {response.text}")
    return response.text


# ── Demo: parallel tool calls ────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("PARALLEL FUNCTION CALLING DEMO")
    print("=" * 60)

    ask("What is the weather in Tokyo, London, and Sydney?")

    print("\n" + "-" * 60)

    ask("What's the weather in Paris and convert 100 miles to kilometres?")

    print("\n" + "-" * 60)

    ask("Calculate 2 to the power of 20 and convert 72 fahrenheit to celsius")
