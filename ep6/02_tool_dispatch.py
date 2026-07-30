"""
Episode 06 — File 3: The Tool Dispatch Pattern
================================================
Run: python 03_tool_dispatch.py

What this file does:
  - Defines all 4 tools (calculate, get_weather, convert_units, search_web)
  - Builds a dispatcher: maps function name → Python function → execute → return
  - Sends a question to Gemini, dispatches the tool call, feeds result back
  - Shows the clean pattern every agent framework uses under the hood

This pattern is the bridge between "Gemini wants to call a tool" and
"the tool actually runs."
"""

import os
import math
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL = "gemini-3.1-flash-lite"


# --- Tool implementations ---

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
        ("inches", "cm"): 2.54, ("cm", "inches"): 0.393701,
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


# --- The dispatch pattern ---

TOOL_FUNCTIONS = {
    "calculate": calculate,
    "get_weather": get_weather,
    "convert_units": convert_units,
    "search_web": search_web,
}


def dispatch_tool_call(function_call) -> dict:
    """
    Takes a function_call from Gemini, looks up the real function,
    executes it, and returns the result as a dict.
    """
    name = function_call.name
    args = dict(function_call.args) if function_call.args else {}

    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        return {"error": f"Unknown tool: {name}"}

    print(f"  Dispatching: {name}({args})")
    result = fn(**args)
    print(f"  Result: {result}")
    return {"result": result}


# --- Tool declarations (same as 02) ---

tool_declarations = types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="calculate",
        description="Evaluate a mathematical expression and return the numeric result.",
        parameters=types.Schema(type=types.Type.OBJECT, properties={
            "expression": types.Schema(type=types.Type.STRING, description="Math expression, e.g. '2**10'"),
        }, required=["expression"]),
    ),
    types.FunctionDeclaration(
        name="get_weather",
        description="Get the current weather conditions for a given city.",
        parameters=types.Schema(type=types.Type.OBJECT, properties={
            "city": types.Schema(type=types.Type.STRING, description="City name, e.g. 'Tokyo'"),
        }, required=["city"]),
    ),
    types.FunctionDeclaration(
        name="convert_units",
        description="Convert a value from one unit to another (miles/km, fahrenheit/celsius, etc.).",
        parameters=types.Schema(type=types.Type.OBJECT, properties={
            "value": types.Schema(type=types.Type.NUMBER, description="Numeric value to convert"),
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


# --- Full flow: question → tool call → dispatch → result → final answer ---

question = "What is 2 to the power of 16?"
print(f"User: {question}\n")

response = client.models.generate_content(
    model=MODEL,
    contents=question,
    config=types.GenerateContentConfig(tools=[tool_declarations]),
)

part = response.candidates[0].content.parts[0]
print(part)

if part.function_call:
    result = dispatch_tool_call(part.function_call)

    function_response = types.Part.from_function_response(
        name=part.function_call.name,
        response=result,
    )

    final = client.models.generate_content(
        model=MODEL,
        contents=[
            types.Content(role="user", parts=[types.Part.from_text(text=question)]),
            types.Content(role="model", parts=[part]),
            types.Content(role="user", parts=[function_response]),
        ],
        config=types.GenerateContentConfig(tools=[tool_declarations]),
    )
    print(f"\nAssistant: {final.text}")
else:
    print(f"Assistant: {response.text}")
