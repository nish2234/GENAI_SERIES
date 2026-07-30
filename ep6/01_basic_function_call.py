"""
Episode 06 — File 1: Your First Function Call
==============================================
Run: python 01_basic_function_call.py

What this file does:
  - Defines a simple Python function (get_current_time)
  - Declares it as a tool for Gemini
  - Asks "what time is it?" — Gemini returns a function_call, NOT text
  - Executes the function and sends the result back
  - Gemini generates the final answer using the tool result

This is the complete function-calling flow in its simplest form.
"""

import os
from datetime import datetime
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL = "gemini-3.1-flash-lite"


# --- Step 1: Define a real Python function ---

def get_current_time() -> str:
    """Returns the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# --- Step 2: Declare the tool for Gemini ---

tools = [types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="get_current_time",
        description="Returns the current date and time.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={},
        ),
    ),
])]


# --- Step 3: Ask Gemini a question that needs this tool ---

user_question = "What is the current date and time?"
print(f"User: {user_question}\n")

response = client.models.generate_content(
    model=MODEL,
    contents=user_question,
    config=types.GenerateContentConfig(tools=tools),
)

part = response.candidates[0].content.parts[0]
print(f"Gemini returned: {part}\n")


# --- Step 4: Execute the function and send result back ---

if part.function_call:
    print(f"Tool requested: {part.function_call.name}")
    result = get_current_time()
    print(f"Tool returned:  {result}\n")

    function_response = types.Part.from_function_response(
        name="get_current_time",
        response={"result": result},
    )

    final = client.models.generate_content(
        model=MODEL,
        contents=[
            types.Content(role="user", parts=[types.Part.from_text(text=user_question)]),
            types.Content(role="model", parts=[part]),
            types.Content(role="user", parts=[function_response]),
        ],
        config=types.GenerateContentConfig(tools=tools),
    )
    print(f"Gemini says: {final.text}")
else:
    print(f"Gemini answered directly: {response.text}")
