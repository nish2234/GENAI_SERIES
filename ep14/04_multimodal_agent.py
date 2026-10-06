"""
Episode 14 — File 4: The Multimodal Agent (MAIN FILE)
=====================================================
Run: python 04_multimodal_agent.py

What this file does:
  - A complete multimodal agent that handles BOTH text and image inputs
  - Vision tools: describe_image, extract_text, analyze_image
  - Text tools: calculate, search (stub)
  - Interactive loop: user types text OR provides image paths
  - The agent decides whether to use vision tools or text tools
  - Demonstrates combining multimodal capabilities with agent tooling

This is everything from Phase 2 coming together — function calling + vision + agent loop.
"""

import json
import math
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
from gemini_client import generate_with_image, get_client

load_dotenv()

MODEL = "gemini-3.1-flash-lite"


# =============================================================
# TOOL IMPLEMENTATIONS
# =============================================================

def describe_image(image_path: str) -> str:
    """Describe an image in detail."""
    return generate_with_image(
        prompt="Describe this image in detail. Include objects, colors, setting, people, text, and overall mood.",
        image_path=image_path,
        temperature=0.5,
        max_tokens=1024,
    )


def extract_text(image_path: str) -> str:
    """Extract all visible text from an image (OCR)."""
    return generate_with_image(
        prompt=(
            "Extract ALL text visible in this image. "
            "Return it exactly as written, preserving layout where possible. "
            "If there is no text, say 'No text found in image.'"
        ),
        image_path=image_path,
        temperature=0.1,
        max_tokens=2048,
    )


def analyze_image(image_path: str, question: str) -> str:
    """Answer a specific question about an image."""
    return generate_with_image(
        prompt=question,
        image_path=image_path,
        temperature=0.5,
        max_tokens=1024,
    )


def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return f"Error: Invalid characters in expression."
    try:
        result = eval(expression, {"__builtins__": {}}, {"math": math})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def search(query: str) -> str:
    """Search the web for information (stub for demo)."""
    return (
        f"[Search results for '{query}']\n"
        f"1. Result about {query} — example.com\n"
        f"2. More info on {query} — docs.example.com\n"
        f"3. Tutorial: Getting started with {query} — blog.example.com\n"
        "(This is a stub — in production, connect to a real search API)"
    )


# =============================================================
# TOOL DECLARATIONS FOR GEMINI
# =============================================================

tool_declarations = types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="describe_image",
        description="Describe an image in detail. Use when the user wants a description of what's in an image.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "image_path": types.Schema(
                    type=types.Type.STRING,
                    description="File path or URL of the image to describe",
                ),
            },
            required=["image_path"],
        ),
    ),
    types.FunctionDeclaration(
        name="extract_text",
        description="Extract all visible text from an image (OCR). Use when the user wants to read text from a photo, document, or screenshot.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "image_path": types.Schema(
                    type=types.Type.STRING,
                    description="File path or URL of the image to extract text from",
                ),
            },
            required=["image_path"],
        ),
    ),
    types.FunctionDeclaration(
        name="analyze_image",
        description="Answer a specific question about an image. Use when the user asks something specific about an image's content.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "image_path": types.Schema(
                    type=types.Type.STRING,
                    description="File path or URL of the image to analyze",
                ),
                "question": types.Schema(
                    type=types.Type.STRING,
                    description="The specific question to answer about the image",
                ),
            },
            required=["image_path", "question"],
        ),
    ),
    types.FunctionDeclaration(
        name="calculate",
        description="Evaluate a mathematical expression. Use for any math calculations.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "expression": types.Schema(
                    type=types.Type.STRING,
                    description="The math expression to evaluate (e.g., '25 * 4 + 10')",
                ),
            },
            required=["expression"],
        ),
    ),
    types.FunctionDeclaration(
        name="search",
        description="Search the web for information. Use when the user asks about facts, news, or information you don't have.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "query": types.Schema(
                    type=types.Type.STRING,
                    description="The search query",
                ),
            },
            required=["query"],
        ),
    ),
])


# =============================================================
# TOOL DISPATCHER
# =============================================================

TOOL_MAP = {
    "describe_image": lambda args: describe_image(args["image_path"]),
    "extract_text": lambda args: extract_text(args["image_path"]),
    "analyze_image": lambda args: analyze_image(args["image_path"], args["question"]),
    "calculate": lambda args: calculate(args["expression"]),
    "search": lambda args: search(args["query"]),
}


def dispatch_tool(function_call) -> str:
    """Execute a tool call and return the result."""
    name = function_call.name
    args = dict(function_call.args) if function_call.args else {}

    print(f"  🔧 Calling tool: {name}")
    print(f"     Args: {json.dumps(args, indent=6)}")

    if name in TOOL_MAP:
        result = TOOL_MAP[name](args)
        print(f"  ✓ Tool returned ({len(result)} chars)")
        return result
    else:
        return f"Error: Unknown tool '{name}'"


# =============================================================
# AGENT LOOP
# =============================================================

def run_agent():
    """Main agent loop — handles text + image inputs with tool calling."""
    client = get_client()
    conversation: list[types.Content] = []

    system_instruction = (
        "You are a helpful multimodal AI assistant. You can analyze images, extract text, "
        "answer questions, and do calculations.\n\n"
        "When the user provides an image path or URL, use the appropriate vision tool:\n"
        "- describe_image: for general description\n"
        "- extract_text: for reading text/OCR from documents, receipts, signs\n"
        "- analyze_image: for answering specific questions about an image\n\n"
        "When the user asks a math question, use calculate.\n"
        "When the user asks for information, use search.\n\n"
        "If the user provides a file path or URL that looks like an image "
        "(ends in .jpg, .png, .webp, .gif, or is a URL to an image), "
        "proactively use a vision tool on it.\n\n"
        "Always be helpful, concise, and accurate."
    )

    config = types.GenerateContentConfig(
        tools=[tool_declarations],
        system_instruction=system_instruction,
        temperature=0.7,
        max_output_tokens=2048,
    )

    print("=" * 60)
    print("MULTIMODAL AGENT — Vision + Text")
    print("=" * 60)
    print("\nI can analyze images, extract text, answer questions, and calculate.")
    print("\nTry:")
    print("  • Paste an image URL to analyze it")
    print("  • Ask a math question")
    print("  • Ask me to extract text from a document image")
    print("  • Ask any general question")
    print("\nType 'quit' to exit.")
    print("-" * 60)

    while True:
        user_input = input("\nYou: ").strip()

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("\nGoodbye! 👋")
            break

        conversation.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_input)],
            )
        )

        # Generate response (may include tool calls)
        response = client.models.generate_content(
            model=MODEL,
            contents=conversation,
            config=config,
        )

        # Handle tool call loop
        while response.candidates[0].content.parts[0].function_call:
            model_content = response.candidates[0].content
            conversation.append(model_content)

            # Process all tool calls in the response
            tool_results = []
            for part in model_content.parts:
                if part.function_call:
                    result = dispatch_tool(part.function_call)
                    tool_results.append(
                        types.Part.from_function_response(
                            name=part.function_call.name,
                            response={"result": result},
                        )
                    )

            # Send tool results back to the model
            conversation.append(
                types.Content(role="user", parts=tool_results)
            )

            response = client.models.generate_content(
                model=MODEL,
                contents=conversation,
                config=config,
            )

        # Get the final text response
        final_text = response.text
        conversation.append(response.candidates[0].content)

        print(f"\nAgent: {final_text}")


# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":
    print("\n🤖 Starting Multimodal Agent...\n")

    run_agent()
