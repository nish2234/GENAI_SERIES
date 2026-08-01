"""
01 — JSON Mode: The Simplest Structured Output
================================================

The easiest way to get valid JSON from Gemini: set response_mime_type
to "application/json". The model is then constrained to only output
valid JSON — no extra text, no markdown fences, just parseable JSON.

Key parameter:
  response_mime_type="application/json"

Run:
  python 01_json_mode.py
"""

import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL = "gemini-3.1-flash-lite"





def demo_with_json_mode():
    """Show what happens WITH JSON mode — always valid JSON."""
    print("\n" + "=" * 60)
    print("WITH JSON MODE (guaranteed valid JSON)")
    print("=" * 60)

    response = client.models.generate_content(
        model=MODEL,
        contents=(
            "List 3 programming languages with their year of creation. "
            "Return as JSON with keys: name, year."
        ),
        config=types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
        ),
    )

    print(f"\nRaw response:\n{response.text}")

    data = json.loads(response.text)
    print(f"\nParsed successfully! Type: {type(data).__name__}")
    print(f"Data: {json.dumps(data, indent=2)}")



if __name__ == "__main__":
    #demo_without_json_mode()
    demo_with_json_mode()
    

    
