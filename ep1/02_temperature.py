"""
Episode 01 — File 2: Understanding Temperature
===============================================
Run: python 02_temperature.py

What this file does:
  - Runs the SAME prompt at 3 different temperature values
  - Shows how temperature 0.0 = deterministic, 1.5 = creative/random
  - Also shows max_output_tokens in action

Temperature guide:
  0.0 – 0.3  →  Factual, consistent, structured (use for: code, data extraction, classification)
  0.5 – 0.8  →  Balanced, natural (use for: chatbots, summarisation)
  1.0 – 1.5  →  Creative, varied (use for: brainstorming, creative writing)
"""

import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

PROMPT = "Write a one-sentence tagline for a coffee shop."

print(f"Prompt: \"{PROMPT}\"\n")
print("=" * 60)

for temp in [0.0, 0.7, 1.5]:
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=PROMPT,
        config=types.GenerateContentConfig(
            temperature=temp,
            max_output_tokens=60,
        ),
    )
    print(f"\nTemperature {temp}:")
    print(f"  {response.text.strip()}")
    print(f"  (tokens used: {response.usage_metadata.total_token_count})")

print("\n" + "=" * 60)
print("\nTip: Run this script multiple times.")
print("Notice temperature 0.0 gives nearly the same answer every time.")
print("Temperature 1.5 gives a different answer on almost every run.")
