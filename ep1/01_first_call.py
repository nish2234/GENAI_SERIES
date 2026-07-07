"""
Episode 01 — File 1: Your First Gemini API Call
================================================
Run: python 01_first_call.py

What this file does:
  - Loads your API key from .env
  - Creates a Gemini client
  - Sends a prompt and gets a response
  - Prints the response text AND the full metadata (tokens, finish reason)
"""

import os
from dotenv import load_dotenv
from google import genai

# Load GEMINI_API_KEY from .env file
load_dotenv()

# Create the client — this is all the setup you need
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# --- Make your first API call ---
response = client.models.generate_content(
    model="gemini-2.5-flash-lite",
    contents="Explain what a Large Language Model is in 3 sentences.",
)

# Print the response text
print("=== Response ===")
print(response.text)

# Print the full metadata — always useful to understand what happened
print("\n=== Metadata ===")
print(f"Model version   : {response.model_version}")
print(f"Finish reason   : {response.candidates[0].finish_reason}")
print(f"Input tokens    : {response.usage_metadata.prompt_token_count}")
print(f"Output tokens   : {response.usage_metadata.candidates_token_count}")
print(f"Total tokens    : {response.usage_metadata.total_token_count}")
