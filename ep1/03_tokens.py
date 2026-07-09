"""
Episode 01 — File 3: Token Counting
=====================================
Run: python 03_tokens.py

What this file does:
  - Counts tokens for different types of text BEFORE making a request
  - Shows how token count grows with complexity
  - Demonstrates the cost_tokens API call (no generation = free)

Why this matters:
  - Gemini API has per-token pricing and per-request token limits
  - In future episodes (RAG, agents) we will manage token budgets carefully
  - Always count before sending large documents or long conversations
"""

import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

MODEL = "gemini-3.1-flash-lite"

# Different texts to compare token counts
samples = [
    "Hello",
    "Hello, my name is Raj.",
    "Explain the history of artificial intelligence from Alan Turing to today.",
    "Write a Python function that takes a list of numbers and returns the top 5 sorted in descending order with their indices.",
    "You are a helpful assistant. The user will ask you questions about Python programming. "
    "Always provide working code examples. Always explain your code line by line. "
    "If you are unsure about something, say so clearly.",
]

print(f"{'Tokens':>7}  |  Text")
print("-" * 80)

for text in samples:
    result = client.models.count_tokens(model=MODEL, contents=text)
    preview = text[:65] + "..." if len(text) > 65 else text
    print(f"{result.total_tokens:>7}  |  {preview}")

# Practical example: check token count of a system prompt + user message together
print("\n" + "=" * 80)
print("Practical example: counting a full conversation before sending it\n")

system_prompt = "You are an expert Python developer. Answer concisely."
user_message = "How do I read a CSV file in Python and filter rows where the age column is greater than 30?"

combined = f"{system_prompt}\n\nUser: {user_message}"
result = client.models.count_tokens(model=MODEL, contents=combined)

print(f"System prompt tokens : {client.models.count_tokens(model=MODEL, contents=system_prompt).total_tokens}")
print(f"User message tokens  : {client.models.count_tokens(model=MODEL, contents=user_message).total_tokens}")
print(f"Combined total       : {result.total_tokens}")

