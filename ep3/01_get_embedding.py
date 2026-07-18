"""
Episode 03 — File 1: Get Your First Embedding
===============================================
Run: python 01_get_embedding.py

What this file does:
  - Sends a sentence to Gemini's text-embedding-004 model
  - Gets back a 768-dimensional embedding vector
  - Prints its shape, first few values, and basic stats
  - Shows what an embedding actually looks like as raw numbers
"""

import os

import numpy as np
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

EMBEDDING_MODEL = "gemini-embedding-2"

# --- Embed a single sentence ---

text = "Artificial intelligence is transforming how we build software."

result = client.models.embed_content(
    model=EMBEDDING_MODEL,
    contents=text,
)

embedding = result.embeddings[0].values
vector = np.array(embedding)

print(f"Text: \"{text}\"")
print(f"\nEmbedding dimensions: {vector.shape[0]}")
print(f"First 10 values:     {vector[:10]}")
print(f"Last 5 values:       {vector[-5:]}")
print(f"Min value:           {vector.min():.6f}")
print(f"Max value:           {vector.max():.6f}")
print(f"Mean value:          {vector.mean():.6f}")

# --- Embed a few more texts to see how dimensions stay the same ---

print("\n--- Embedding more texts ---\n")

texts = [
    "Hello",
    "The quick brown fox jumps over the lazy dog.",
    "Quantum computing uses qubits that can exist in superposition, allowing parallel computation.",
]

for t in texts:
    r = client.models.embed_content(model=EMBEDDING_MODEL, contents=t)
    v = np.array(r.embeddings[0].values)
    print(f"  [{v.shape[0]} dims] \"{t[:60]}\"")

print("\nEvery text — short or long — becomes the same size vector.")
print("That fixed size is what makes comparison possible.")
