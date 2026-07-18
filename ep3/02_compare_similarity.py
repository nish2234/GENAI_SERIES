"""
Episode 03 — File 2: Compare Sentences with Cosine Similarity
==============================================================
Run: python 02_compare_similarity.py

What this file does:
  - Embeds pairs of sentences using Gemini's embedding model
  - Computes cosine similarity between each pair
  - Shows which pairs are semantically similar vs dissimilar
  - Demonstrates that embeddings capture meaning, not just keywords
"""

import os

import numpy as np
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

EMBEDDING_MODEL = "gemini-embedding-2"


def get_embedding(text: str) -> np.ndarray:
    """Embed a text and return as a numpy vector."""
    result = client.models.embed_content(model=EMBEDDING_MODEL, contents=text)
    return np.array(result.embeddings[0].values)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# --- Pairs to compare ---

pairs = [
    # Similar meaning, different words
    ("The cat sat on the mat", "A kitten was resting on the rug"),
    ("How to bake chocolate cookies", "Tips for making brownies at home"),
    ("Python is a great programming language", "I love writing code in Python"),
    ("The stock market crashed today", "Financial markets saw a major decline"),
    ("She sprinted across the finish line", "The runner completed the race at full speed"),

    # Completely different topics
    ("The cat sat on the mat", "Stock prices rose 5% today"),
    ("Python is a great programming language", "The recipe calls for two eggs and flour"),
    ("Black holes warp the fabric of spacetime", "Learning to play guitar takes practice"),
]

print("=== Cosine Similarity Between Sentence Pairs ===\n")
print("Similar pairs:")
print("-" * 60)

for text_a, text_b in pairs[:5]:
    vec_a = get_embedding(text_a)
    vec_b = get_embedding(text_b)
    sim = cosine_similarity(vec_a, vec_b)
    print(f"\n  Similarity: {sim:.4f}")
    print(f"  A: \"{text_a}\"")
    print(f"  B: \"{text_b}\"")

print(f"\n{'=' * 60}")
print("Dissimilar pairs:")
print("-" * 60)

for text_a, text_b in pairs[5:]:
    vec_a = get_embedding(text_a)
    vec_b = get_embedding(text_b)
    sim = cosine_similarity(vec_a, vec_b)
    print(f"\n  Similarity: {sim:.4f}")
    print(f"  A: \"{text_a}\"")
    print(f"  B: \"{text_b}\"")

print(f"\n{'=' * 60}")
print("\nNotice: similar pairs score 0.75–0.95, dissimilar pairs score 0.10–0.35.")
print("Embeddings capture MEANING, not just word overlap.")
