"""
Episode 04 — File 3: Retrieval with Cosine Similarity
======================================================
Run: python 03_retrieval.py

What this file does:
  - Loads the document, chunks it, embeds all chunks (reusing previous files)
  - Given a user query, embeds it and finds the top-k most relevant chunks
  - Uses numpy cosine similarity to rank chunks by relevance
  - Prints retrieved chunks with their similarity scores

This is Step 3 of the RAG pipeline: finding the right context for a question.
"""

import importlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from gemini_client import embed

_chunking = importlib.import_module("01_chunking")
load_text_file = _chunking.load_text_file
chunk_text = _chunking.chunk_text

_embedding = importlib.import_module("02_embed_chunks")
embed_chunks = _embedding.embed_chunks


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(vec_a)
    b = np.array(vec_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def retrieve(
    query: str,
    embedded_chunks: list[dict],
    top_k: int = 3,
) -> list[dict]:
    """
    Find the top-k most relevant chunks for a query.

    Args:
        query:           The user's question as a string.
        embedded_chunks: List of dicts with "index", "text", "embedding" keys.
        top_k:           Number of chunks to return.

    Returns:
        List of dicts: [{"index": int, "text": str, "score": float}, ...]
        sorted by relevance (highest score first).
    """
    query_embedding = embed(query)

    scored = []
    for chunk in embedded_chunks:
        score = cosine_similarity(query_embedding, chunk["embedding"])
        scored.append({
            "index": chunk["index"],
            "text": chunk["text"],
            "score": score,
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def print_results(query: str, results: list[dict]) -> None:
    """Pretty-print retrieval results."""
    print(f"\nQuery: \"{query}\"\n")
    for i, r in enumerate(results, 1):
        print(f"  #{i}  Chunk {r['index']} (score: {r['score']:.4f})")
        print(f"      {r['text'][:100]}...")
        print()


if __name__ == "__main__":
    filepath = os.path.join(os.path.dirname(__file__), "sample_data", "sample.txt")

    print("=" * 60)
    print("RETRIEVAL DEMO")
    print("=" * 60)

    # Build the index
    text = load_text_file(filepath)
    chunks = chunk_text(text, chunk_size=500, overlap=100)
    print(f"\nBuilding index: {len(chunks)} chunks...")
    embedded_chunks = embed_chunks(chunks)
    print("\nIndex ready.\n")

    # Test queries
    test_queries = [
        "What products does NovaTech sell?",
        "Who is the CTO?",
        "What is the return policy?"
    ]

    print("=" * 60)
    print("RUNNING TEST QUERIES")
    print("=" * 60)

    for query in test_queries:
        results = retrieve(query, embedded_chunks, top_k=3)
        print_results(query, results)
        print("-" * 60)

    print("\nRetrieval complete. Next step: full RAG chatbot (04_rag_chatbot.py)")
