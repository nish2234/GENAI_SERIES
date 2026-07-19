"""
Episode 04 — File 2: Embedding Chunks
======================================
Run: python 02_embed_chunks.py

What this file does:
  - Loads the sample document and chunks it (reusing logic from 01_chunking.py)
  - Embeds each chunk using Google's text-embedding-004 model
  - Stores embeddings alongside the chunk text in a list of dicts
  - Prints the structure so you can see what the "in-memory vector store" looks like

This is Step 2 of the RAG pipeline: turning text chunks into searchable vectors.
"""

import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from gemini_client import embed

_chunking = importlib.import_module("01_chunking")
load_text_file = _chunking.load_text_file
chunk_text = _chunking.chunk_text


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed each chunk and return enriched dicts with the embedding vector.

    Args:
        chunks: List of {"index": int, "text": str} dicts from chunk_text().

    Returns:
        List of {"index": int, "text": str, "embedding": list[float]} dicts.
    """
    embedded = []
    for chunk in chunks:
        print(f"  Embedding chunk {chunk['index']}...", end=" ", flush=True)
        vector = embed(chunk["text"])
        embedded.append({
            "index": chunk["index"],
            "text": chunk["text"],
            "embedding": vector,
        })
        print(f"done ({len(vector)} dimensions)")
    return embedded


if __name__ == "__main__":
    filepath = os.path.join(os.path.dirname(__file__), "sample_data", "sample.txt")

    print("=" * 60)
    print("CHUNK EMBEDDING DEMO")
    print("=" * 60)

    # Load and chunk
    text = load_text_file(filepath)
    chunks = chunk_text(text, chunk_size=500, overlap=100)
    print(f"\nLoaded document: {len(text):,} characters → {len(chunks)} chunks\n")

    # Embed all chunks
    print("Embedding chunks using text-embedding-004...\n")
    embedded_chunks = embed_chunks(chunks)

    # Show the structure
    print(f"\n{'=' * 60}")
    print("IN-MEMORY VECTOR STORE STRUCTURE")
    print("=" * 60)
    print(f"\nTotal entries: {len(embedded_chunks)}")
    print(f"Embedding dimensions: {len(embedded_chunks[0]['embedding'])}")

    print("\nSample entry (chunk 0):")
    sample = embedded_chunks[0]
    print(f"  index:     {sample['index']}")
    print(f"  text:      {sample['text'][:80]}...")
    print(f"  embedding: [{sample['embedding'][0]:.6f}, {sample['embedding'][1]:.6f}, "
          f"{sample['embedding'][2]:.6f}, ... ] ({len(sample['embedding'])} values)")

    print(f"\n{'=' * 60}")
    print("Embeddings ready. Next step: build retrieval (03_retrieval.py)")
    print("=" * 60)
