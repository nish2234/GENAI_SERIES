"""
Episode 04 — File 1: Document Chunking
=======================================
Run: python 01_chunking.py

What this file does:
  - Loads a text file from disk
  - Splits it into overlapping chunks using a sliding window
  - Demonstrates different chunk sizes and overlaps
  - Prints each chunk with its index so you can see the structure

This is Step 1 of the RAG pipeline: preparing your document for embedding.
"""

import os


def load_text_file(filepath: str) -> str:
    """Load a text file and return its contents as a string."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[dict]:
    """
    Split text into overlapping chunks using a sliding window.

    Args:
        text:       The full document text.
        chunk_size: Number of characters per chunk.
        overlap:    Number of characters that overlap between consecutive chunks.

    Returns:
        List of dicts: [{"index": 0, "text": "..."}, ...]
    """
    chunks = []
    start = 0
    index = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append({"index": index, "text": chunk})
            index += 1
        start += chunk_size - overlap

    return chunks


def print_chunks(chunks: list[dict], show_full: bool = False) -> None:
    """Print chunks with their index and a preview of the text."""
    for chunk in chunks:
        preview = chunk["text"] if show_full else chunk["text"][:80] + "..."
        print(f"  Chunk {chunk['index']:>2}: ({len(chunk['text']):>4} chars) {preview}")


if __name__ == "__main__":
    filepath = os.path.join(os.path.dirname(__file__), "sample_data", "sample.txt")

    print("=" * 60)
    print("DOCUMENT CHUNKING DEMO")
    print("=" * 60)

    text = load_text_file(filepath)
    print(f"\nLoaded: {filepath}")
    print(f"Document length: {len(text):,} characters\n")

    # --- Default chunking: 500 chars, 100 overlap ---
    print("-" * 60)
    print("Chunk size: 500 | Overlap: 100")
    print("-" * 60)
    chunks = chunk_text(text, chunk_size=500, overlap=100)
    print(f"Total chunks: {len(chunks)}\n")
    print_chunks(chunks)



    # --- Try different sizes ---
    # print(f"\n{'=' * 60}")
    # print("EXPERIMENTING WITH CHUNK SIZES")
    # print("=" * 60)

    # for size, olap in [(200, 50), (500, 100), (1000, 200)]:
    #     c = chunk_text(text, chunk_size=size, overlap=olap)
    #     print(f"\n  size={size:>4}, overlap={olap:>3}  →  {len(c):>3} chunks")

    # print(f"\n{'=' * 60}")
    # print("Chunking complete. Next step: embed these chunks (02_embed_chunks.py)")
    # print("=" * 60)
