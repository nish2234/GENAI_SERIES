"""
Episode 04 — File 4: Full RAG Chatbot
=======================================
Run: python 04_rag_chatbot.py

What this file does:
  - Loads a document (text or PDF)
  - Chunks it with overlap
  - Embeds all chunks into an in-memory vector store
  - Runs an interactive Q&A loop:
      1. User types a question
      2. Question is embedded and top-3 chunks are retrieved
      3. Chunks are injected into a grounded prompt template
      4. Gemini generates an answer based ONLY on the provided context
      5. Answer is printed with source chunk references
  - Also demonstrates what happens WITHOUT RAG (hallucination)

This is the MAIN file of Episode 04 — the complete RAG pipeline.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from gemini_client import generate, embed

# ---------------------------------------------------------------------------
# RAG prompt template — instructs the model to only answer from context
# ---------------------------------------------------------------------------

RAG_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions based ONLY on the provided context.

CONTEXT (from the document):
{context}

USER QUESTION:
{question}

INSTRUCTIONS:
- Answer the question using ONLY the information in the context above.
- If the context does not contain enough information to answer, say "I don't have enough information in the document to answer this question."
- Mention which chunk(s) your answer is based on (e.g. "Based on Chunk 3...").
- Be concise and accurate."""


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------

def load_document(filepath: str) -> str:
    """Load a .txt or .pdf file and return its text content."""
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".txt":
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    elif ext == ".pdf":
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            raise ImportError("PyPDF2 is required for PDF support: pip install PyPDF2")

        reader = PdfReader(filepath)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)

    else:
        raise ValueError(f"Unsupported file type: {ext}. Use .txt or .pdf")


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[dict]:
    """Split text into overlapping chunks using a sliding window."""
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


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Embed each chunk and return dicts with the embedding vector attached."""
    embedded = []
    for chunk in chunks:
        vector = embed(chunk["text"])
        embedded.append({
            "index": chunk["index"],
            "text": chunk["text"],
            "embedding": vector,
        })
    return embedded


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

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
    """Find the top-k most relevant chunks for a query."""
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


# ---------------------------------------------------------------------------
# RAG prompt builder
# ---------------------------------------------------------------------------

def build_rag_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    """Build the full RAG prompt by injecting retrieved chunks as context."""
    context_parts = []
    for chunk in retrieved_chunks:
        context_parts.append(f"[Chunk {chunk['index']}]: {chunk['text']}")

    context_str = "\n\n".join(context_parts)
    return RAG_PROMPT_TEMPLATE.format(context=context_str, question=question)


# ---------------------------------------------------------------------------
# Main chatbot loop
# ---------------------------------------------------------------------------

def main():
    filepath = os.path.join(os.path.dirname(__file__), "sample_data", "sample.txt")

    print("=" * 60)
    print("RAG DOCUMENT Q&A CHATBOT")
    print("=" * 60)

    # Step 1: Load
    print(f"\nLoading document: {filepath}")
    text = load_document(filepath)
    print(f"Document length: {len(text):,} characters")

    # Step 2: Chunk
    chunk_size = 500
    overlap = 100
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    print(f"Splitting into chunks... {len(chunks)} chunks created "
          f"(size={chunk_size}, overlap={overlap})")

    # Step 3: Embed
    print("Embedding chunks... ", end="", flush=True)
    embedded_chunks = embed_chunks(chunks)
    print("Done.")

    print(f"\n{'=' * 60}")
    print("RAG Chatbot ready! Ask questions about your document.")
    print("Type 'quit' to exit.")
    print("=" * 60)

    

    # Step 4: Interactive Q&A loop
    while True:
        try:
            question = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        # Retrieve relevant chunks
        relevant = retrieve(question, embedded_chunks, top_k=3)

        # Show what was retrieved
        print(f"\n  [Retrieved chunks: {', '.join(f'#{r['index']} ({r['score']:.3f})' for r in relevant)}]")

        # Build RAG prompt and generate
        rag_prompt = build_rag_prompt(question, relevant)
        answer = generate(rag_prompt, temperature=0.3)

        print(f"\nAssistant: {answer.strip()}")


if __name__ == "__main__":
    main()
