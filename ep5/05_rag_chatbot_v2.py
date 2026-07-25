"""
Episode 05 — File 5: RAG Chatbot v2 (ChromaDB + Gemini)
=========================================================
Run: python 05_rag_chatbot_v2.py

What this file does:
  1. Loads multiple documents from sample_data/
  2. Chunks them with overlap for better retrieval
  3. Stores chunks in ChromaDB with metadata (source filename, chunk index)
  4. Interactive Q&A loop:
     - User asks a question
     - ChromaDB retrieves top-k relevant chunks (with optional metadata filter)
     - Injects context into a prompt with source attribution
     - Gets an answer from Gemini
     - Prints the answer + which documents/chunks were used
  5. Persistence: second run skips indexing because data is already in ChromaDB

This is the MAIN file of Episode 5 — the upgraded RAG chatbot.
"""

import os
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from gemini_client import get_client, generate
import chromadb.utils.embedding_functions as embedding_functions


# ─────────────────────────────────────────────
# Document Loading
# ─────────────────────────────────────────────

def load_documents(data_dir: str = "sample_data") -> list[dict]:
    """
    Load all .txt files from the data directory.

    Returns a list of dicts: [{"filename": ..., "content": ...}, ...]
    """
    docs = []
    if not os.path.exists(data_dir):
        print(f"Warning: '{data_dir}' directory not found.")
        return docs

    for filename in sorted(os.listdir(data_dir)):
        if filename.endswith(".txt"):
            filepath = os.path.join(data_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read().strip()
            docs.append({"filename": filename, "content": content})
            print(f"  Loaded: {filename} ({len(content):,} chars)")

    return docs


# ─────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """
    Split text into overlapping chunks.

    Args:
        text:       The full document text.
        chunk_size: Target characters per chunk.
        overlap:    Characters of overlap between consecutive chunks.

    Returns:
        List of text chunks.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks


# ─────────────────────────────────────────────
# Indexing (Load → Chunk → Store in ChromaDB)
# ─────────────────────────────────────────────

def index_documents(collection, data_dir: str = "sample_data") -> int:
    """
    Load documents, chunk them, and store in ChromaDB with metadata.

    Skips indexing if the collection already has documents (persistence!).

    Returns:
        Number of chunks added.
    """
    existing_count = collection.count()
    if existing_count > 0:
        print(f"\n  Collection already has {existing_count} chunks — skipping indexing.")
        print("  (This is persistence in action! Data survived the restart.)")
        return 0

    print("\n  Loading documents...")
    docs = load_documents(data_dir)

    if not docs:
        print("  No documents found to index.")
        return 0

    all_chunks = []
    all_ids = []
    all_metadatas = []
    chunk_counter = 0

    print("\n  Chunking documents...")
    for doc in docs:
        chunks = chunk_text(doc["content"])
        source_name = doc["filename"].replace(".txt", "")

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{source_name}_chunk_{i}")
            all_metadatas.append({
                "source": doc["filename"],
                "source_name": source_name,
                "chunk_index": i,
                "total_chunks": len(chunks),
            })
            chunk_counter += 1

        print(f"    {doc['filename']} → {len(chunks)} chunks")

    print(f"\n  Storing {chunk_counter} chunks in ChromaDB...")
    batch_size = 50
    for i in range(0, len(all_chunks), batch_size):
        collection.add(
            documents=all_chunks[i:i + batch_size],
            ids=all_ids[i:i + batch_size],
            metadatas=all_metadatas[i:i + batch_size],
        )

    print(f"  Done! {chunk_counter} chunks indexed.")
    return chunk_counter


# ─────────────────────────────────────────────
# RAG Query
# ─────────────────────────────────────────────

def rag_query(
    question: str,
    collection,
    top_k: int = 5,
    source_filter: str | None = None,
) -> str:
    """
    Retrieve relevant chunks from ChromaDB and generate an answer with Gemini.

    Args:
        question:      The user's question.
        collection:    ChromaDB collection to search.
        top_k:         Number of chunks to retrieve.
        source_filter: Optional — restrict search to a specific source file.

    Returns:
        The generated answer string.
    """
    where_filter = None
    if source_filter:
        where_filter = {"source": source_filter}

    results = collection.query(
        query_texts=[question],
        n_results=top_k,
        where=where_filter,
    )

    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    if not chunks:
        return "No relevant documents found for your question."

    # Build context with source attribution
    context_parts = []
    sources_used = []
    for i, (chunk, meta, dist) in enumerate(zip(chunks, metadatas, distances)):
        source_label = f"{meta['source']} (chunk {meta['chunk_index'] + 1}/{meta['total_chunks']})"
        context_parts.append(f"[Source: {source_label}]\n{chunk}")
        sources_used.append(source_label)

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are a helpful assistant that answers questions based ONLY on the provided context.
If the context does not contain enough information to answer, say so honestly.
Always mention which source documents you used in your answer.

CONTEXT:
{context}

QUESTION: {question}

ANSWER (cite your sources):"""

    answer = generate(prompt, temperature=0.3, max_tokens=512)

    # Print source attribution
    print("\n  Sources used:")
    for i, (source, dist) in enumerate(zip(sources_used, distances)):
        print(f"    {i + 1}. {source} (distance: {dist:.4f})")

    return answer


# ─────────────────────────────────────────────
# Interactive Chat Loop
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  RAG CHATBOT v2 — ChromaDB + Gemini")
    print("  Episode 05 of The Complete Gen AI Series")
    print("=" * 60)

    # Use PersistentClient so data survives restarts
    db_path = os.path.join(os.path.dirname(__file__) or ".", "chroma_db")
    chroma_client = chromadb.PersistentClient(path=db_path)

    gemini_ef = gemini_ef = embedding_functions.GoogleGeminiEmbeddingFunction(
    model_name="gemini-embedding-001",
    task_type="RETRIEVAL_DOCUMENT",
    )

    collection = chroma_client.get_or_create_collection(
        name="rag_documents",
        embedding_function=gemini_ef,
    )

    # Index documents (skips if already indexed)
    index_documents(collection)

    print(f"\n  Collection has {collection.count()} chunks ready for search.")
    print("\n  Commands:")
    print("    - Type a question to search and get an answer")
    print("    - Type 'filter:<filename>' to restrict to a source (e.g., filter:faq.txt)")
    print("    - Type 'sources' to list available sources")
    print("    - Type 'count' to see how many chunks are stored")
    print("    - Type 'quit' to exit")

    active_filter = None

    while True:
        print("\n" + "-" * 60)
        filter_label = f" [filter: {active_filter}]" if active_filter else ""
        user_input = input(f"  You{filter_label}: ").strip()

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("\n  Goodbye! Your data is persisted — run again and it's still here.")
            break

        if user_input.lower() == "sources":
            all_meta = collection.get(include=["metadatas"])
            sources = sorted(set(m["source"] for m in all_meta["metadatas"]))
            print("\n  Available sources:")
            for s in sources:
                count = sum(1 for m in all_meta["metadatas"] if m["source"] == s)
                print(f"    - {s} ({count} chunks)")
            continue

        if user_input.lower() == "count":
            print(f"\n  Total chunks in ChromaDB: {collection.count()}")
            continue

        if user_input.lower().startswith("filter:"):
            filter_value = user_input.split(":", 1)[1].strip()
            if filter_value.lower() == "none" or filter_value == "":
                active_filter = None
                print("  Filter cleared — searching all sources.")
            else:
                active_filter = filter_value
                print(f"  Filter set to: {active_filter}")
            continue

        # RAG query
        print("\n  Searching...")
        answer = rag_query(
            question=user_input,
            collection=collection,
            source_filter=active_filter,
        )

        print(f"\n  Assistant: {answer}")


if __name__ == "__main__":
    main()
