"""
Episode 05 — File 4: Gemini Embeddings with ChromaDB
======================================================
Run: python 04_gemini_embeddings_with_chroma.py

What this file does:
  - Creates a custom embedding function that calls Gemini's text-embedding-004
  - Plugs it into ChromaDB (replaces the default embedding model)
  - Adds documents and queries using Gemini embeddings
  - Shows why you might want this: better quality, same model family as generation

Why use Gemini embeddings instead of ChromaDB's default?
  - Higher quality embeddings (768 dimensions, trained on more data)
  - Consistency — same model family for embedding and generation
  - Better multilingual support
"""
import os
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from gemini_client import get_client ,get_embeddings
import chromadb.utils.embedding_functions as embedding_functions



api_key = os.getenv("GEMINI_API_KEY")

# --- Create ChromaDB client with Gemini embeddings ---
chroma_client = chromadb.Client()
gemini_ef = embedding_functions.GoogleGeminiEmbeddingFunction(
    model_name="gemini-embedding-001",
    task_type="RETRIEVAL_DOCUMENT",
)

collection = chroma_client.create_collection(
    name="gemini_embeddings_demo_1",
    embedding_function=gemini_ef,
)

print("Collection created with Gemini embedding function\n")

# --- Add documents ---
documents = [
    "Python is a programming language known for its simplicity and readability.",
    "JavaScript is the language of the web, running in every browser.",
    "Machine learning is a subset of AI that learns patterns from data.",
    "React is a JavaScript library for building user interfaces.",
    "PyTorch and TensorFlow are popular frameworks for deep learning.",
    "SQL is used to query and manage relational databases.",
    "Docker containers package applications with all their dependencies.",
    "Kubernetes orchestrates container deployments at scale.",
]

ids = [f"doc_{i}" for i in range(len(documents))]

print("Adding documents (Gemini will generate embeddings)...")
print(len(documents) , len(ids))
collection.add(documents=documents, ids=ids)
print(f"Added {len(documents)} documents\n")

# --- Query using Gemini embeddings ---
queries = [
    "What tools are used for artificial intelligence?",
    "How do I manage databases?",
    "What technologies help with deploying software?",
]

for query in queries:
    print("=" * 60)
    print(f"QUERY: '{query}'")
    print("=" * 60)

    results = collection.query(query_texts=[query], n_results=3)

    for i in range(len(results["documents"][0])):
        doc = results["documents"][0][i]
        dist = results["distances"][0][i]
        print(f"  [{i + 1}] (distance: {dist:.4f}) {doc}")
    print()

# --- Compare: peek at the raw embedding ---
print("=" * 60)
print("RAW EMBEDDING PEEK")
print("=" * 60)
test_embedding = gemini_ef(["Hello world"])
vec = test_embedding[0]
print(f"  Dimensions: {len(vec)}")
print(f"  First 5 values: {[round(v, 6) for v in vec[:5]]}")
print(f"  Last 5 values:  {[round(v, 6) for v in vec[-5:]]}")

print("\nDone! Gemini embeddings are now powering ChromaDB searches.")
