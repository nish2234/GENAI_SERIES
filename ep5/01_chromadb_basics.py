"""
Episode 05 — File 1: ChromaDB Hello World
==========================================
Run: python 01_chromadb_basics.py

What this file does:
  - Creates a ChromaDB client (in-memory for quick demo)
  - Creates a collection (like a table for vectors)
  - Adds documents to the collection
  - Queries by similarity — finds the most relevant documents
  - Shows the full result structure (documents, distances, IDs)

No Gemini needed here — ChromaDB has a built-in embedding model for basics.
"""

import chromadb

# --- Step 1: Create a ChromaDB client ---
# EphemeralClient = in-memory, no persistence (for this quick demo)
client = chromadb.Client()

print("ChromaDB client created (in-memory mode)\n")

# --- Step 2: Create a collection ---
# A collection is like a table — it holds documents + their embeddings
collection = client.create_collection(name="my_first_collection")
print(f"Collection created: '{collection.name}'")

# --- Step 3: Add documents ---
# ChromaDB auto-generates embeddings using its built-in model
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

collection.add(
    documents=documents,
    ids=ids,
)
print(f"\nAdded {len(documents)} documents to the collection")

# --- Step 4: Query the collection ---
print("\n" + "=" * 60)
print("QUERY: 'What is used for AI and machine learning?'")
print("=" * 60)

results = collection.query(
    query_texts=["What is used for AI and machine learning?"],
    n_results=3,
)

for i, (doc, distance, doc_id) in enumerate(
    zip(results["documents"][0], results["distances"][0], results["ids"][0])
):
    print(f"\n  Result {i + 1} (distance: {distance:.4f}) [{doc_id}]")
    print(f"  → {doc}")

# --- Step 5: Try another query ---
print("\n" + "=" * 60)
print("QUERY: 'How do I deploy applications?'")
print("=" * 60)

results = collection.query(
    query_texts=["How do I deploy applications?"],
    n_results=3,
)

for i, (doc, distance, doc_id) in enumerate(
    zip(results["documents"][0], results["distances"][0], results["ids"][0])
):
    print(f"\n  Result {i + 1} (distance: {distance:.4f}) [{doc_id}]")
    print(f"  → {doc}")

# --- Step 6: Check collection info ---
print("\n" + "=" * 60)
print("COLLECTION INFO")
print("=" * 60)
print(f"  Name:           {collection.name}")
print(f"  Document count: {collection.count()}")
print(f"\nDone! ChromaDB basics complete.")
