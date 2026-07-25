"""
Episode 05 — File 2: Adding Documents with Metadata
=====================================================
Run: python 02_add_with_metadata.py

What this file does:
  - Creates a ChromaDB collection
  - Adds documents WITH metadata (source, category, date)
  - Shows how metadata is stored alongside the vector
  - Queries and displays metadata in results

Key insight: metadata lets you FILTER before searching — like adding
a WHERE clause to your similarity search.
"""

import chromadb

client = chromadb.Client()

collection = client.create_collection(name="docs_with_metadata")

# --- Add documents with rich metadata ---
# Each document gets an ID, the text, and a metadata dict

documents = [
    "All employees receive 20 days of paid time off per year.",
    "The Pro plan costs $12 per user per month with unlimited projects.",
    "Password reset links expire after 24 hours.",
    "Remote work is allowed up to 3 days per week with manager approval.",
    "TaskFlow integrates with Slack, GitHub, Jira, and 40+ other tools.",
    "Expense reports must be submitted within 30 days of the transaction.",
    "Conference rooms can be booked in 30-minute increments up to 4 hours.",
    "The Enterprise plan includes SSO, audit logs, and 99.9% uptime SLA.",
    "Parental leave is 16 weeks fully paid for primary caregivers.",
    "The company matches 401(k) contributions up to 4% of base salary.",
]

ids = [f"doc_{i}" for i in range(len(documents))]

metadatas = [
    {"source": "company_policies", "category": "hr",      "date": "2025-01-15"},
    {"source": "product_docs",     "category": "pricing",  "date": "2025-03-01"},
    {"source": "faq",              "category": "it",       "date": "2025-02-20"},
    {"source": "company_policies", "category": "hr",       "date": "2025-01-15"},
    {"source": "product_docs",     "category": "features", "date": "2025-03-01"},
    {"source": "company_policies", "category": "finance",  "date": "2025-01-15"},
    {"source": "faq",              "category": "office",   "date": "2025-02-20"},
    {"source": "product_docs",     "category": "pricing",  "date": "2025-03-01"},
    {"source": "company_policies", "category": "hr",       "date": "2025-01-15"},
    {"source": "company_policies", "category": "finance",  "date": "2025-01-15"},
]

collection.add(
    documents=documents,
    ids=ids,
    metadatas=metadatas,
)

print(f"Added {len(documents)} documents with metadata\n")

# --- Query and see metadata in results ---
print("=" * 60)
print("QUERY: 'How much time off do employees get?'")
print("=" * 60)

results = collection.query(
    query_texts=["How much time off do employees get?"],
    n_results=3,
)

for i in range(len(results["documents"][0])):
    doc = results["documents"][0][i]
    meta = results["metadatas"][0][i]
    dist = results["distances"][0][i]
    doc_id = results["ids"][0][i]

    print(f"\n  Result {i + 1} (distance: {dist:.4f}) [{doc_id}]")
    print(f"  Source:   {meta['source']}")
    print(f"  Category: {meta['category']}")
    print(f"  Date:     {meta['date']}")
    print(f"  Text:     {doc[:80]}...")

# --- Get a specific document by ID ---
print("\n" + "=" * 60)
print("GET BY ID: 'doc_1'")
print("=" * 60)

result = collection.get(ids=["doc_1"], include=["documents", "metadatas"])
print(f"  Text:     {result['documents'][0]}")
print(f"  Metadata: {result['metadatas'][0]}")

print(f"\nTotal documents in collection: {collection.count()}")
print("Done! Metadata demo complete.")
