"""
Episode 05 — File 3: Metadata Filtering in ChromaDB
=====================================================
Run: python 03_metadata_filtering.py

What this file does:
  - Adds documents with metadata (source, category, date)
  - Demonstrates `where` filters — filter by metadata before similarity search
  - Demonstrates `where_document` filters — filter by document content
  - Shows combining filters with $and / $or operators
  - Shows the difference: unfiltered vs filtered search results

This is one of the killer features of a vector DB over plain in-memory search.
"""

import chromadb

client = chromadb.Client()
collection = client.create_collection(name="filtered_search")

# --- Load sample documents with metadata ---
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
    "Annual performance reviews happen in June and December.",
    "The Free plan supports up to 5 users and 3 projects.",
]

ids = [f"doc_{i}" for i in range(len(documents))]

metadatas = [
    {"source": "company_policies", "category": "hr",       "year": 2025},
    {"source": "product_docs",     "category": "pricing",   "year": 2025},
    {"source": "faq",              "category": "it",        "year": 2025},
    {"source": "company_policies", "category": "hr",        "year": 2025},
    {"source": "product_docs",     "category": "features",  "year": 2025},
    {"source": "company_policies", "category": "finance",   "year": 2024},
    {"source": "faq",              "category": "office",    "year": 2024},
    {"source": "product_docs",     "category": "pricing",   "year": 2025},
    {"source": "company_policies", "category": "hr",        "year": 2024},
    {"source": "company_policies", "category": "finance",   "year": 2024},
    {"source": "company_policies", "category": "hr",        "year": 2025},
    {"source": "product_docs",     "category": "pricing",   "year": 2025},
]

collection.add(documents=documents, ids=ids, metadatas=metadatas)
print(f"Loaded {len(documents)} documents with metadata\n")


def print_results(results, label=""):
    """Helper to print query results cleanly."""
    if label:
        print(f"\n{'=' * 60}")
        print(f"  {label}")
        print(f"{'=' * 60}")

    if not results["documents"][0]:
        print("  No results found.")
        return

    for i in range(len(results["documents"][0])):
        doc = results["documents"][0][i]
        meta = results["metadatas"][0][i]
        dist = results["distances"][0][i]
        print(f"\n  [{i + 1}] (dist: {dist:.4f}) [{meta['source']} | {meta['category']} | {meta['year']}]")
        print(f"      {doc[:90]}")


query = "What are the employee benefits?"

# --- 1. No filter (search everything) ---
results = collection.query(query_texts=[query], n_results=4)
print_results(results, "NO FILTER — search all documents")

# --- 2. Filter by source ---
results = collection.query(
    query_texts=[query],
    n_results=4,
    where={"source": "company_policies"},
)
print_results(results, "WHERE source = 'company_policies'")

# --- 3. Filter by category ---
results = collection.query(
    query_texts=[query],
    n_results=4,
    where={"category": "hr"},
)
print_results(results, "WHERE category = 'hr'")

# --- 4. Filter with comparison operator ---
results = collection.query(
    query_texts=[query],
    n_results=4,
    where={"year": {"$gte": 2025}},
)
print_results(results, "WHERE year >= 2025")

# --- 5. Combine filters with $and ---
results = collection.query(
    query_texts=[query],
    n_results=4,
    where={
        "$and": [
            {"source": "company_policies"},
            {"year": {"$gte": 2025}},
        ]
    },
)
print_results(results, "WHERE source = 'company_policies' AND year >= 2025")

# --- 6. Filter by document content with where_document ---
results = collection.query(
    query_texts=[query],
    n_results=4,
    where_document={"$contains": "plan"},
)
print_results(results, "WHERE document CONTAINS 'plan'")

# --- 7. Combine $or filter ---
results = collection.query(
    query_texts=["pricing and costs"],
    n_results=4,
    where={
        "$or": [
            {"category": "pricing"},
            {"category": "finance"},
        ]
    },
)
print_results(results, "WHERE category = 'pricing' OR category = 'finance'")

print("\n\nDone! Metadata filtering demo complete.")
print("Key takeaway: filters run BEFORE similarity search, making results precise and fast.")
