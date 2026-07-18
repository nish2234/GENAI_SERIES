"""
Episode 03 — File 3: Semantic Search Engine (MAIN FILE)
========================================================
Run: python 03_search_engine.py

What this file does:
  - Stores a collection of 20 documents about diverse topics
  - Embeds all documents at startup using Gemini's embedding model
  - Accepts user queries in an interactive loop
  - Embeds each query and computes cosine similarity against all documents
  - Returns the top 5 most relevant results ranked by similarity score
  - No keyword matching — pure semantic understanding
"""

import os

import numpy as np
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

EMBEDDING_MODEL = "gemini-embedding-2"

# ---------------------------------------------------------------------------
# Document collection — 20 short paragraphs across diverse topics
# ---------------------------------------------------------------------------

DOCUMENTS = [
    "Artificial intelligence is a branch of computer science that aims to create "
    "machines capable of intelligent behaviour. Modern AI systems can recognise "
    "images, understand speech, and generate human-like text.",

    "Photosynthesis is the process by which green plants convert sunlight into "
    "chemical energy. Chlorophyll in the leaves absorbs light, which drives the "
    "conversion of carbon dioxide and water into glucose and oxygen.",

    "Basketball was invented in 1891 by James Naismith. It is played by two teams "
    "of five players each, with the objective of shooting a ball through the "
    "opponent's hoop to score points.",

    "Black holes are regions of spacetime where gravity is so strong that nothing, "
    "not even light, can escape. They form when massive stars collapse at the end "
    "of their life cycle.",

    "The guitar is a stringed musical instrument usually played by strumming or "
    "plucking the strings. It comes in acoustic and electric varieties and is "
    "central to genres like rock, blues, and classical music.",

    "Democracy is a system of government where citizens exercise power by voting. "
    "It originated in ancient Athens and remains the most widely adopted form of "
    "governance in the modern world.",

    "Cooking involves applying heat to food to change its texture, flavour, and "
    "nutritional properties. Techniques include grilling, roasting, sautéing, "
    "steaming, and baking.",

    "Machine learning is a subset of AI where algorithms learn patterns from data "
    "without being explicitly programmed. It powers recommendation systems, fraud "
    "detection, and autonomous vehicles.",

    "The Great Wall of China stretches over 13,000 miles and was built over many "
    "centuries to protect Chinese states from invasions. It is one of the most "
    "impressive architectural feats in human history.",

    "Sleep is essential for physical and mental health. During deep sleep, the body "
    "repairs tissues, consolidates memories, and releases hormones critical for "
    "growth and immune function.",

    "Quantum computing leverages quantum mechanical phenomena like superposition "
    "and entanglement to perform computations that classical computers cannot "
    "efficiently solve.",

    "Italian cuisine is known for its regional diversity and emphasis on fresh, "
    "high-quality ingredients. Staples include pasta, olive oil, tomatoes, and "
    "a wide variety of cheeses and cured meats.",

    "The human brain contains approximately 86 billion neurons connected by "
    "trillions of synapses. It controls everything from breathing and heartbeat "
    "to complex thought and emotion.",

    "Climate change refers to long-term shifts in global temperatures and weather "
    "patterns. Human activities, particularly burning fossil fuels, have been the "
    "main driver since the Industrial Revolution.",

    "Soccer, known as football outside North America, is the world's most popular "
    "sport with over 4 billion fans. The FIFA World Cup is the most watched "
    "sporting event globally.",

    "Yoga is an ancient practice combining physical postures, breathing exercises, "
    "and meditation. It improves flexibility, reduces stress, and promotes overall "
    "well-being.",

    "The solar system consists of the Sun, eight planets, dwarf planets, moons, "
    "asteroids, and comets. Earth is the third planet from the Sun and the only "
    "known world to harbour life.",

    "Blockchain is a decentralised digital ledger that records transactions across "
    "many computers. It is the technology behind cryptocurrencies like Bitcoin "
    "and has applications in supply chain and healthcare.",

    "Reading fiction improves empathy, vocabulary, and critical thinking. Studies "
    "show that regular readers perform better on tests of social cognition and "
    "creative problem-solving.",

    "Nutrition science studies how food affects health. A balanced diet rich in "
    "fruits, vegetables, whole grains, and lean protein supports energy, immunity, "
    "and long-term disease prevention.",
]


# ---------------------------------------------------------------------------
# Embedding and search helpers
# ---------------------------------------------------------------------------

def get_embedding(text: str) -> np.ndarray:
    """Embed a text and return as a numpy vector."""
    result = client.models.embed_content(model=EMBEDDING_MODEL, contents=text)
    return np.array(result.embeddings[0].values)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def search(query: str, top_k: int = 5) -> list[tuple[float, int]]:
    """
    Search the document collection for the most relevant results.

    Returns a list of (similarity_score, document_index) tuples,
    sorted by similarity in descending order.
    """
    query_embedding = get_embedding(query)

    similarities = []
    for i, doc_emb in enumerate(doc_embeddings):
        sim = cosine_similarity(query_embedding, doc_emb)
        similarities.append((sim, i))

    similarities.sort(reverse=True)
    return similarities[:top_k]


# ---------------------------------------------------------------------------
# Embed all documents at startup
# ---------------------------------------------------------------------------

print("=" * 60)
print("  SEMANTIC SEARCH ENGINE")
print("  Powered by Gemini Embeddings (text-embedding-004)")
print("=" * 60)
print(f"\nLoading {len(DOCUMENTS)} documents...\n")

doc_embeddings: list[np.ndarray] = []
for i, doc in enumerate(DOCUMENTS):
    embedding = get_embedding(doc)
    doc_embeddings.append(embedding)
    print(f"  ✓ Embedded document {i + 1:>2}/{len(DOCUMENTS)}")

print(f"\nAll {len(DOCUMENTS)} documents embedded and ready to search!")
print("Type a query and press Enter. Leave blank to quit.\n")

# ---------------------------------------------------------------------------
# Interactive search loop
# ---------------------------------------------------------------------------

while True:
    query = input("Search: ").strip()
    if not query:
        print("Goodbye!")
        break

    results = search(query, top_k=5)

    print(f"\n  Top {len(results)} results for: \"{query}\"\n")
    for rank, (score, idx) in enumerate(results, 1):
        preview = DOCUMENTS[idx][:120].replace("\n", " ")
        print(f"  {rank}. [{score:.4f}] {preview}...")
    print()
