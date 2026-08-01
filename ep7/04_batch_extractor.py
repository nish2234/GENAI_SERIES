"""
05 — Batch Extractor: Process Multiple Texts with Deduplication
================================================================

A practical pipeline that:
  1. Processes multiple texts sequentially
  2. Collects all extracted entities
  3. Deduplicates entities across documents
  4. Outputs a unified summary report

This shows how structured outputs enable building real data pipelines.

Run:
  python 05_batch_extractor.py
"""

import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from enum import Enum
from collections import defaultdict

load_dotenv()

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL = "gemini-3.1-flash-lite"


# ──────────────────────────────────────────────────────────────
# Models (same as 04_entity_extractor.py)
# ──────────────────────────────────────────────────────────────

class Person(BaseModel):
    name: str
    role: str | None = None


class DateEntity(BaseModel):
    value: str
    context: str


class MoneyEntity(BaseModel):
    amount: float
    currency: str
    context: str


class Company(BaseModel):
    name: str
    industry: str | None = None


class LocationType(str, Enum):
    city = "city"
    country = "country"
    state = "state"
    address = "address"
    region = "region"
    other = "other"


class Location(BaseModel):
    name: str
    type: LocationType


class ExtractionResult(BaseModel):
    people: list[Person] = Field(default_factory=list)
    dates: list[DateEntity] = Field(default_factory=list)
    money: list[MoneyEntity] = Field(default_factory=list)
    companies: list[Company] = Field(default_factory=list)
    locations: list[Location] = Field(default_factory=list)
    summary: str = ""


# ──────────────────────────────────────────────────────────────
# Batch Processing Pipeline
# ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise entity extraction system.
Extract ALL entities from the provided text. Be thorough.
If a field is not applicable, use null for optional fields or empty lists."""


def extract_from_text(text: str) -> ExtractionResult:
    """Extract entities from a single text."""
    response = client.models.generate_content(
        model=MODEL,
        contents=f"Extract all entities from this text:\n\n{text}",
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=2048,
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=ExtractionResult,
        ),
    )
    raw = json.loads(response.text)
    return ExtractionResult.model_validate(raw)


def deduplicate_entities(results: list[ExtractionResult]) -> dict:
    """Merge and deduplicate entities across multiple extraction results."""
    seen_people: dict[str, Person] = {}
    seen_companies: dict[str, Company] = {}
    seen_locations: dict[str, Location] = {}
    all_dates: list[DateEntity] = []
    all_money: list[MoneyEntity] = []

    for result in results:
        for person in result.people:
            key = person.name.lower().strip()
            seen_people[key] = person

        for company in result.companies:
            key = company.name.lower().strip()
            seen_companies[key] = company

        for location in result.locations:
            key = location.name.lower().strip()
            seen_locations[key] = location

        all_dates.extend(result.dates)
        all_money.extend(result.money)

    return {
        "people": list(seen_people.values()),
        "companies": list(seen_companies.values()),
        "locations": list(seen_locations.values()),
        "dates": all_dates,
        "money": all_money,
    }





# ──────────────────────────────────────────────────────────────
# Sample Documents
# ──────────────────────────────────────────────────────────────

DOCUMENTS = [
    """
    Microsoft CEO Satya Nadella announced a $3.2 billion investment in cloud
    infrastructure across Southeast Asia on March 15, 2025. The new data centers
    will be built in Jakarta, Indonesia and Bangkok, Thailand. Microsoft's cloud
    chief Scott Guthrie said Azure capacity would triple in the region by 2026.
    """,
    """
    In a separate filing, Microsoft disclosed a $1.5 billion AI research
    partnership with Tokyo University, effective April 1, 2025. VP of Research
    Peter Lee and Professor Hiroshi Tanaka will co-lead the initiative.
    The research lab will be located in Shibuya, Tokyo.
    """,
    """
    Satya Nadella also met with Singapore Prime Minister Lawrence Wong on
    March 16, 2025 to discuss AI governance. Microsoft pledged $500 million
    for AI safety research in Singapore. The meeting took place at the
    Istana presidential palace.
    """,
]


if __name__ == "__main__":
    print("Processing batch of documents...\n")

    results = []
    for i, doc in enumerate(DOCUMENTS, 1):
        print(f"  Processing document {i}/{len(DOCUMENTS)}...", end=" ")
        result = extract_from_text(doc)
        results.append(result)
        entity_count = (
            len(result.people) + len(result.companies) +
            len(result.locations) + len(result.dates) + len(result.money)
        )
        print(f"found {entity_count} entities")

    print("\nDeduplicating across all documents...")
    merged = deduplicate_entities(results)

  

    print("\n\n  Exporting to JSON...")
    export = {
        "people": [p.model_dump() for p in merged["people"]],
        "companies": [c.model_dump() for c in merged["companies"]],
        "locations": [loc.model_dump() for loc in merged["locations"]],
        "money": [m.model_dump() for m in merged["money"]],
        "dates": [d.model_dump() for d in merged["dates"]],
    }
    print(f"\n{json.dumps(export, indent=2)}")
    print("\n  ✓ Batch extraction complete!")
