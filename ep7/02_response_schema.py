"""
02 — Response Schemas: Constrain Model Output to Your Structure
================================================================

With response_schema, YOU define the exact JSON structure the model
must follow. The model's output is constrained via constrained decoding —
it literally cannot produce output that doesn't match your schema.

Key parameters:
  response_mime_type="application/json"
  response_schema=<your JSON schema dict>

Run:
  python 02_response_schema.py
"""

import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL = "gemini-3.1-flash-lite"


def demo_simple_schema():
    """Extract name, age, city from a sentence using a defined schema."""
    print("=" * 60)
    print("SIMPLE SCHEMA — Extract {name, age, city}")
    print("=" * 60)

    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "The person's full name"},
            "age": {"type": "integer", "description": "The person's age"},
            "city": {"type": "string", "description": "The city they live in"},
        },
        "required": ["name", "age", "city"],
    }

    sentence = "My name is Sarah Chen, I'm 28 years old and I live in San Francisco."

    response = client.models.generate_content(
        model=MODEL,
        contents=f"Extract the person's info from this sentence:\n\n{sentence}",
        config=types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=schema,
        ),
    )

    data = json.loads(response.text)
    print(f"\nInput:  {sentence}")
    print(f"Output: {json.dumps(data, indent=2)}")
    print(f"\nName: {data['name']}, Age: {data['age']}, City: {data['city']}")


def demo_array_schema():
    """Extract a list of items matching a schema."""
    print("\n" + "=" * 60)
    print("ARRAY SCHEMA — Extract list of products")
    print("=" * 60)

    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "product": {"type": "string"},
                "price": {"type": "number"},
                "currency": {"type": "string"},
            },
            "required": ["product", "price", "currency"],
        },
    }

    text = """
    Today's deals: MacBook Pro at $2,499, AirPods Max for $549,
    and the iPad Air starting at $599. All prices in USD.
    """

    response = client.models.generate_content(
        model=MODEL,
        contents=f"Extract all products with their prices:\n\n{text}",
        config=types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=schema,
        ),
    )

    products = json.loads(response.text)
    print(f"\nExtracted {len(products)} products:")
    for p in products:
        print(f"  - {p['product']}: {p['currency']} {p['price']}")


def demo_enum_schema():
    """Use enum to constrain values to specific options."""
    print("\n" + "=" * 60)
    print("ENUM SCHEMA — Classify sentiment")
    print("=" * 60)

    schema = {
        "type": "object",
        "properties": {
            "sentiment": {
                "type": "string",
                "enum": ["positive", "negative", "neutral"],
            },
            "confidence": {
                "type": "number",
                "description": "Confidence score between 0 and 1",
            },
            "reasoning": {
                "type": "string",
                "description": "Brief explanation for the classification",
            },
        },
        "required": ["sentiment", "confidence", "reasoning"],
    }

    reviews = [
        "This product is absolutely fantastic! Best purchase I've made all year.",
        "Terrible quality. Broke after two days. Want my money back.",
        "It's okay. Does what it says. Nothing special.",
    ]

    for review in reviews:
        response = client.models.generate_content(
            model=MODEL,
            contents=f"Classify the sentiment of this review:\n\n{review}",
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )

        result = json.loads(response.text)
        print(f"\n  Review: \"{review[:50]}...\"")
        print(f"  → {result['sentiment']} (confidence: {result['confidence']})")
        print(f"    Reason: {result['reasoning']}")


if __name__ == "__main__":
    demo_simple_schema()
    demo_array_schema()
    demo_enum_schema()

    print("\n" + "=" * 60)
    print("KEY TAKEAWAY")
    print("=" * 60)
    print("""
Response schemas give you FULL CONTROL over the output structure.
The model is constrained — it can ONLY output valid JSON matching
your schema. Keys, types, required fields — all enforced.

But writing JSON Schema dicts by hand gets tedious for complex structures.
Next up: Pydantic models — define schemas as Python classes.
""")
