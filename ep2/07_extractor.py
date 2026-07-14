"""Pattern 7: Extract structured data from unstructured text, output as JSON.

Demonstrates how to turn messy real-world text (emails, contracts, receipts)
into clean structured JSON by providing an exact schema in the prompt.
Key technique: specify the schema, constrain to JSON-only output, temp=0.
"""

import json
from gemini_client import generate

email = """
Hi Sarah,

Following up on our call yesterday — confirming that Acme Corp will proceed 
with the Enterprise plan at $2,400/month starting March 15, 2025. The contract 
is for 24 months. Please send the invoice to billing@acmecorp.com.

Our main contact going forward will be James Rodriguez (j.rodriguez@acmecorp.com, 
+1-555-0142). He'll handle onboarding for the 50-seat license.

Let me know if you need anything else.

Best,
Michael Chen
VP of Operations, Acme Corp
"""

prompt = f"""Extract ALL structured information from this email and return it as valid JSON.

Use this exact schema:
{{
  "sender": {{"name": "", "title": "", "company": ""}},
  "plan": {{"name": "", "price_monthly": 0, "currency": "", "start_date": "", "duration_months": 0}},
  "contact": {{"name": "", "email": "", "phone": ""}},
  "details": {{"seats": 0, "billing_email": ""}}
}}

Email:
\"\"\"
{email.strip()}
\"\"\"

Return ONLY valid JSON. No markdown code fences, no explanation."""


if __name__ == "__main__":
    print("=== Structured Data Extraction ===\n")
    print("Input: (business email with scattered details)\n")

    result = generate(prompt, temperature=0.0)

    # Clean potential markdown fencing
    cleaned = result.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(cleaned.split("\n")[1:-1])

    try:
        data = json.loads(cleaned)
        print("Extracted JSON:")
        print(json.dumps(data, indent=2))
    except json.JSONDecodeError:
        print("Raw output (not valid JSON):")
        print(result)
