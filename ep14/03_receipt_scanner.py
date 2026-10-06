"""
Episode 14 — File 3: Receipt Scanner
=====================================
Run: python 03_receipt_scanner.py

What this file does:
  - Takes a receipt image (or any document with structured info)
  - Asks Gemini to extract: store name, date, items, prices, total
  - Outputs clean structured text AND JSON format
  - Demonstrates practical OCR + data extraction with vision AI

This is a killer real-world use case for multimodal AI.
"""


from gemini_client import generate_with_image

# Demo with a public receipt-like image
# (Using a sample receipt image from Wikipedia)
DEFAULT_RECEIPT = "https://upload.wikimedia.org/wikipedia/commons/0/0b/ReceiptSwiss.jpg"

print("=" * 60)
print("RECEIPT SCANNER — Extract Data from Receipt Images")
print("=" * 60)

user_image = input("\nReceipt image path or URL (Enter for demo): ").strip()
image_path = user_image if user_image else DEFAULT_RECEIPT

print(f"\nScanning: {image_path}")
print("Extracting information...\n")

# --- Step 1: Extract as structured text ---
structured_prompt = """Analyze this receipt image and extract ALL information you can find.

Format your response EXACTLY like this:

STORE: [store name]
DATE: [date if visible]
ADDRESS: [address if visible]

ITEMS:
- [item name] ... [price]
- [item name] ... [price]
(list every item you can read)

SUBTOTAL: [amount]
TAX: [amount if shown]
TOTAL: [total amount]
PAYMENT METHOD: [if visible]

If you cannot read a field clearly, write [unclear].
If a field is not present, write [not shown]."""

print("=" * 40)
print("STRUCTURED EXTRACTION")
print("=" * 40)

result = generate_with_image(
    prompt=structured_prompt,
    image_path=image_path,
    temperature=0.2,
    max_tokens=2048,
)
print(result)


summary = generate_with_image(
    prompt="In one sentence: what store is this receipt from, and what was the total amount?",
    image_path=image_path,
    temperature=0.2,
)
print(summary)

print("\n\nDone! Try with your own receipt photos for best results.")
