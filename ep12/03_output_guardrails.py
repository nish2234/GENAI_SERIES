"""
Episode 12 — File 3: Output Guardrails
========================================
Run: python 03_output_guardrails.py

What this file does:
  - Builds output validation layers that scan the response BEFORE returning it:
    1. PII detection — regex for emails, phone numbers, SSNs, credit cards
    2. PII redaction — replaces detected PII with [REDACTED] placeholders
    3. Content safety check — LLM judges whether the response is appropriate
  - Combines them into check_output(response) → (is_safe, reason, cleaned)
  - Demos with responses that contain PII and borderline content

These are the "back door" guards — they run AFTER the LLM generates but BEFORE
the user sees the response.
"""

import re
from gemini_client import generate


# ── Layer 1: PII Detection ──────────────────────────────────────────────────

PII_PATTERNS = {
    "email": re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    ),
    "phone_us": re.compile(
        r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    ),
    "ssn": re.compile(
        r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"
    ),
    "credit_card": re.compile(
        r"\b(?:\d{4}[-\s]?){3}\d{4}\b"
    ),
    "ip_address": re.compile(
        r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
    ),
}


def detect_pii(text: str) -> list[dict]:
    """Scan text for PII patterns. Returns list of {type, match, start, end}."""
    findings = []
    for pii_type, pattern in PII_PATTERNS.items():
        for match in pattern.finditer(text):
            findings.append({
                "type": pii_type,
                "match": match.group(),
                "start": match.start(),
                "end": match.end(),
            })
    return findings


def redact_pii(text: str) -> tuple[str, list[dict]]:
    """Replace all PII with [REDACTED-TYPE] placeholders. Returns (cleaned, findings)."""
    findings = detect_pii(text)
    if not findings:
        return text, []

    redacted = text
    for finding in sorted(findings, key=lambda f: f["start"], reverse=True):
        placeholder = f"[REDACTED-{finding['type'].upper()}]"
        redacted = redacted[:finding["start"]] + placeholder + redacted[finding["end"]:]

    return redacted, findings


# ── Layer 2: Content Safety Check ────────────────────────────────────────────

def check_content_safety(response: str) -> tuple[bool, str]:
    """Uses the LLM as a judge to check if the response content is appropriate."""
    judge_prompt = f"""You are a content safety reviewer. Analyze the following AI response
and determine if it contains any problematic content:

1. Instructions for illegal activities
2. Hate speech, discrimination, or harassment
3. Dangerous misinformation (medical, legal, financial advice presented as fact)
4. Content that could cause real-world harm
5. Content that bypasses safety guidelines

AI Response to review:
\"\"\"{response}\"\"\"

Is this response safe to show to users? Respond with EXACTLY one of:
- "SAFE" if the response is appropriate
- "UNSAFE: <brief reason>" if the response contains problematic content

Your verdict:"""

    result = generate(judge_prompt, temperature=0.0, max_tokens=50)
    verdict = result.strip()

    if verdict.upper().startswith("SAFE"):
        return True, "Content is safe"
    return False, verdict


# ── Combined Output Check ────────────────────────────────────────────────────

def check_output(response: str) -> tuple[bool, str, str]:
    """
    Run all output guardrails.
    Returns (is_safe, reason, cleaned_response).

    The cleaned_response has PII redacted even if the content is otherwise safe.
    If content is unsafe, cleaned_response is a refusal message.
    """
    # Layer 1: redact PII (always runs — PII should never leak)
    cleaned, pii_findings = redact_pii(response)

    pii_warning = ""
    if pii_findings:
        types_found = set(f["type"] for f in pii_findings)
        pii_warning = f"PII redacted: {', '.join(types_found)}"

    # Layer 2: content safety check
    is_safe, safety_reason = check_content_safety(response)

    if not is_safe:
        refusal = (
            "I'm sorry, but I can't provide that response as it may contain "
            "inappropriate content. Please ask a different question."
        )
        reason = f"Content blocked: {safety_reason}"
        if pii_warning:
            reason += f" | {pii_warning}"
        return False, reason, refusal

    if pii_findings:
        return True, f"Content safe but {pii_warning}", cleaned

    return True, "All output checks passed", cleaned


# ── Demo ─────────────────────────────────────────────────────────────────────

def demo_output_check(label: str, response: str):
    print(f"\n{'─' * 60}")
    print(f"TEST: {label}")
    print(f"{'─' * 60}")
    print(f"\nOriginal response:\n  \"{response[:200]}{'...' if len(response) > 200 else ''}\"")

    is_safe, reason, cleaned = check_output(response)
    status = "✅ PASSED" if is_safe else "❌ BLOCKED"
    print(f"\nVerdict:  {status}")
    print(f"Reason:   {reason}")
    print(f"Output:   \"{cleaned[:200]}{'...' if len(cleaned) > 200 else ''}\"")


if __name__ == "__main__":
    print("=" * 60)
    print("DEMO: Output Guardrails")
    print("=" * 60)

    # --- Clean response (should pass) ---
    demo_output_check(
        "Clean Response",
        "To sort a list in Python, use the sorted() function or the .sort() method. "
        "For example: sorted([3, 1, 2]) returns [1, 2, 3].",
    )

    # --- Response with email PII ---
    demo_output_check(
        "Response with Email",
        "Sure! You can reach the support team at support@company.com or "
        "john.doe@example.com for more help.",
    )

    # --- Response with phone numbers ---
    demo_output_check(
        "Response with Phone Number",
        "For customer support, call (555) 123-4567 or +1-800-555-0199. "
        "They're available 24/7.",
    )

    # --- Response with SSN ---
    demo_output_check(
        "Response with SSN",
        "Here's a sample record: Name: Jane Smith, SSN: 123-45-6789, "
        "Email: jane@test.com, Phone: 555-987-6543.",
    )

    # --- Response with credit card ---
    demo_output_check(
        "Response with Credit Card",
        "Your payment was processed with card number 4532-1234-5678-9012. "
        "Transaction ID: TXN-2024-001.",
    )

    # --- Multiple PII types ---
    demo_output_check(
        "Multiple PII Types",
        "User profile: John Doe, email john@example.com, phone (555) 111-2222, "
        "SSN 987-65-4321, card 4111-1111-1111-1111, IP 192.168.1.100.",
    )


