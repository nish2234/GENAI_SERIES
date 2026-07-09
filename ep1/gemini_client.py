"""
Reusable Gemini Client — Used in Every Episode of This Series
==============================================================

Multi-provider fallback: Gemini (primary) → Groq → Cerebras
If Gemini is down or rate-limited, automatically falls through to backup providers.

Setup your .env file:
  GEMINI_API_KEY=...           # Required — primary provider
  GROQ_API_KEY=...             # Optional — fallback 1 (get free at console.groq.com)
  CEREBRAS_API_KEY=...         # Optional — fallback 2 (get free at cloud.cerebras.ai)

Usage (in any future episode file):
  from gemini_client import generate, count_tokens

  answer = generate("What is a transformer?")
  tokens = count_tokens("How long is this sentence?")
"""

import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# --- Config ---
GEMINI_MODEL = "gemini-3.1-flash-lite"
GROQ_MODEL = "groq/openai/gpt-oss-120b"
CEREBRAS_MODEL = "cerebras/gpt-oss-120b"
MAX_RETRIES = 2
INITIAL_WAIT = 2

# Singleton client
_client: genai.Client | None = None


def get_client() -> genai.Client:
    """Returns the shared Gemini client, creating it on first call."""
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found.\n"
                "Make sure your .env file exists and contains: GEMINI_API_KEY=your_key_here"
            )
        _client = genai.Client(api_key=api_key)
    return _client


def _is_retryable(error: Exception) -> bool:
    """Check if an error is worth retrying."""
    error_str = str(error).lower()
    return any(k in error_str for k in [
        "503", "429", "overloaded", "resource exhausted",
        "deadline exceeded", "timeout", "unavailable",
        "internal", "500", "connection",
    ])


def _call_gemini(prompt: str, temperature: float, max_tokens: int, system: str | None) -> str:
    """Call Gemini with retry logic."""
    client = get_client()
    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
        system_instruction=system,
    )

    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL, contents=prompt, config=config,
            )
            return response.text
        except Exception as e:
            if not _is_retryable(e) or attempt == MAX_RETRIES - 1:
                raise
            wait = INITIAL_WAIT * (2 ** attempt)
            print(f"  [Gemini retry {attempt + 1}] Waiting {wait}s...")
            time.sleep(wait)


def _call_litellm(model: str, prompt: str, temperature: float, max_tokens: int, system: str | None) -> str:
    """Call any model via litellm (OpenAI-compatible)."""
    import litellm

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(MAX_RETRIES):
        try:
            response = litellm.completion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            if not _is_retryable(e) or attempt == MAX_RETRIES - 1:
                raise
            wait = INITIAL_WAIT * (2 ** attempt)
            provider = model.split("/")[0]
            print(f"  [{provider} retry {attempt + 1}] Waiting {wait}s...")
            time.sleep(wait)


def generate(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    system: str | None = None,
) -> str:
    """
    Send a prompt and get a response. Falls through to backup providers on failure.

    Fallback chain: Gemini → Groq → Cerebras
    Backup providers only activate if their API key exists in .env

    Args:
        prompt:      The user message / question.
        model:       Primary model (default: gemini-3.1-flash-lite).
        temperature: 0.0 = deterministic, 1.5 = creative. Default: 0.7
        max_tokens:  Maximum tokens in the response. Default: 1024
        system:      Optional system instruction.

    Returns:
        The model's response as a plain string.
    """
    # Attempt 1: Gemini (primary)
    try:
        return _call_gemini(prompt, temperature, max_tokens, system)
    except Exception as e:
        print(f"  [Gemini failed: {str(e)[:80]}]")

    # Attempt 2: Groq (if key exists)
    if os.environ.get("GROQ_API_KEY"):
        try:
            print("  [Falling back to Groq...]")
            return _call_litellm(GROQ_MODEL, prompt, temperature, max_tokens, system)
        except Exception as e:
            print(f"  [Groq failed: {str(e)[:80]}]")

    # Attempt 3: Cerebras (if key exists)
    if os.environ.get("CEREBRAS_API_KEY"):
        try:
            print("  [Falling back to Cerebras...]")
            return _call_litellm(CEREBRAS_MODEL, prompt, temperature, max_tokens, system)
        except Exception as e:
            print(f"  [Cerebras failed: {str(e)[:80]}]")

    raise RuntimeError(
        "All providers failed. Check your API keys and network connection.\n"
        "Required: GEMINI_API_KEY\n"
        "Optional fallbacks: GROQ_API_KEY, CEREBRAS_API_KEY"
    )


def generate_with_metadata(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    system: str | None = None,
) -> dict:
    """
    Same as generate() but also returns token usage and finish reason.
    Note: metadata is only available when Gemini responds (not on fallback).
    """
    client = get_client()
    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
        system_instruction=system,
    )

    try:
        response = client.models.generate_content(
            model=model, contents=prompt, config=config,
        )
        return {
            "text": response.text,
            "input_tokens": response.usage_metadata.prompt_token_count,
            "output_tokens": response.usage_metadata.candidates_token_count,
            "total_tokens": response.usage_metadata.total_token_count,
            "finish_reason": str(response.candidates[0].finish_reason),
            "provider": "gemini",
        }
    except Exception:
        text = generate(prompt, model, temperature, max_tokens, system)
        return {
            "text": text,
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "finish_reason": "stop",
            "provider": "fallback",
        }


def count_tokens(prompt: str, model: str = GEMINI_MODEL) -> int:
    """Count how many tokens a prompt uses WITHOUT making a generation request."""
    client = get_client()

    for attempt in range(MAX_RETRIES):
        try:
            result = client.models.count_tokens(model=model, contents=prompt)
            return result.total_tokens
        except Exception as e:
            if not _is_retryable(e) or attempt == MAX_RETRIES - 1:
                raise
            time.sleep(INITIAL_WAIT * (2 ** attempt))


# --- Self-test when run directly ---
if __name__ == "__main__":
    print("Testing Gemini client (with multi-provider fallback)...\n")

    print("Providers configured:")
    print(f"  Gemini:   {'YES' if os.environ.get('GEMINI_API_KEY') else 'NO'}")
    print(f"  Groq:     {'YES' if os.environ.get('GROQ_API_KEY') else 'NO (optional)'}")
    print(f"  Cerebras: {'YES' if os.environ.get('CEREBRAS_API_KEY') else 'NO (optional)'}")

    # Token count test
    print("\nCounting tokens...")
    tokens = count_tokens("Hello, this is a test of the Gemini client.")
    print(f"Token count: {tokens}")

    # Generation test
    print("\nCalling generate()...")
    result = generate("Say hello in 5 different languages, one per line.")
    print(result)

    # Metadata test
    print("\nCalling generate_with_metadata()...")
    meta = generate_with_metadata("What is 2 + 2? Answer in one word.")
    print(f"Answer   : {meta['text'].strip()}")
    print(f"Provider : {meta['provider']}")
    if meta['total_tokens']:
        print(f"Tokens   : {meta['total_tokens']}")

    print("\nAll tests passed. Client is ready.")
