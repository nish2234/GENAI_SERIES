"""
Reusable Gemini Client — Episode 11 (Multi-Agent)
====================================================
Multi-provider fallback: Gemini → Groq → Cerebras
Includes: generate(), generate_structured(), count_tokens(), get_client()
"""

import json
import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

GEMINI_MODEL = "gemini-3.1-flash-lite"
GROQ_MODEL = "groq/openai/gpt-oss-120b"
CEREBRAS_MODEL = "cerebras/gpt-oss-120b"
MAX_RETRIES = 2
INITIAL_WAIT = 2

_client: genai.Client | None = None


def _is_retryable(error: Exception) -> bool:
    error_str = str(error).lower()
    return any(k in error_str for k in [
        "503", "429", "overloaded", "resource exhausted",
        "deadline exceeded", "timeout", "unavailable", "internal", "500", "connection",
    ])


def get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found. Check your .env file.")
        _client = genai.Client(api_key=api_key)
    return _client


def _call_gemini(prompt: str, temperature: float, max_tokens: int, system: str | None) -> str:
    client = get_client()
    config = types.GenerateContentConfig(
        temperature=temperature, max_output_tokens=max_tokens, system_instruction=system,
    )
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt, config=config)
            return response.text
        except Exception as e:
            if not _is_retryable(e) or attempt == MAX_RETRIES - 1:
                raise
            time.sleep(INITIAL_WAIT * (2 ** attempt))


def _call_litellm(model: str, prompt: str, temperature: float, max_tokens: int, system: str | None) -> str:
    import litellm
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    for attempt in range(MAX_RETRIES):
        try:
            response = litellm.completion(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
            return response.choices[0].message.content
        except Exception as e:
            if not _is_retryable(e) or attempt == MAX_RETRIES - 1:
                raise
            time.sleep(INITIAL_WAIT * (2 ** attempt))


def generate(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    system: str | None = None,
) -> str:
    """Send a prompt, get a response. Falls through to Groq/Cerebras on failure."""
    try:
        return _call_gemini(prompt, temperature, max_tokens, system)
    except Exception:
        pass

    if os.environ.get("GROQ_API_KEY"):
        try:
            return _call_litellm(GROQ_MODEL, prompt, temperature, max_tokens, system)
        except Exception:
            pass

    if os.environ.get("CEREBRAS_API_KEY"):
        try:
            return _call_litellm(CEREBRAS_MODEL, prompt, temperature, max_tokens, system)
        except Exception:
            pass

    raise RuntimeError("All providers failed. Check your API keys.")


def generate_structured(
    prompt: str,
    response_schema: dict | type | None = None,
    model: str = GEMINI_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    system: str | None = None,
) -> dict:
    """
    Send a prompt and get structured JSON output.
    Uses Gemini's native response_schema. Falls back to prompting for JSON on backup models.
    """
    client = get_client()

    config = types.GenerateContentConfig(
        temperature=temperature, max_output_tokens=max_tokens,
        system_instruction=system, response_mime_type="application/json",
        response_schema=response_schema,
    )
    try:
        for attempt in range(MAX_RETRIES):
            try:
                response = client.models.generate_content(model=model, contents=prompt, config=config)
                return json.loads(response.text)
            except json.JSONDecodeError:
                raise
            except Exception as e:
                if not _is_retryable(e) or attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(INITIAL_WAIT * (2 ** attempt))
    except Exception:
        pass

    json_prompt = f"{prompt}\n\nRespond ONLY with valid JSON. No markdown, no explanation."
    if system:
        json_prompt = f"System: {system}\n\n{json_prompt}"

    for fallback_model, key in [(GROQ_MODEL, "GROQ_API_KEY"), (CEREBRAS_MODEL, "CEREBRAS_API_KEY")]:
        if os.environ.get(key):
            try:
                text = _call_litellm(fallback_model, json_prompt, temperature, max_tokens, None)
                cleaned = text.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
                return json.loads(cleaned)
            except Exception:
                continue

    raise RuntimeError("All providers failed for structured output.")


def count_tokens(prompt: str, model: str = GEMINI_MODEL) -> int:
    """Count tokens without generating a response."""
    client = get_client()
    for attempt in range(MAX_RETRIES):
        try:
            result = client.models.count_tokens(model=model, contents=prompt)
            return result.total_tokens
        except Exception as e:
            if not _is_retryable(e) or attempt == MAX_RETRIES - 1:
                raise
            time.sleep(INITIAL_WAIT * (2 ** attempt))


if __name__ == "__main__":
    print("Testing client (multi-provider fallback)...\n")
    result = generate("Say hello.")
    print(f"Generate: {result[:80]}")
    print("All tests passed.")
