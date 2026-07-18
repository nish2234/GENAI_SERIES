"""
Reusable Gemini Client — Episode 03 (Semantic Search)
======================================================
Multi-provider fallback: Gemini → Groq → Cerebras
Includes: generate(), get_embedding(), count_tokens()
"""

import os
import time
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

GEMINI_MODEL = "gemini-3.1-flash-lite"
EMBEDDING_MODEL = "models/text-embedding-004"
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


def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> np.ndarray:
    """Get an embedding vector for a piece of text (Gemini only — no fallback needed)."""
    client = get_client()
    for attempt in range(MAX_RETRIES):
        try:
            result = client.models.embed_content(model=model, contents=text)
            return np.array(result.embeddings[0].values)
        except Exception as e:
            if not _is_retryable(e) or attempt == MAX_RETRIES - 1:
                raise
            time.sleep(INITIAL_WAIT * (2 ** attempt))


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
    print(f"  Gemini:   {'YES' if os.environ.get('GEMINI_API_KEY') else 'NO'}")
    print(f"  Groq:     {'YES' if os.environ.get('GROQ_API_KEY') else 'NO (optional)'}")
    print(f"  Cerebras: {'YES' if os.environ.get('CEREBRAS_API_KEY') else 'NO (optional)'}")

    result = generate("Say hello in 3 languages.")
    print(f"\nGenerate: {result[:100]}")

    vec = get_embedding("Semantic search is powerful.")
    print(f"Embedding shape: {vec.shape}")

    print("\nAll tests passed.")
