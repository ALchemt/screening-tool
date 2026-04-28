"""Thin OpenRouter client wrapper with cost tracking.

Centralised so agent.py and judge.py share routing/pricing logic.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
    # Drop-in replacement that auto-traces every call to Langfuse.
    from langfuse.openai import OpenAI  # type: ignore
else:
    from openai import OpenAI

# Per-1M-token prices (USD) as listed on OpenRouter, 2026-04.
# Sources: openrouter.ai/models/<model>
PRICING: dict[str, tuple[float, float]] = {
    "openai/gpt-4o-mini": (0.15, 0.60),
    "openai/gpt-4o": (2.50, 10.00),
    "anthropic/claude-sonnet-4.5": (3.00, 15.00),
    "anthropic/claude-haiku-4.5": (1.00, 5.00),
    "google/gemini-2.5-flash": (0.075, 0.30),
    "openai/text-embedding-3-small": (0.02, 0.0),
}


@dataclass
class LLMResult:
    content: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    model: str


def _client() -> OpenAI:
    return OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )


def cost_of(model: str, prompt_tok: int, completion_tok: int) -> float:
    p_in, p_out = PRICING.get(model, (0.0, 0.0))
    return (prompt_tok * p_in + completion_tok * p_out) / 1_000_000


def chat(messages: list[dict[str, str]], *,
         model: str | None = None,
         temperature: float = 0.7,
         max_tokens: int = 1024,
         json_mode: bool = False) -> LLMResult:
    """Single chat completion via OpenRouter."""
    model = model or os.environ.get("AGENT_MODEL", "openai/gpt-4o-mini")
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = _client().chat.completions.create(**kwargs)
    u = resp.usage
    return LLMResult(
        content=resp.choices[0].message.content or "",
        prompt_tokens=u.prompt_tokens,
        completion_tokens=u.completion_tokens,
        cost_usd=cost_of(model, u.prompt_tokens, u.completion_tokens),
        model=model,
    )


def chat_json(messages: list[dict[str, str]], **kwargs) -> tuple[dict, LLMResult]:
    """Chat with JSON output. Strips ``` fences if present."""
    res = chat(messages, json_mode=True, **kwargs)
    text = res.content.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.rsplit("```", 1)[0].strip()
    return json.loads(text), res
