"""Generate candidate responses on behalf of a persona.

Used by eval suite — given a screening question + persona, produce a
plausible candidate answer in that persona's voice. Uses gpt-4o-mini for
cost (each 32-session eval ≈ 32 × 5 questions = 160 respondent calls).
"""
from __future__ import annotations

import os

from .llm import chat, LLMResult
from .personas import Persona


def respond(question: str, persona: Persona, jd_title: str,
            jd_seniority: str, model: str | None = None,
            temperature: float = 0.85) -> tuple[str, LLMResult]:
    """Return persona's answer to question. Cost tracked via LLMResult."""
    model = model or os.environ.get(
        "RESPONDENT_MODEL", "openai/gpt-4o-mini")
    sys_prompt = (
        persona.system_prompt
        + f"\n\nThe role you are interviewing for is "
        f"'{jd_title}' ({jd_seniority}). Stay in character throughout. "
        "Output only the answer text — no preamble like 'Sure, ...' or "
        "'Great question'."
    )
    res = chat(
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content":
                f"Recruiter asked: {question}\n\nYour answer:"},
        ],
        model=model, temperature=temperature, max_tokens=400,
    )
    return res.content.strip(), res
