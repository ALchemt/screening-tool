"""LLM-as-judge for candidate responses.

Rubric: technical_accuracy / depth / communication / red_flags (0/1/2 each).
Uses JUDGE_MODEL (default claude-sonnet-4.5) — different family from agent
to avoid self-bias (lesson from P1 RAG).
"""
from __future__ import annotations

import os
from typing import Any

from .llm import chat_json
from .state import JudgeScore

JUDGE_SYSTEM = """You are a senior technical recruiter evaluating a candidate's
response to a screening question. Score strictly using the rubric.

Rubric (each 0/1/2):
- technical_accuracy: 0=wrong/fabricated, 1=partially correct, 2=correct and precise
- depth: 0=surface-level/buzzwords, 1=some specifics, 2=concrete examples + tradeoffs
- communication: 0=unclear/rambling, 1=acceptable, 2=structured and concise
- red_flags: 0=none, 1=minor concern (vague non-answer, mild contradiction),
             2=major (lying, hostility, off-topic refusal)

Output strict JSON only. No prose outside JSON."""

JUDGE_USER_TEMPLATE = """Job context:
- Role: {title} ({seniority})
- Company: {company}
- Relevant JD requirements (retrieved via RAG):
{jd_context}

Question asked: {question}

Candidate response:
{response}

Return JSON with keys exactly:
{{
  "technical_accuracy": int,
  "depth": int,
  "communication": int,
  "red_flags": int,
  "reasoning": "1-2 sentences citing specific evidence from the response"
}}"""


def evaluate(*, question: str, response: str,
             jd_title: str, jd_seniority: str, jd_company: str,
             jd_context: str) -> tuple[JudgeScore, float]:
    """Judge one Q/A exchange. Returns (score_dict, cost_usd)."""
    model = os.environ.get("JUDGE_MODEL", "anthropic/claude-sonnet-4.5")
    user = JUDGE_USER_TEMPLATE.format(
        title=jd_title, seniority=jd_seniority, company=jd_company,
        jd_context=jd_context, question=question, response=response,
    )
    data, res = chat_json(
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user},
        ],
        model=model, temperature=0.2, max_tokens=512,
    )
    score: JudgeScore = {
        "technical_accuracy": int(data.get("technical_accuracy", 0)),
        "depth": int(data.get("depth", 0)),
        "communication": int(data.get("communication", 0)),
        "red_flags": int(data.get("red_flags", 0)),
        "reasoning": str(data.get("reasoning", "")).strip(),
    }
    return score, res.cost_usd


JUDGE_RELEVANCE_SYSTEM = """You evaluate whether a screening question is relevant
to a job description. Score 0/1/2:
- 0: off-topic (not related to required skills)
- 1: tangentially related
- 2: directly probes a required qualification

Output strict JSON: {"score": int, "reasoning": "..."}"""


JUDGE_HALLUC_SYSTEM = """You audit a screening report and classify every
factual claim about the candidate into one of three buckets.

Categories:

1. FABRICATED — a positive factual claim about the candidate that has NO support
   in their answers. Counts only if the report ASSERTS something the candidate
   did/used/built/achieved that they did not actually say.
   Examples:
   - "Candidate used Pinecone in production" (they never mentioned Pinecone)
   - "Candidate has 5 years of LangChain experience" (no number was given)
   - "Built a chatbot for DataVision Inc." (no employer named)

2. CONCERN — a negative observation about gaps or missing evidence.
   THESE ARE NOT FABRICATIONS. They are honest disclaimers and must be
   classified as concerns, not fabrications.
   Examples:
   - "Did not address evaluation methodology"
   - "Did not demonstrate hands-on RAG experience"
   - "Lacks evidence of production deployment"
   - "No specifics on chunking strategy were provided"
   - "Candidate did not mention vector databases"

3. JUDGEMENT — generic recruiter assessments / value statements.
   THESE ARE NOT FABRICATIONS.
   Examples:
   - "Strong communication"
   - "Surface-level understanding"
   - "Recommend lean hire"
   - "Answers were structured and concise"

Decision rule: when in doubt between FABRICATED and CONCERN, prefer CONCERN.
A claim only counts as FABRICATED if the report invents a positive fact.
Negative phrasings about absence ("did not", "lacks", "no evidence of") are
ALWAYS concerns, never fabrications.

Output strict JSON:
{
  "fabricated": ["exact quote from report", ...],
  "concerns": ["exact quote from report", ...],
  "count_fabricated": int,
  "count_concerns": int,
  "reasoning": "1-2 sentences"
}"""


def evaluate_report_hallucinations(*, report: str,
                                   exchanges: list[dict]) -> tuple[dict, float]:
    """Judge №3 — classify report claims into fabricated/concerns/judgements.

    Returns dict with keys: fabricated, concerns, count_fabricated,
    count_concerns, reasoning. Backwards-compat keys 'count' and
    'hallucinated_claims' mirror the fabricated bucket.
    """
    model = os.environ.get("JUDGE_MODEL", "anthropic/claude-sonnet-4.5")
    transcript = "\n\n".join(
        f"Q: {e['question']}\nA: {e['response']}" for e in exchanges
    )
    data, res = chat_json(
        messages=[
            {"role": "system", "content": JUDGE_HALLUC_SYSTEM},
            {"role": "user", "content":
                f"Candidate answers:\n{transcript}\n\n"
                f"Generated report:\n{report}"},
        ],
        model=model, temperature=0.0, max_tokens=900,
    )
    fabricated = data.get("fabricated", []) or []
    concerns = data.get("concerns", []) or []
    out = {
        "fabricated": fabricated,
        "concerns": concerns,
        "count_fabricated": int(data.get("count_fabricated", len(fabricated))),
        "count_concerns": int(data.get("count_concerns", len(concerns))),
        "reasoning": str(data.get("reasoning", "")).strip(),
        # backwards-compat for older eval_metrics callers
        "hallucinated_claims": fabricated,
        "count": int(data.get("count_fabricated", len(fabricated))),
    }
    return out, res.cost_usd


def evaluate_question_relevance(*, question: str, jd_text: str) -> tuple[dict, float]:
    """Judge №2 — used in eval suite for question_relevance metric."""
    model = os.environ.get("JUDGE_MODEL", "anthropic/claude-sonnet-4.5")
    data, res = chat_json(
        messages=[
            {"role": "system", "content": JUDGE_RELEVANCE_SYSTEM},
            {"role": "user", "content":
                f"JD:\n{jd_text}\n\nQuestion: {question}"},
        ],
        model=model, temperature=0.0, max_tokens=400,
    )
    return data, res.cost_usd
