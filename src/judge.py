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


JUDGE_HALLUC_SYSTEM = """You audit a screening report for hallucinations.
Given the candidate's actual answers and the report, find every factual claim
in the report about the candidate that is NOT supported by their answers.

A claim is hallucinated if:
- it attributes a tool/skill/project/metric to the candidate that they never mentioned
- it invents specifics (numbers, names, dates) not in the answers
- it asserts the candidate did or didn't do X without evidence in the answers

Generic recruiter judgements ("strong communication", "lacks depth") are NOT
hallucinations — only fabricated facts about the candidate count.

Output strict JSON:
{"hallucinated_claims": ["...", "..."], "count": int, "reasoning": "..."}"""


def evaluate_report_hallucinations(*, report: str,
                                   exchanges: list[dict]) -> tuple[dict, float]:
    """Judge №3 — count fabricated claims in screening report."""
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
        model=model, temperature=0.0, max_tokens=600,
    )
    return data, res.cost_usd


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
