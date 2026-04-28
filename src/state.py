"""LangGraph state schema for a screening session."""
from __future__ import annotations

from typing import Annotated, Any, Optional, TypedDict
from operator import add


class JudgeScore(TypedDict):
    technical_accuracy: int  # 0/1/2
    depth: int
    communication: int
    red_flags: int  # 0 = none, 1 = minor, 2 = major
    reasoning: str


class QAExchange(TypedDict):
    question: str
    response: str
    judge: JudgeScore
    cost_usd: float


class ScreeningState(TypedDict, total=False):
    # Inputs
    session_id: str
    jd_id: str
    candidate_responses: list[str]  # batch mode: pre-loaded responses
    max_questions: int

    # Loaded JD context
    jd_title: str
    jd_seniority: str
    jd_company: str
    jd_role_summary: str
    jd_chunks: list[dict[str, Any]]  # retrieval results

    # Loop state
    questions: Annotated[list[str], add]
    exchanges: Annotated[list[QAExchange], add]
    current_index: int

    # Final
    screening_report: Optional[str]
    final_recommendation: Optional[str]  # "strong_hire" | "hire" | "no_hire" | "uncertain"
    total_cost_usd: float
