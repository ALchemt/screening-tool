"""LangGraph state machine for recruiter screening.

Nodes:
  load_jd          — pull JD context (top RAG chunks) into state
  generate_questions — produce N adaptive screening questions grounded in JD
  ask_question     — emit current question (placeholder; in batch mode no-op)
  evaluate_response — call judge on (question, response)
  final_report     — synthesise hiring report + recommendation
  decide_continue  — conditional edge: more questions or report?

Batch mode (used for eval and CLI): candidate_responses pre-loaded,
evaluate_response pulls i-th response, loop ends when index >= len.

Interactive mode (future): same graph + interrupt() before evaluate_response.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from langgraph.graph import StateGraph, START, END

from .judge import evaluate
from .jd_corpus import parse_jd_file
from .llm import chat, chat_json
from .retrieval import retrieve
from .state import ScreeningState, QAExchange

# Optional Langfuse tracing — only active when keys are set.
if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
    from langfuse import observe  # type: ignore
else:
    def observe(*args, **kwargs):  # noqa: D401
        """No-op fallback when Langfuse not configured."""
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def deco(fn):
            return fn
        return deco

ROOT = Path(__file__).resolve().parents[1]


# ─── Nodes ──────────────────────────────────────────────────────────────────

@observe(name="load_jd")
def node_load_jd(state: ScreeningState) -> dict:
    jd_path = ROOT / "data" / "jd_corpus" / f"{state['jd_id']}.md"
    fm, body = parse_jd_file(jd_path)

    # Pull top RAG chunks for THIS jd (Required Quals + Responsibilities).
    chunks = retrieve(
        query=f"{fm['title']} required qualifications responsibilities",
        k=4, where={"jd_id": state["jd_id"]},
    )
    return {
        "jd_title": fm["title"],
        "jd_seniority": fm["seniority"],
        "jd_company": fm["company"],
        "jd_role_summary": body[:1500],
        "jd_chunks": [{"section": c.metadata["section"], "text": c.text}
                      for c in chunks],
        "current_index": 0,
        "total_cost_usd": 0.0,
    }


GEN_Q_SYSTEM = """You are a senior technical recruiter conducting a phone screen.
Generate exactly {n} screening questions that probe whether the candidate matches
the JD. Mix:
- 1-2 specific technical questions tied to listed tools/skills
- 1 scenario/problem-solving question
- 1 experience-depth question
- 1 question on a nice-to-have or edge case

Questions should be open-ended (not yes/no) and answerable in 30-90 seconds.
Output strict JSON: {{"questions": ["q1", "q2", ...]}}"""


@observe(name="generate_questions")
def node_generate_questions(state: ScreeningState) -> dict:
    n = state.get("max_questions", 5)
    jd_ctx = "\n\n".join(c["text"] for c in state["jd_chunks"])
    data, res = chat_json(
        messages=[
            {"role": "system", "content": GEN_Q_SYSTEM.format(n=n)},
            {"role": "user", "content":
                f"Role: {state['jd_title']} ({state['jd_seniority']}) "
                f"@ {state['jd_company']}\n\n"
                f"JD context:\n{jd_ctx}"},
        ],
        model=os.environ.get("AGENT_MODEL", "openai/gpt-4o-mini"),
        temperature=0.7, max_tokens=900,
    )
    questions = [q.strip() for q in data.get("questions", []) if q.strip()][:n]
    return {
        "questions": questions,
        "total_cost_usd": state.get("total_cost_usd", 0.0) + res.cost_usd,
    }


@observe(name="evaluate_response")
def node_evaluate_response(state: ScreeningState) -> dict:
    """Pull i-th candidate response (batch mode), score it, append exchange."""
    i = state["current_index"]
    question = state["questions"][i]
    responses = state.get("candidate_responses", [])
    if i >= len(responses):
        # Edge: fewer responses than questions — synthesise empty.
        response_text = "[no response provided]"
    else:
        response_text = responses[i]

    jd_ctx = "\n".join(c["text"] for c in state["jd_chunks"][:2])
    score, judge_cost = evaluate(
        question=question, response=response_text,
        jd_title=state["jd_title"],
        jd_seniority=state["jd_seniority"],
        jd_company=state["jd_company"],
        jd_context=jd_ctx,
    )
    exchange: QAExchange = {
        "question": question,
        "response": response_text,
        "judge": score,
        "cost_usd": judge_cost,
    }
    return {
        "exchanges": [exchange],
        "current_index": i + 1,
        "total_cost_usd": state.get("total_cost_usd", 0.0) + judge_cost,
    }


def decide_continue(state: ScreeningState) -> Literal["evaluate_response", "final_report"]:
    if state["current_index"] < len(state.get("questions", [])):
        return "evaluate_response"
    return "final_report"


REPORT_SYSTEM = """You are a senior technical recruiter writing a screening report.
Be concise, evidence-based, and honest.

STRICT GROUNDING RULES (mandatory — violations make the report invalid):
- Every factual claim about the candidate MUST be supported by an exact quote
  from their answers in the transcript. If you cannot quote it, do not claim it.
- DO NOT invent: tools/libraries the candidate did not name, employer or
  product names, durations of experience ("X+ years/months"), metrics
  (req/s, accuracy %, dataset sizes), credentials, locations.
- If the candidate did not mention a topic, write "did not address" — never
  speculate what they "probably" know or "would" do beyond their own words.
- Generic recruiter judgements ("strong communication", "lacks depth on X")
  are allowed when grounded in the judge's rubric scores.
- Quote candidate text verbatim using "..." — paraphrasing is allowed but
  must not add new facts.

Output structure:
## Summary (2-3 sentences, evidence-grounded only)
## Strengths (bullets, each with a quoted candidate phrase)
## Concerns (bullets; cite the question topic and what was missing)
## Rubric scores
| dimension | mean |
| ... |
## Recommendation
One of: strong_hire / hire / no_hire / uncertain — plus one-sentence rationale that cites the rubric.

Then on a new line, output exactly: RECOMMENDATION: <value>"""


@observe(name="final_report")
def node_final_report(state: ScreeningState) -> dict:
    exchanges = state.get("exchanges", [])
    transcript = "\n\n".join(
        f"Q{i+1}: {e['question']}\nA: {e['response']}\n"
        f"Judge: TA={e['judge']['technical_accuracy']} D={e['judge']['depth']} "
        f"C={e['judge']['communication']} RF={e['judge']['red_flags']} — "
        f"{e['judge']['reasoning']}"
        for i, e in enumerate(exchanges)
    )
    res = chat(
        messages=[
            {"role": "system", "content": REPORT_SYSTEM},
            {"role": "user", "content":
                f"Role: {state['jd_title']} ({state['jd_seniority']}) "
                f"@ {state['jd_company']}\n\n"
                f"Screening transcript with judge scores:\n\n{transcript}"},
        ],
        model=os.environ.get("AGENT_MODEL", "openai/gpt-4o-mini"),
        temperature=0.1, max_tokens=900,
    )
    report = res.content.strip()
    rec = "uncertain"
    for line in report.splitlines():
        if line.upper().startswith("RECOMMENDATION:"):
            rec = line.split(":", 1)[1].strip().lower()
            break
    return {
        "screening_report": report,
        "final_recommendation": rec,
        "total_cost_usd": state.get("total_cost_usd", 0.0) + res.cost_usd,
    }


# ─── Graph ──────────────────────────────────────────────────────────────────

def _build(checkpointer=None, interactive: bool = False):
    g = StateGraph(ScreeningState)
    g.add_node("load_jd", node_load_jd)
    g.add_node("generate_questions", node_generate_questions)
    g.add_node("evaluate_response", node_evaluate_response)
    g.add_node("final_report", node_final_report)

    g.add_edge(START, "load_jd")
    g.add_edge("load_jd", "generate_questions")
    g.add_edge("generate_questions", "evaluate_response")
    g.add_conditional_edges(
        "evaluate_response", decide_continue,
        {"evaluate_response": "evaluate_response",
         "final_report": "final_report"},
    )
    g.add_edge("final_report", END)

    kwargs = {}
    if checkpointer is not None:
        kwargs["checkpointer"] = checkpointer
    if interactive:
        # Pause before each evaluate_response so the API can inject
        # the next candidate response between turns.
        kwargs["interrupt_before"] = ["evaluate_response"]
    return g.compile(**kwargs)


def build_graph():
    """Batch-mode graph: candidate_responses pre-loaded, runs to END."""
    return _build()


def build_interactive_graph(checkpointer):
    """Interactive graph for FastAPI: interrupts before evaluate_response."""
    return _build(checkpointer=checkpointer, interactive=True)


def run_screening(*, jd_id: str, candidate_responses: list[str],
                  session_id: str = "smoke", max_questions: int = 5) -> dict:
    graph = build_graph()
    final = graph.invoke({
        "session_id": session_id,
        "jd_id": jd_id,
        "candidate_responses": candidate_responses,
        "max_questions": max_questions,
    })
    return final
