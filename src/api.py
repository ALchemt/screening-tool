"""FastAPI app — REST entrypoint to screening flow.

Endpoints:
  POST /screening/start          — create session, return first question
  POST /screening/respond        — submit answer, get next question or final report
  GET  /screening/{id}/report    — final report (404 if not done)
  GET  /screening/{id}/state     — current session state (debug)
  GET  /healthz                  — liveness
  GET  /jds                      — list available JDs (id + title)

Persistence: LangGraph SqliteSaver checkpoints session state across requests
(thread_id == session_id). DB path: $CHECKPOINT_DB or data/checkpoints.sqlite.

Pydantic v2 request/response models. Run with `uvicorn src.api:app`.
"""
from __future__ import annotations

import os
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel, Field

from .agent import build_interactive_graph
from .jd_corpus import parse_jd_file

# Explicit Langfuse client init — drop-in import alone doesn't always auto-init in v3.
_LF_CLIENT = None
if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
    try:
        from langfuse import Langfuse  # type: ignore
        _LF_CLIENT = Langfuse(
            public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
            secret_key=os.environ["LANGFUSE_SECRET_KEY"],
            host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
        print(f"[langfuse] client initialized; host={os.environ.get('LANGFUSE_HOST')}", flush=True)
    except Exception as e:
        print(f"[langfuse] init failed: {e}", flush=True)
else:
    print("[langfuse] keys not set — tracing disabled", flush=True)


def _flush_langfuse():
    if _LF_CLIENT is not None:
        try:
            _LF_CLIENT.flush()
        except Exception as e:
            print(f"[langfuse] flush failed: {e}", flush=True)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data" / "checkpoints.sqlite"

app = FastAPI(
    title="Screening Tool API",
    description="LangGraph-powered recruiter screening with LLM-as-judge.",
    version="0.1.0",
)


# ─── Models ─────────────────────────────────────────────────────────────────

class StartRequest(BaseModel):
    jd_id: str = Field(..., examples=["jd_005"])
    max_questions: int = Field(5, ge=1, le=10)
    session_id: Optional[str] = Field(None, description="Auto-generated if omitted")


class StartResponse(BaseModel):
    session_id: str
    jd_id: str
    jd_title: str
    question_index: int
    total_questions: int
    question: str


class RespondRequest(BaseModel):
    session_id: str
    response: str = Field(..., min_length=1)


class RespondResponse(BaseModel):
    session_id: str
    done: bool
    question_index: Optional[int] = None  # next question, if any
    total_questions: int
    next_question: Optional[str] = None
    recommendation: Optional[str] = None  # only when done
    report_url: Optional[str] = None      # only when done


class ReportResponse(BaseModel):
    session_id: str
    jd_id: str
    jd_title: str
    jd_company: str
    recommendation: str
    total_cost_usd: float
    exchanges: list[dict]
    screening_report: str


class JDListItem(BaseModel):
    jd_id: str
    title: str
    seniority: str
    company: str


# ─── Helpers ────────────────────────────────────────────────────────────────

def _config(session_id: str) -> dict:
    return {"configurable": {"thread_id": session_id}}


@contextmanager
def _graph():
    """Open SqliteSaver and yield a compiled interactive graph."""
    db_path = os.environ.get("CHECKPOINT_DB", str(DEFAULT_DB))
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(db_path) as saver:
        yield build_interactive_graph(saver)


def _state_values(graph, config: dict) -> dict:
    snap = graph.get_state(config)
    return snap.values if snap else {}


# ─── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/jds", response_model=list[JDListItem])
def list_jds():
    out = []
    for path in sorted((ROOT / "data" / "jd_corpus").glob("jd_*.md")):
        fm, _ = parse_jd_file(path)
        out.append(JDListItem(
            jd_id=fm["id"], title=fm["title"],
            seniority=fm["seniority"], company=fm["company"],
        ))
    return out


@app.post("/screening/start", response_model=StartResponse)
def start_screening(req: StartRequest):
    session_id = req.session_id or f"sess_{uuid.uuid4().hex[:12]}"
    jd_path = ROOT / "data" / "jd_corpus" / f"{req.jd_id}.md"
    if not jd_path.exists():
        raise HTTPException(404, f"JD {req.jd_id} not found")

    with _graph() as graph:
        config = _config(session_id)
        # Runs through load_jd + generate_questions, then interrupts before
        # the first evaluate_response (no candidate responses yet).
        graph.invoke({
            "session_id": session_id,
            "jd_id": req.jd_id,
            "candidate_responses": [],
            "max_questions": req.max_questions,
        }, config)
        state = _state_values(graph, config)

    questions = state.get("questions", [])
    if not questions:
        raise HTTPException(500, "Question generation failed")
    _flush_langfuse()
    return StartResponse(
        session_id=session_id,
        jd_id=req.jd_id,
        jd_title=state.get("jd_title", ""),
        question_index=0,
        total_questions=len(questions),
        question=questions[0],
    )


@app.post("/screening/respond", response_model=RespondResponse)
def respond(req: RespondRequest):
    with _graph() as graph:
        config = _config(req.session_id)
        snap = graph.get_state(config)
        if not snap or not snap.values:
            raise HTTPException(404, f"Session {req.session_id} not found")

        prev_responses = snap.values.get("candidate_responses", [])
        graph.update_state(config, {
            "candidate_responses": prev_responses + [req.response],
        })

        # Resume: runs evaluate_response, then either interrupts before
        # the next one (more questions) or runs final_report → END.
        graph.invoke(None, config)
        state = _state_values(graph, config)

    questions = state.get("questions", [])
    idx = state.get("current_index", 0)
    _flush_langfuse()
    if state.get("screening_report"):
        return RespondResponse(
            session_id=req.session_id,
            done=True,
            total_questions=len(questions),
            recommendation=state.get("final_recommendation"),
            report_url=f"/screening/{req.session_id}/report",
        )
    if idx < len(questions):
        return RespondResponse(
            session_id=req.session_id,
            done=False,
            question_index=idx,
            total_questions=len(questions),
            next_question=questions[idx],
        )
    raise HTTPException(500, "Inconsistent state: not done but no next question")


@app.get("/screening/{session_id}/report", response_model=ReportResponse)
def get_report(session_id: str):
    with _graph() as graph:
        snap = graph.get_state(_config(session_id))
    if not snap or not snap.values:
        raise HTTPException(404, f"Session {session_id} not found")
    state = snap.values
    if not state.get("screening_report"):
        raise HTTPException(409, "Session not completed yet")
    return ReportResponse(
        session_id=session_id,
        jd_id=state["jd_id"],
        jd_title=state.get("jd_title", ""),
        jd_company=state.get("jd_company", ""),
        recommendation=state.get("final_recommendation", "uncertain"),
        total_cost_usd=float(state.get("total_cost_usd", 0.0)),
        exchanges=list(state.get("exchanges", [])),
        screening_report=state["screening_report"],
    )


@app.get("/screening/{session_id}/state")
def get_state(session_id: str):
    """Debug: dump raw state."""
    with _graph() as graph:
        snap = graph.get_state(_config(session_id))
    if not snap or not snap.values:
        raise HTTPException(404, f"Session {session_id} not found")
    return {
        "session_id": session_id,
        "current_index": snap.values.get("current_index", 0),
        "questions_count": len(snap.values.get("questions", [])),
        "exchanges_count": len(snap.values.get("exchanges", [])),
        "done": bool(snap.values.get("screening_report")),
        "next_node": list(snap.next) if snap.next else [],
    }
