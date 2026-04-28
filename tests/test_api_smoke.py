"""End-to-end smoke test for FastAPI flow.

Runs in-process via TestClient — no need to start uvicorn. Goes through
full screening flow on jd_005 with synthesised candidate responses.

Hits real OpenRouter (skips automatically if OPENROUTER_API_KEY missing).

    python -m pytest tests/test_api_smoke.py -v -s
"""
from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# Use a temp checkpoint DB for this test (don't pollute main one).
TMP_DB = ROOT / "data" / "_test_checkpoints.sqlite"
os.environ["CHECKPOINT_DB"] = str(TMP_DB)

import sys
sys.path.insert(0, str(ROOT))
from src.api import app  # noqa: E402

client = TestClient(app)

REQUIRES_API = pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set",
)


def _adaptive_response(question: str) -> str:
    """Return a plausibly-on-topic response for any generated question.

    Chooses one of three pre-written paragraphs based on keyword match.
    Guarantees that responses won't be 100% off-topic, so judge scores
    stay realistic for the smoke test.
    """
    q = question.lower()
    if "langgraph" in q or "agent" in q or "multi-agent" in q or "orchestr" in q:
        return ("I've used LangGraph for ~8 months on a contract-review pilot "
                "with three agents (triage, retrieval, drafting). Shared state "
                "via SqliteSaver in dev, Postgres checkpointer in prod. The "
                "explicit state machine made loops and tool errors easy to debug "
                "compared to the LangChain agent executor.")
    if "python" in q or "experience" in q or "project" in q or "ll" in q:
        return ("I've shipped a few production Python services on top of LLMs — "
                "most relevant: a support-ticket triage agent that processes "
                "~2k tickets/day, written with FastAPI + LangGraph + Pinecone, "
                "deployed on Fly.io with structured logging via Langfuse.")
    if "pdf" in q or "parser" in q or "document" in q or "extract" in q:
        return ("I've used pdfplumber for layout-sensitive contracts and "
                "Unstructured for messy scanned docs. For tables I usually "
                "fall back to Camelot. Edge case I hit: encrypted PDFs need a "
                "pre-step with pikepdf before any of these work.")
    if "debug" in q or "fail" in q or "error" in q or "wrong" in q:
        return ("Last week an agent kept hallucinating a clause that wasn't in "
                "the contract. I bisected by logging retrieval results — turned "
                "out the chunker split a sentence mid-clause and the model "
                "confabulated the rest. Fix: overlap chunks + add a verification "
                "pass that grounds each cited clause back to source span.")
    if "communic" in q or "non-technical" in q or "stakeholder" in q or "team" in q:
        return ("I write a short async update each Friday — what shipped, "
                "what's blocked, ask for help once explicitly. For agent demos "
                "to non-technical stakeholders I record a 3-min Loom rather "
                "than scheduling a call. Async-first because we're distributed.")
    return ("I'd want to learn more about your specific setup before answering "
            "in detail. From past projects: I lean on LangGraph, FastAPI, and "
            "either Pinecone or ChromaDB depending on scale. Happy to go deeper "
            "on any of these if you share which constraint matters most to you.")


@REQUIRES_API
def test_full_screening_flow():
    if TMP_DB.exists():
        TMP_DB.unlink()

    # 1. List JDs
    r = client.get("/jds")
    assert r.status_code == 200
    jds = r.json()
    assert len(jds) == 8
    assert any(j["jd_id"] == "jd_005" for j in jds)

    # 2. Start session
    t0 = time.time()
    r = client.post("/screening/start", json={"jd_id": "jd_005", "max_questions": 3})
    assert r.status_code == 200, r.text
    start = r.json()
    session_id = start["session_id"]
    assert start["jd_title"] == "Agentic AI Developer"
    assert start["total_questions"] == 3
    assert start["question_index"] == 0
    print(f"\n[start] {time.time() - t0:.1f}s — Q1: {start['question'][:80]}...")

    # 3. Loop respond until done
    next_q = start["question"]
    last_resp = None
    for turn in range(start["total_questions"]):
        t1 = time.time()
        ans = _adaptive_response(next_q)
        r = client.post("/screening/respond",
                        json={"session_id": session_id, "response": ans})
        assert r.status_code == 200, r.text
        last_resp = r.json()
        print(f"[respond {turn+1}] {time.time() - t1:.1f}s — done={last_resp['done']}")
        if last_resp["done"]:
            break
        next_q = last_resp["next_question"]
        print(f"  next Q: {next_q[:80]}...")

    assert last_resp["done"] is True
    assert last_resp["recommendation"] in (
        "strong_hire", "hire", "no_hire", "uncertain")

    # 4. Get full report
    r = client.get(f"/screening/{session_id}/report")
    assert r.status_code == 200
    report = r.json()
    assert report["session_id"] == session_id
    assert len(report["exchanges"]) == 3
    assert report["screening_report"].strip()
    assert report["total_cost_usd"] > 0
    print(f"\n[report] cost=${report['total_cost_usd']:.4f} "
          f"recommendation={report['recommendation']}")
    print(f"[report] total wall: {time.time() - t0:.1f}s")

    # 5. Re-start should NOT find this session — but also state endpoint
    r = client.get(f"/screening/{session_id}/state")
    assert r.status_code == 200
    assert r.json()["done"] is True
    assert r.json()["exchanges_count"] == 3

    # cleanup
    if TMP_DB.exists():
        TMP_DB.unlink()


@REQUIRES_API
def test_404_unknown_jd():
    r = client.post("/screening/start", json={"jd_id": "jd_999"})
    assert r.status_code == 404


@REQUIRES_API
def test_404_unknown_session():
    r = client.get("/screening/sess_nope/report")
    assert r.status_code == 404
