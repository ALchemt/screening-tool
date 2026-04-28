"""Run the eval matrix: 4 JDs × 8 personas = 32 sessions.

For each session:
  1. node_load_jd  — pull JD ctx + RAG chunks
  2. node_generate_questions — 5 questions
  3. for each Q: respondent.respond(persona) → answer
  4. node_evaluate_response × 5 — judge each (Q, A)
  5. node_final_report — synthesised report + recommendation

Writes one JSON row per session to runs/eval_<timestamp>.jsonl with:
  session_id, jd_id, persona_id, persona_level, questions, exchanges,
  screening_report, recommendation, cost breakdown, latency_seconds.

Usage:
    python scripts/run_eval.py            # all 32 sessions
    python scripts/run_eval.py --limit 4  # smoke: first 4 sessions only
    python scripts/run_eval.py --jds jd_005 --personas strong_a,weak_a
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.agent import (  # noqa: E402
    node_load_jd, node_generate_questions,
    node_evaluate_response, node_final_report,
)
from src.personas import PERSONAS, get as get_persona  # noqa: E402
from src.respondent import respond  # noqa: E402

DEFAULT_JDS = ["jd_002", "jd_003", "jd_005", "jd_007"]
DEFAULT_PERSONAS = [p.persona_id for p in PERSONAS]
N_QUESTIONS = 5


def run_session(jd_id: str, persona_id: str) -> dict:
    persona = get_persona(persona_id)
    session_id = f"eval_{jd_id}_{persona_id}"
    t0 = time.time()

    state: dict = {
        "session_id": session_id,
        "jd_id": jd_id,
        "candidate_responses": [],
        "max_questions": N_QUESTIONS,
    }

    # 1. Load JD
    state.update(node_load_jd(state))
    # 2. Generate questions
    state.update(node_generate_questions(state))
    questions = state["questions"]

    # 3. Respondent answers each question
    respondent_cost = 0.0
    responses: list[str] = []
    for q in questions:
        ans, res = respond(
            q, persona,
            jd_title=state["jd_title"],
            jd_seniority=state["jd_seniority"],
        )
        responses.append(ans)
        respondent_cost += res.cost_usd
    state["candidate_responses"] = responses

    # 4. Evaluate each response — node mutates current_index
    exchanges_acc: list[dict] = []
    while state["current_index"] < len(questions):
        upd = node_evaluate_response(state)
        # node returns {"exchanges": [new], "current_index": i+1, "total_cost_usd": ...}
        exchanges_acc.extend(upd["exchanges"])
        state["current_index"] = upd["current_index"]
        state["total_cost_usd"] = upd["total_cost_usd"]
    state["exchanges"] = exchanges_acc

    # 5. Final report
    state.update(node_final_report(state))

    latency = time.time() - t0
    return {
        "session_id": session_id,
        "jd_id": jd_id,
        "jd_title": state["jd_title"],
        "jd_seniority": state["jd_seniority"],
        "jd_company": state["jd_company"],
        "persona_id": persona_id,
        "persona_level": persona.level,
        "persona_name": persona.name,
        "questions": questions,
        "exchanges": state["exchanges"],
        "screening_report": state["screening_report"],
        "recommendation": state["final_recommendation"],
        "agent_judge_cost_usd": round(state["total_cost_usd"], 6),
        "respondent_cost_usd": round(respondent_cost, 6),
        "total_cost_usd": round(state["total_cost_usd"] + respondent_cost, 6),
        "latency_seconds": round(latency, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jds", default=",".join(DEFAULT_JDS),
                    help="comma-separated JD IDs")
    ap.add_argument("--personas", default=",".join(DEFAULT_PERSONAS),
                    help="comma-separated persona IDs")
    ap.add_argument("--limit", type=int, default=None,
                    help="run only first N (jd, persona) pairs")
    ap.add_argument("--out", default=None, help="output jsonl path")
    args = ap.parse_args()

    jds = [s.strip() for s in args.jds.split(",") if s.strip()]
    persona_ids = [s.strip() for s in args.personas.split(",") if s.strip()]
    pairs = [(j, p) for j in jds for p in persona_ids]
    if args.limit:
        pairs = pairs[: args.limit]

    out_path = Path(args.out) if args.out else (
        ROOT / "runs" / f"eval_{datetime.now().strftime('%Y%m%d_%H%M')}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        rel = out_path.resolve().relative_to(ROOT)
    except ValueError:
        rel = out_path
    print(f"Running {len(pairs)} sessions → {rel}")
    total_cost = 0.0
    t_start = time.time()
    with out_path.open("w") as fh:
        for i, (jd_id, persona_id) in enumerate(pairs, 1):
            t0 = time.time()
            try:
                row = run_session(jd_id, persona_id)
            except Exception as e:
                print(f"  [{i}/{len(pairs)}] {jd_id}/{persona_id} FAILED: {e}")
                continue
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            total_cost += row["total_cost_usd"]
            print(f"  [{i}/{len(pairs)}] {jd_id}/{persona_id} "
                  f"({row['persona_level']}) — "
                  f"rec={row['recommendation']:<12} "
                  f"cost=${row['total_cost_usd']:.4f} "
                  f"t={time.time()-t0:.1f}s")

    print(f"\nDone. {len(pairs)} sessions, "
          f"total cost ${total_cost:.4f}, "
          f"wall {time.time()-t_start:.1f}s")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
