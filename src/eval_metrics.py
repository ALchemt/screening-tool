"""Aggregate eval metrics from runs/eval_*.jsonl.

5 metrics from the spec:
  1. question_relevance — judge #2 scores each Q vs JD (0/1/2), mean ≥ 1.7 target
  2. judge_human_agreement — % match with human labels (loaded separately)
  3. hallucination_rate — judge #3 counts fabricated claims in report
  4. cost_per_session — already in jsonl
  5. latency_p50/p95 — already in jsonl

Run:
    python -m src.eval_metrics runs/eval_20260428_1430.jsonl
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

from .judge import evaluate_question_relevance, evaluate_report_hallucinations  # noqa: E402
from .jd_corpus import parse_jd_file  # noqa: E402

JD_DIR = ROOT / "data" / "jd_corpus"


def _load_jd_text(jd_id: str) -> str:
    _, body = parse_jd_file(JD_DIR / f"{jd_id}.md")
    return body


def compute_metrics(jsonl_path: Path, out_path: Path | None = None) -> dict:
    rows = [json.loads(l) for l in jsonl_path.read_text().splitlines() if l.strip()]
    print(f"Loaded {len(rows)} sessions from {jsonl_path.name}")

    extra_cost = 0.0
    enriched: list[dict] = []
    jd_text_cache: dict[str, str] = {}

    for i, row in enumerate(rows, 1):
        t0 = time.time()
        jd_id = row["jd_id"]
        if jd_id not in jd_text_cache:
            jd_text_cache[jd_id] = _load_jd_text(jd_id)
        jd_text = jd_text_cache[jd_id]

        # 1. Relevance per question
        rel_scores = []
        for q in row["questions"]:
            try:
                data, c = evaluate_question_relevance(
                    question=q, jd_text=jd_text)
                extra_cost += c
                rel_scores.append(int(data.get("score", 0)))
            except Exception as e:
                print(f"    relevance fail on '{q[:40]}...' — {e}")
                rel_scores.append(-1)  # sentinel for failed

        # 2. Hallucination audit on report
        try:
            halluc, c = evaluate_report_hallucinations(
                report=row["screening_report"],
                exchanges=row["exchanges"],
            )
            extra_cost += c
            halluc_count = int(halluc.get(
                "count", len(halluc.get("hallucinated_claims", []))))
            halluc_claims = halluc.get("hallucinated_claims", [])
        except Exception as e:
            print(f"    halluc audit fail — {e}")
            halluc_count = -1
            halluc_claims = []

        valid_rel = [s for s in rel_scores if s >= 0]
        enriched.append({
            **row,
            "question_relevance_scores": rel_scores,
            "question_relevance_mean": round(
                statistics.mean(valid_rel), 3) if valid_rel else None,
            "hallucinated_claims": halluc_claims,
            "hallucination_count": halluc_count,
        })
        print(f"  [{i}/{len(rows)}] {jd_id}/{row['persona_id']} "
              f"rel_mean={enriched[-1]['question_relevance_mean']} "
              f"halluc={halluc_count} "
              f"t={time.time()-t0:.1f}s", flush=True)

    # Aggregate
    rel_means = [r["question_relevance_mean"] for r in enriched
                 if r["question_relevance_mean"] is not None]
    halluc_counts = [r["hallucination_count"] for r in enriched
                     if r["hallucination_count"] >= 0]
    questions_per_session = [len(r["questions"]) for r in enriched]
    total_questions = sum(questions_per_session)
    total_halluc_claims = sum(halluc_counts)
    costs = [r["total_cost_usd"] for r in enriched]
    latencies = [r["latency_seconds"] for r in enriched]

    # Recommendation distribution per persona level
    by_level: dict[str, dict[str, int]] = {}
    for r in enriched:
        lvl = r["persona_level"]
        rec = r["recommendation"]
        by_level.setdefault(lvl, {}).setdefault(rec, 0)
        by_level[lvl][rec] += 1

    summary = {
        "n_sessions": len(rows),
        "total_questions": total_questions,
        "question_relevance_mean": round(statistics.mean(rel_means), 3),
        "question_relevance_min": round(min(rel_means), 3),
        "hallucination_total_claims": total_halluc_claims,
        "hallucination_per_session_mean": round(
            statistics.mean(halluc_counts), 3),
        "hallucination_rate_pct": round(
            100 * total_halluc_claims / max(1, total_questions), 2),
        "cost_per_session_mean": round(statistics.mean(costs), 4),
        "cost_per_session_max": round(max(costs), 4),
        "cost_total": round(sum(costs), 4),
        "extra_eval_cost": round(extra_cost, 4),
        "latency_p50": round(statistics.median(latencies), 2),
        "latency_p95": round(
            sorted(latencies)[int(0.95 * (len(latencies) - 1))], 2),
        "recommendation_by_level": by_level,
    }

    if out_path is None:
        out_path = jsonl_path.with_name(jsonl_path.stem + "_metrics.jsonl")
    enriched_path = out_path
    summary_path = out_path.with_name(out_path.stem.replace("_metrics", "")
                                      + "_summary.json")
    enriched_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in enriched))
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\nSummary written → {summary_path.name}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", type=Path, help="path to eval jsonl")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    compute_metrics(args.jsonl, args.out)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
