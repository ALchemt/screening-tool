"""Re-run halluc judge v2 on existing eval rows without re-running candidates.

Reads runs/eval_v2_anti_halluc_metrics.jsonl, calls the new
evaluate_report_hallucinations() (which now separates fabricated vs concerns),
writes runs/eval_v2_v3_metrics.jsonl + runs/eval_v2_v3_summary.json.

Usage:
    python scripts/rejudge_halluc.py
    python scripts/rejudge_halluc.py --limit 3      # sanity check
    python scripts/rejudge_halluc.py --in <path>    # custom input
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
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.judge import evaluate_report_hallucinations  # noqa: E402

DEFAULT_IN = ROOT / "runs" / "eval_v2_anti_halluc_metrics.jsonl"
DEFAULT_OUT = ROOT / "runs" / "eval_v2_v3_metrics.jsonl"
DEFAULT_SUMMARY = ROOT / "runs" / "eval_v2_v3_summary.json"


def run(in_path: Path, out_path: Path, summary_path: Path, limit: int | None):
    rows = [json.loads(l) for l in in_path.read_text().splitlines() if l.strip()]
    if limit:
        rows = rows[:limit]
    print(f"Re-judging halluc on {len(rows)} sessions from {in_path.name}")

    extra_cost = 0.0
    enriched: list[dict] = []
    for i, row in enumerate(rows, 1):
        t0 = time.time()
        try:
            data, c = evaluate_report_hallucinations(
                report=row["screening_report"],
                exchanges=row["exchanges"],
            )
            extra_cost += c
        except Exception as e:
            print(f"  [{i}] FAIL — {e}")
            data = {"fabricated": [], "concerns": [],
                    "count_fabricated": -1, "count_concerns": -1,
                    "reasoning": f"error: {e}"}

        enriched_row = {
            **row,
            "fabricated_v3": data.get("fabricated", []),
            "concerns_v3": data.get("concerns", []),
            "count_fabricated_v3": data.get("count_fabricated", -1),
            "count_concerns_v3": data.get("count_concerns", -1),
            "halluc_reasoning_v3": data.get("reasoning", ""),
        }
        enriched.append(enriched_row)
        print(f"  [{i}/{len(rows)}] {row['jd_id']}/{row['persona_id']} "
              f"fab={enriched_row['count_fabricated_v3']} "
              f"conc={enriched_row['count_concerns_v3']} "
              f"t={time.time()-t0:.1f}s", flush=True)

    out_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in enriched))
    print(f"\nWrote {out_path.name}")

    fabs = [r["count_fabricated_v3"] for r in enriched
            if r["count_fabricated_v3"] >= 0]
    concs = [r["count_concerns_v3"] for r in enriched
             if r["count_concerns_v3"] >= 0]
    total_q = sum(len(r["questions"]) for r in enriched)

    summary = {
        "n_sessions": len(rows),
        "total_questions": total_q,
        "fabricated_total": sum(fabs),
        "fabricated_per_session_mean": round(statistics.mean(fabs), 3) if fabs else None,
        "fabricated_rate_pct": round(100 * sum(fabs) / max(1, total_q), 2),
        "concerns_total": sum(concs),
        "concerns_per_session_mean": round(statistics.mean(concs), 3) if concs else None,
        "extra_eval_cost_usd": round(extra_cost, 4),
        "judge_prompt_version": "v3_separated_concerns",
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote {summary_path.name}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    ap.add_argument("--limit", type=int, default=None,
                    help="re-judge only first N rows (sanity)")
    args = ap.parse_args()
    run(args.in_path, args.out, args.summary, args.limit)


if __name__ == "__main__":
    main()
