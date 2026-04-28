"""CLI: run one screening session in batch mode.

    python scripts/run_screening.py --jd jd_005 --responses sample.json [--max-q 5]

`sample.json` — list of strings (candidate's pre-loaded answers).
If shorter than max_questions, missing answers become "[no response provided]".

Writes runs/<session_id>.json with full state + report.md alongside.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))

from src.agent import run_screening  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jd", required=True, help="jd_id, e.g. jd_005")
    ap.add_argument("--responses", required=True, help="path to JSON list of strings")
    ap.add_argument("--max-q", type=int, default=5)
    ap.add_argument("--session-id", default=None)
    args = ap.parse_args()

    responses = json.loads(Path(args.responses).read_text())
    session_id = args.session_id or f"{args.jd}_{int(time.time())}"

    t0 = time.time()
    final = run_screening(
        jd_id=args.jd,
        candidate_responses=responses,
        session_id=session_id,
        max_questions=args.max_q,
    )
    elapsed = time.time() - t0

    out_dir = ROOT / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{session_id}.json"
    md_path = out_dir / f"{session_id}.md"

    json_path.write_text(json.dumps(final, indent=2, default=str))
    md_path.write_text(
        f"# Screening report — {session_id}\n\n"
        f"- JD: **{final['jd_title']}** ({final['jd_seniority']}) "
        f"@ {final['jd_company']}\n"
        f"- Recommendation: **{final.get('final_recommendation', 'n/a')}**\n"
        f"- Total cost: ${final.get('total_cost_usd', 0):.4f}\n"
        f"- Wall time: {elapsed:.1f}s\n\n"
        f"---\n\n{final.get('screening_report', '(no report)')}\n"
    )

    print(f"\n=== {session_id} done ===")
    print(f"  recommendation: {final.get('final_recommendation')}")
    print(f"  cost: ${final.get('total_cost_usd', 0):.4f}")
    print(f"  wall: {elapsed:.1f}s")
    print(f"  → {json_path}")
    print(f"  → {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
