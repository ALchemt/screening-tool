"""Interactive CLI to collect human labels on judge scores.

Picks N random (session, exchange) pairs from an eval jsonl, shows
question + candidate answer + judge's score + reasoning, asks Andrey
to score each axis (technical_accuracy, depth, communication, red_flags)
on the same 0/1/2 scale. Writes labels to runs/human_labels.jsonl
(append mode — safe to resume).

Then computes judge-vs-human agreement: % exact match per axis.

Usage:
    python scripts/label_human.py runs/eval_X.jsonl --n 10
    python scripts/label_human.py runs/eval_X.jsonl --report   # only show agreement
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LABELS_FILE = ROOT / "runs" / "human_labels.jsonl"
AXES = ["technical_accuracy", "depth", "communication", "red_flags"]


def _ask_score(axis: str, judge_val: int) -> int:
    while True:
        s = input(f"  your {axis} (0/1/2) [judge={judge_val}, "
                  f"Enter=agree]: ").strip()
        if s == "":
            return judge_val
        if s in ("0", "1", "2"):
            return int(s)
        print("    enter 0, 1, 2 or Enter")


def _show_session(row: dict, idx: int) -> dict | None:
    """Show one exchange and capture human scores. Returns label dict."""
    ex = row["exchanges"][idx]
    j = ex["judge"]
    print("\n" + "=" * 72)
    print(f"Session: {row['session_id']}  ({row['persona_level']} persona)")
    print(f"Role:    {row['jd_title']} ({row['jd_seniority']})")
    print(f"Exchange #{idx+1}/{len(row['exchanges'])}")
    print("-" * 72)
    print(f"Q: {ex['question']}")
    print(f"\nA: {ex['response']}")
    print(f"\nJudge: TA={j['technical_accuracy']} D={j['depth']} "
          f"C={j['communication']} RF={j['red_flags']}")
    print(f"Judge reason: {j['reasoning']}")
    print("-" * 72)
    print("Score each axis (Enter to agree with judge, 's' to skip session):")

    first = input(f"  your technical_accuracy (0/1/2) [judge="
                  f"{j['technical_accuracy']}, Enter=agree, s=skip]: ").strip()
    if first.lower() == "s":
        return None
    if first == "":
        ta = j["technical_accuracy"]
    elif first in ("0", "1", "2"):
        ta = int(first)
    else:
        print("invalid, defaulting to judge")
        ta = j["technical_accuracy"]

    d = _ask_score("depth", j["depth"])
    c = _ask_score("communication", j["communication"])
    rf = _ask_score("red_flags", j["red_flags"])

    return {
        "session_id": row["session_id"],
        "exchange_index": idx,
        "persona_level": row["persona_level"],
        "judge": {ax: j[ax] for ax in AXES},
        "human": {"technical_accuracy": ta, "depth": d,
                  "communication": c, "red_flags": rf},
    }


def label(jsonl_path: Path, n: int, seed: int = 42):
    rows = [json.loads(l) for l in jsonl_path.read_text().splitlines() if l.strip()]
    rng = random.Random(seed)
    pairs = [(r, i) for r in rows for i in range(len(r["exchanges"]))]
    rng.shuffle(pairs)

    LABELS_FILE.parent.mkdir(exist_ok=True)
    already = set()
    if LABELS_FILE.exists():
        for line in LABELS_FILE.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                already.add((d["session_id"], d["exchange_index"]))
        print(f"Loaded {len(already)} existing labels — will skip those.")

    targets = [p for p in pairs
               if (p[0]["session_id"], p[1]) not in already][:n]
    print(f"Will label {len(targets)} new exchanges.")

    with LABELS_FILE.open("a") as fh:
        for k, (row, idx) in enumerate(targets, 1):
            print(f"\n[{k}/{len(targets)}]", end="")
            try:
                label_row = _show_session(row, idx)
            except (KeyboardInterrupt, EOFError):
                print("\nInterrupted — saving what we have.")
                break
            if label_row is None:
                continue
            fh.write(json.dumps(label_row, ensure_ascii=False) + "\n")
            fh.flush()
    print(f"\nLabels saved to {LABELS_FILE}")


def report():
    if not LABELS_FILE.exists():
        print("No labels yet. Run without --report first.")
        return
    labels = [json.loads(l) for l in LABELS_FILE.read_text().splitlines() if l.strip()]
    if not labels:
        print("Empty labels file.")
        return
    n = len(labels)
    per_axis_match = {ax: 0 for ax in AXES}
    per_axis_diff_sum = {ax: 0 for ax in AXES}
    for lab in labels:
        for ax in AXES:
            if lab["judge"][ax] == lab["human"][ax]:
                per_axis_match[ax] += 1
            per_axis_diff_sum[ax] += abs(
                lab["judge"][ax] - lab["human"][ax])

    print(f"\nJudge–human agreement on {n} labelled exchanges:")
    print(f"{'axis':<22} {'exact_match':>12} {'mean_abs_diff':>14}")
    for ax in AXES:
        match_pct = 100 * per_axis_match[ax] / n
        mad = per_axis_diff_sum[ax] / n
        print(f"{ax:<22} {match_pct:>11.1f}% {mad:>14.2f}")

    overall = sum(per_axis_match.values()) / (n * len(AXES))
    print(f"\nOverall exact-match agreement: {100 * overall:.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", type=Path, nargs="?",
                    help="path to eval jsonl (omit if using --report)")
    ap.add_argument("--n", type=int, default=10,
                    help="number of exchanges to label")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report", action="store_true",
                    help="print agreement stats from existing labels")
    args = ap.parse_args()

    if args.report:
        report()
        return
    if not args.jsonl:
        ap.error("jsonl required unless --report")
    label(args.jsonl, args.n, args.seed)
    report()


if __name__ == "__main__":
    main()
