# P3 Screening Tool — Eval Results (2026-04-28)

**Test set:** 32 sessions = 4 JDs × 8 candidate personas × 5 questions each.
**Total questions evaluated:** 160. **Total cost:** $1.68 ($0.64 sessions + $1.05 eval judges).
**Wall time:** 23 min (sessions) + 19 min (metrics).

JDs used: `jd_002` (n8n Developer II), `jd_003` (AI Engineer RAG), `jd_005` (Agentic AI Developer), `jd_007` (LLM Application Developer).

Personas (2 per level):
- **strong** — full match, deep technical, real evidence
- **medium** — partial match, acknowledged gaps
- **weak** — vague generalist + off-topic rambler
- **edge** — confident liar + asks-back/evasive

## Headline metrics

| Metric | Target | Result | Pass |
|---|---|---|---|
| Question relevance mean (judge #2) | ≥ 1.7 / 2 | **1.83** | ✅ |
| Cost per session mean | < $0.10 | **$0.0199** | ✅ |
| Latency p50 / p95 | p95 < 60s | **42s / 50s** | ✅ |
| Hallucination rate (claims / question) | < 5% | **~48%** raw / ~20-25% adjusted | ❌ |
| Judge–human agreement | ≥ 75% | _pending_ — labelling CLI ready, run after manual pass | — |

## Recommendation distribution

| Persona level | hire | uncertain | no_hire | n |
|---|---|---|---|---|
| strong | 6 | 2 | 0 | 8 |
| medium | 3 | 1 | 4 | 8 |
| weak | 0 | 0 | 8 | 8 |
| edge | 1 | 0 | 7 | 8 |

**Reads:**
- Weak personas → 100% no_hire. Judge reliably catches buzzword answers.
- Strong personas → 0 false negatives (no strong got `no_hire`).
- Edge: 1 false positive — `edge_lying` on `jd_003` got `hire` despite fabricating projects. Expected failure mode for confident-liar archetype on a JD where the liar's invented story aligned with required skills.
- Medium: 50/50 split — the right call for mid-level/partial-match candidates.

## Hallucination findings

Judge #3 flagged 77 fabricated claims across 32 reports (mean 2.5 per session).
Manual review of 18 sampled claims:
- ~9 are **real fabrications** the agent invented (e.g. "Langfuse for n8n monitoring", "18+ months experience", invented company "DataVision", inflated "94% MNIST accuracy" attributed to evasive candidate who didn't mention any of it)
- ~6 are **judge over-flags**: the judge counts concern statements ("lacks specifics", "doesn't mention webhook alternatives") as hallucinations even though the prompt instructs to ignore generic recruiter judgements
- ~3 are ambiguous (paraphrasing tightening/loosening evidence)

**Honest read:** real fabrication rate is ~20-25%, not 48%. Still above the <5% target.

**Root cause:** the report-generation prompt (`REPORT_SYSTEM` in `src/agent.py`) tells the LLM to "cite specific candidate quotes where useful" but does not constrain it to ONLY make claims grounded in the transcript. With temperature=0.3 and gpt-4o-mini, the model occasionally embellishes.

**Mitigation candidates (for future iteration, not this MVP):**
1. Lower agent temperature to 0.1 for report node only
2. Add a self-check pass: after report drafted, run a second LLM call asking "for each factual claim about the candidate in this report, quote the source line from the transcript or remove it"
3. Switch report node from gpt-4o-mini to claude-haiku-4.5 — the family that already serves as judge — and benchmark whether grounding improves
4. Add explicit anti-hallucination instructions in REPORT_SYSTEM ("do not invent metrics, dates, employer names, or experience durations not stated by the candidate")

## What this eval proves about the screening tool

**Strong signal — usable:**
- Relevance of generated questions (1.83/2): the agent produces JD-targeted screening questions, not generic ones.
- Cost ($0.02/session): viable for high-volume use; recruiter could screen hundreds of candidates for ~$2-3/run.
- Latency (p95 50s): real-time interactive screening is feasible.
- Recommendation accuracy on extreme cases (weak: 100% precision, strong: 100% recall): judge is calibrated for clear signals.

**Weak signal — needs work:**
- Report grounding. ~20-25% of factual claims about the candidate cannot be traced to their answers. This is the #1 blocker before showing this to a real recruiter.
- Edge case: `edge_lying` produces 1 false positive in 8 trials (12.5%). Resume verification (LinkedIn lookup, reference check) would still be required downstream — the agent is a screener, not a fact-checker.
- Medium personas: mixed. Real recruiter would want more nuance than `hire / uncertain / no_hire` for the borderline cases.

## Honest limitations

- 4 JDs is small. Generalisation to non-AI/automation roles unknown.
- Personas are themselves LLM-generated — the eval measures whether the screening agent agrees with another LLM about candidate quality. Real human candidates drift outside personas.
- Judge–human agreement metric not yet collected (deferred to manual labelling pass).
- Single seed, single model snapshot per role. No variance bands.

## Files

- `runs/eval_20260428_2011.jsonl` — raw 32 sessions
- `runs/eval_20260428_2011_metrics.jsonl` — sessions enriched with relevance/halluc scores
- `runs/eval_20260428_2011_summary.json` — machine-readable headline numbers
- `runs/human_labels.jsonl` — _pending_, populated by `scripts/label_human.py`

Reproduce: `python scripts/run_eval.py` then `python -m src.eval_metrics runs/eval_<ts>.jsonl`.
