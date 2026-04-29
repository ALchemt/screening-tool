# P3 Eval v2 — Anti-hallucination iteration (2026-04-29)

After v1 (2026-04-28) flagged ~20-25% real fabrication rate in screening reports, I tightened the report-generation prompt with explicit grounding rules and lowered temperature to 0.1. Then re-ran the same 32-session eval matrix.

## Comparison

| Metric | v1 (baseline) | v2 (anti-halluc) | Δ |
|---|---|---|---|
| Question relevance mean | 1.83 / 2 | 1.78 / 2 | -0.05 (still ≥ 1.7) |
| Cost per session | $0.0199 | $0.0199 | flat |
| Latency p95 | 50.1s | 47.7s | -2.4s |
| **Hallucination rate (raw, judge #3)** | **48.1%** | **48.1%** | **flat** |
| **Hallucination rate (manual review)** | **~20-25%** | **~0-5%** | **massive drop** |

## What manual review revealed

I sampled the top 5 v2 sessions by hallucination count and read every flagged claim. Examples:

| Flagged claim (v2) | Verdict |
|---|---|
| "lacks depth in n8n architecture details" | concern statement, not a fabrication |
| "Did not address specific execution modes, scaling, deployment context" | absence-of-evidence note, valid |
| "frequently deflected questions back to the interviewer" | accurate observation about evasive persona |
| "My workflows handled over 500 req/s" | actual quote from strong persona's answer |

In v1, the same audit showed ~half of flagged claims were real fabrications:
- "Langfuse for n8n monitoring" (candidate never said this)
- "18+ months of dedicated experience" (invented duration)
- "DataVision startup" (invented employer)
- "94% accuracy on MNIST" attributed to evasive persona who said no such thing

**v2 has effectively zero of these.** The grounding rules + lower temperature stopped the agent from inventing tools, durations, and employer names.

The judge can't tell the difference because its prompt counts any claim not directly quotable as "hallucinated", which conflates:
- Fabricated facts (real problem in v1)
- Concern statements about absence ("did not mention X" — valid recruiter language)
- Paraphrased quotes (loose paraphrasing of what candidate actually said)

## Recommendation distribution shift

v1 → v2:

| Persona level | v1 hire / uncertain / no_hire | v2 strong_hire / hire / uncertain / no_hire |
|---|---|---|
| strong (8) | 6 / 2 / 0 | 2 / 3 / 2 / 1 |
| medium (8) | 3 / 1 / 4 | 0 / 0 / 0 / 8 |
| weak (8) | 0 / 0 / 8 | 0 / 0 / 0 / 8 |
| edge (8) | 1 / 0 / 7 | 0 / 0 / 1 / 7 |

**Reads:**
- ✅ Strong: now uses full hire spectrum (2 strong_hire vs 0 in v1) — agent more decisive on best candidates
- ✅ Edge: 1 false positive eliminated (no edge_lying got "hire")
- ⚠️ Strong: 1 false negative new (jd_007/strong_a got no_hire — anti-halluc may have refused to back vague-but-strong claims)
- ❌ Medium: regression. v1 had 3/8 hires for medium personas; v2 has 0/8. Anti-halluc rules made the agent unable to hedge positively on partial matches. This is a real cost of the fix.

## Net assessment

**Anti-halluc fix is a positive trade overall** but with one regression:
- Real fabrications: dropped from ~20-25% to near-zero ✅
- Edge cases: 1 false positive eliminated ✅
- Strong: distribution improved (uses strong_hire) ✅
- Medium: 3 hire → 0 hire, agent over-conservative on partial matches ❌

**Next iteration would address:**
1. Tune the report prompt to allow conditional hire ("hire with reservations") instead of forcing no_hire when concerns exist
2. Rewrite the hallucination judge to distinguish fabrication from concern statement (would also drop the 48% raw metric to a real number — likely <10%)
3. Collect human labels on 10 sessions from each version, compute judge-vs-human agreement properly

## What this iteration teaches

The bottleneck moved. v1's bottleneck was the agent (it was making things up). v2's bottleneck is the eval framework itself (the judge can't tell over-strict from over-loose reports apart from fabrications).

This is the kind of finding that only shows up if you actually run an eval, look at the numbers AND read the failure cases. If I had only watched the headline `hallucination_rate_pct`, I'd report v2 as "fix didn't work" — false. The number didn't move because I was measuring the wrong thing.

---

## v3 — judge prompt rewrite (same day, 2026-04-29)

To get the headline metric to actually mean something, I rewrote `JUDGE_HALLUC_SYSTEM` (`src/judge.py`) to classify each report claim into one of three buckets:

- **FABRICATED** — positive factual claim with no support in the candidate's answers (the only thing that should count)
- **CONCERN** — honest absence note ("did not address X", "lacks evidence of Y") — valid recruiter language
- **JUDGEMENT** — generic recruiter assessment ("strong communication", "surface-level")

Re-judged the same 32 v2 sessions (no candidate regeneration, just the new judge). Cost: $0.25.

Results (`runs/eval_v2_v3_summary.json`):

| | v2 raw judge | v3 separated judge |
|---|---|---|
| Fabrications counted | 47 | 6 |
| **Fabrication rate** | **29.4%** | **3.75%** |
| Concerns separated out | (none — folded into halluc count) | 127 |

Manual audit of all 6 v3-flagged fabrications:

| Session | Flagged | Verdict |
|---|---|---|
| jd_005/weak_b | "developed a chatbot... 30% reduction in response times" | **real fabrication** — agent turned hypothetical example into stated fact, invented metric |
| jd_005/edge_evasive | direct quotes about TensorFlow / agent roles | **judge over-flag** — quotes appear verbatim in answers, judge admits this in reasoning but still classified as fab |
| jd_007/weak_b | "solid understanding of building applications in Python" | **borderline** — positive spin on candidate's actual answer ("haven't worked on REST APIs"), not invented from scratch |

**Real fabrication rate after v3 audit: ~1.25% (2 / 160 questions).**

Honest takeaway: v3 metric (3.75%) is much closer to truth than v2 (29.4%), but the judge still over-flags direct-quote framing and positive-spin restating. A purer eval would need a 4-bucket classifier (fab / concern / judgement / spin) — leaving as known limitation rather than over-engineering.

This is what landed in the README as the production metric.

---

## Judge–human alignment (2026-04-29)

10 random (session, exchange) pairs from the v2 run, scored manually on the same 0-2 rubric, agreement vs the LLM judge:

| Axis | Exact match | Mean abs diff |
|---|---|---|
| technical_accuracy | 100.0% | 0.00 |
| depth | 90.0% | 0.10 |
| communication | 90.0% | 0.10 |
| red_flags | 100.0% | 0.00 |
| **Overall** | **95.0%** | **0.05** |

Read: judge is exact on hard binary calls (TA, RF) and off-by-one on subjective calls (D, C). No disagreement was off-by-two. n=10 is small (CI ≈ ±15pp) so the headline number is "not visibly miscalibrated" rather than proof of correctness. Raw labels in `runs/human_labels.jsonl`.
