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
