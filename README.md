# Screening Tool — AI Recruiter Assistant

> **Status:** Live (2026-04-29). MVP complete: RAG, LangGraph agent, FastAPI, eval suite, deployed on HF Spaces, full Langfuse instrumentation.
> Part of [ai-portfolio](https://github.com/ALchemt) — Project 3 of 4.

**Live API:** https://alchemt-screening-tool.hf.space — interactive docs at [`/docs`](https://alchemt-screening-tool.hf.space/docs)

B2B screening tool for technical recruiters. Loads a job description, runs a structured candidate interview through a LangGraph agent loop, evaluates answers with an LLM-as-judge against a rubric, and produces a hiring report. Full observability via Langfuse. REST API via FastAPI.

## Project Goals

1. Demonstrate **multi-agent orchestration** via LangGraph (state machine, not raw loop) — 4 nodes, conditional edges, interrupts for interactive flow
2. Demonstrate **LLM observability** with Langfuse — full execution traces, cost tracking, latency p50/p95
3. Demonstrate **production API** via FastAPI with persistent session state (LangGraph SqliteSaver, thread_id == session_id)
4. Demonstrate **honest evaluation** — 32 sessions × 5 metrics with documented limitations, not just happy-path demo

## Tech Stack

| Layer | Choice |
|---|---|
| Agent loop | LangGraph 0.2+ |
| Observability | Langfuse (self-hosted, docker-compose on VPS) |
| API | FastAPI + uvicorn |
| Vector store | ChromaDB (embedded) |
| LLM (agent) | `openai/gpt-4o-mini` via OpenRouter |
| LLM (judge) | `anthropic/claude-sonnet-4.5` via OpenRouter (split-judge: different family) |
| Embeddings | `text-embedding-3-small` via OpenRouter |
| Deploy | HuggingFace Spaces (Docker SDK) |

Full architecture and rationale in [spec.md](./spec.md).

## Status

- [x] Scaffold + spec
- [x] JD corpus (8 synthetic JDs)
- [x] RAG ingestion + retrieval (47 chunks, ChromaDB, top-1 retrieval validated on 7 queries)
- [x] LangGraph agent + judge (4 nodes, split-judge: gpt-4o-mini agent + claude-sonnet-4.5 judge)
- [x] FastAPI endpoints (6 endpoints, SqliteSaver persistence, end-to-end smoke test passing)
- [x] Eval suite (32 sessions, 5 metrics — see [Eval Results](#eval-results) below)
- [x] Langfuse instrumentation (cloud free tier — self-host VPS deferred, infra blueprint in [`infra/`](./infra/))
- [x] HF Spaces deploy — [live](https://alchemt-screening-tool.hf.space)
- [x] Public GitHub repo — [github.com/ALchemt/screening-tool](https://github.com/ALchemt/screening-tool)

## Eval Methodology

**Test set:** 4 JDs × 8 candidate variations = 32 sessions.

Variations cover the realistic spectrum: strong / medium / weak / edge (lying, off-topic, asking back).

**5 metrics:**

| Metric | Target | How |
|---|---|---|
| Question relevance to JD | mean ≥ 1.7/2 | Independent judge scores each question against JD requirements |
| Judge-vs-human agreement | ≥ 75% match | Manual labels on 10 sessions, agreement vs LLM judge |
| Hallucination rate | < 5% | Independent judge: "any claims in report not present in responses?" |
| Cost per session | < $0.10 | Langfuse cost tracking |
| Latency p95 | < 60s | Langfuse traces |

## Eval Results

Run on 2026-04-28. Full breakdown in [`runs/eval_summary.md`](./runs/eval_summary.md).

| Metric | Target | Result | Pass |
|---|---|---|---|
| Question relevance (judge #2 vs JD) | ≥ 1.7/2 | **1.83/2** | ✅ |
| Cost per session | < $0.10 | **$0.0199** | ✅ |
| Latency p50 / p95 | p95 < 60s | **42s / 50s** | ✅ |
| Hallucination rate | < 5% | **~20-25% (real) / 48% (raw judge)** | ❌ |
| Judge–human agreement | ≥ 75% | _pending labelling_ | — |

**Recommendation distribution across 32 sessions:**

| Persona | hire | uncertain | no_hire |
|---|---|---|---|
| strong (8) | 6 | 2 | 0 |
| medium (8) | 3 | 1 | 4 |
| weak (8) | 0 | 0 | 8 |
| edge (8) | 1 | 0 | 7 |

**What works:** generated questions are JD-targeted (1.83/2 relevance), 100% precision on weak personas, 0 false negatives on strong personas, $0.02/session economics, p95 < 60s.

**What doesn't (yet):** ~20-25% of factual claims in screening reports cannot be traced back to the candidate's answers — the agent embellishes (invents tools, durations, metrics, employer names). 1 false positive in 8 edge-case "confident liar" trials. Mitigation candidates listed in [eval_summary.md](./runs/eval_summary.md#mitigation-candidates-for-future-iteration-not-this-mvp).

## Run locally

```bash
git clone https://github.com/ALchemt/screening-tool
cd screening-tool
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill OPENROUTER_API_KEY and LANGFUSE_*
python scripts/build_index.py  # ingest data/jd_corpus/ → ChromaDB
uvicorn src.api:app --reload
```

API docs: http://localhost:8000/docs

## API Reference

```
GET  /healthz                  → liveness
GET  /jds                      → list 8 JDs (id + title + seniority + company)
POST /screening/start          → create session, get first question
POST /screening/respond        → submit answer, get next question or final report
GET  /screening/{id}/report    → final report + recommendation + cost
GET  /screening/{id}/state     → session state inspection
```

### End-to-end curl example

```bash
BASE=https://alchemt-screening-tool.hf.space

# 1. List available JDs
curl -s $BASE/jds | jq '.[].jd_id'

# 2. Start a session for "Agentic AI Developer"
SID=$(curl -s -X POST $BASE/screening/start \
  -H "Content-Type: application/json" \
  -d '{"jd_id":"jd_005","max_questions":3}' | jq -r .session_id)

# 3. Answer the first question (response shown in start payload)
curl -s -X POST $BASE/screening/respond \
  -H "Content-Type: application/json" \
  -d "{\"session_id\":\"$SID\",\"response\":\"I built a 3-agent LangGraph...\"}"

# 4. Continue until "done": true, then fetch the report
curl -s $BASE/screening/$SID/report | jq .recommendation
```

## Limitations

- **Hallucination in reports (~20-25%)** — main known issue, see [eval_summary.md](./runs/eval_summary.md). Mitigation roadmap documented; not yet shipped.
- **Synthetic JDs and personas** — eval measures agreement with another LLM, not real human candidates
- **Small JD corpus (8 documents)** — generalisation to non-AI/automation roles unknown
- **No UI** — REST only (interactive `/docs` from FastAPI)
- **Single-tenant** — no auth, no user management
- **Cloud-tier Langfuse** — self-host roadmap exists ([`infra/`](./infra/)) but not deployed

## Why this project

Reframe of "Interview Coach for candidate" → B2B screening tool for recruiter (decision 2026-04-25). Targets the automation/agentic JD layer (Mindrift, Newxel, Pearl Talent, Upwork) where current portfolio (P1 RAG, P2 Eval) is already strong but missing three signals: multi-agent (LangGraph), observability (Langfuse), and FastAPI. P3 adds all three.

## Sibling projects

- [P1 RAG Document Q&A](https://alchemt-rag-qa.hf.space/) — grounded Q&A with eval methodology
- [P2 LLM Eval Framework](https://alchemt-llm-eval.hf.space/) — regression eval runner
- P3 Screening Tool — **this**
- P4 Policy/Compliance Q&A Engine — TBD
