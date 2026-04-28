# P3 Screening Tool — Spec

> Статус: scaffold. Старт: 2026-04-28. Дедлайн: ~2026-05-12.
> План одобрен Андреем 2026-04-27. Источник: deep-plan сессия + ответы на open questions.

## Что это

B2B-инструмент для рекрутера. Recruiter передаёт JD → LangGraph-агент проводит structured screening кандидата (адаптивные вопросы под role) → LLM-as-judge оценивает ответы по rubric → итоговый screening report с обоснованием. Доступ через FastAPI, полный observability через Langfuse.

**Не "Interview Coach для кандидата"** — это B2B-угол вместо C2C.

## Зачем (рыночный сигнал)

Покрывает 3 высоких сигнала JD-2026 одновременно, которых нет в P1/P2:

- **multi-agent / LangGraph** — #1 быстрорастущий сигнал
- **observability / Langfuse** — 8.5% JD, единичные junior портфолио с tracing
- **FastAPI** — 17% JD, base layer для production AI

Источник анализа: `.claude/agents-comms/archive/2026-04/2026-04-25-job-hunter-jd-analysis.md` + `2026-04-25-feynman-portfolio-research.md`.

## Архитектура

```
                    ┌─────────────────────────────────────┐
   POST /start ───→ │  FastAPI                            │
   POST /respond ─→ │  (Pydantic + uvicorn)               │
   GET /report ───→ └────────────────┬────────────────────┘
                                     │
                                     ▼
                    ┌─────────────────────────────────────┐
                    │  LangGraph state machine            │
                    │                                     │
                    │  load_jd → generate_questions ──┐   │
                    │     ↑                           │   │
                    │     │                           ▼   │
                    │  evaluate_response ←── ask_question │
                    │     │                               │
                    │     ▼ (judge node)                  │
                    │  continue? ──→ final_report        │
                    └────────────────┬────────────────────┘
                                     │
                ┌────────────────────┼────────────────────┐
                ▼                    ▼                    ▼
         ChromaDB (JD)        OpenRouter LLMs        Langfuse
         embedded             (agent + judge)        (self-hosted, VPS)
```

## Tech stack — обоснование

| Компонент | Выбор | Почему |
|---|---|---|
| **Agent framework** | LangGraph 0.2+ | #1 сигнал JD-2026 (multi-agent). State machine явная, легче трейсить. LangChain agents deprecated в пользу LangGraph. |
| **Observability** | Langfuse self-hosted (VPS) | Open-source, vendor-neutral. Self-hosted = bonus сигнал "знаю как развернуть LLM observability infra" (Mindrift JD это любит). LangSmith — locked в LangChain ecosystem. |
| **API layer** | FastAPI | 17% JD упоминают (high signal). Streamlit в P2 уже есть, повтор = нет нового сигнала. |
| **Vector store** | ChromaDB embedded | Уже использовали в P1, знаем quirks. Для 8 JD embedded в файл идеален. |
| **LLM — agent** | `openai/gpt-4o-mini` через OpenRouter | Дёшево, fast tool calling. OR проксирует — работает с HK картой. |
| **LLM — judge** | `anthropic/claude-sonnet-4.5` через OpenRouter | Split-judge: agent ≠ judge family (урок из P1, избежание self-bias). Через OR роутится на AWS Bedrock — без geo-issues. |
| **Embeddings** | `text-embedding-3-small` через OR | Дёшево ($0.02/1M), 1536 dim. |
| **Deploy** | HF Spaces (Docker SDK) | Бесплатно, аккаунт есть. Поддерживает FastAPI через Dockerfile. |
| **JD corpus** | 8 синтетических (Sonnet generation) | Решено: синтетика чтобы избежать любых вопросов копирайта. Стиль — реальные Remotive/Himalayas posting'и. |

## План этапов

| # | Этап | ETA | Критерий завершения |
|---|---|---|---|
| 1 | Scaffold + JD corpus (8 synthetic JDs) | 2-3ч | `pip install` проходит, 8 JD .md в `data/jd_corpus/` с frontmatter |
| 2 | JD ingestion + RAG retrieval | 3-4ч | Smoke test: query → top-5 chunks с метаданными |
| 3 | LangGraph agent loop + judge | 6-8ч | End-to-end CLI run: JD → questions → responses → report. Langfuse traces видны |
| 4 | FastAPI endpoint | 2-3ч | 3 endpoints, curl flow за <30с, SqliteSaver persistence |
| 5 | Eval suite | 4-5ч | 32 sessions прогнаны, 5 метрик собраны, human labels проставлены |
| 6 | Langfuse self-host on VPS + Deploy HF | 3-4ч | Live URL отвечает, traces идут в self-hosted Langfuse |
| 7 | Buffer / polish (README, GIF, LinkedIn post) | 3-5ч | Public GitHub, GIF demo, README с честными limitations |

**Total ETA:** 23-32ч → влезает в 2 недели × 10-15ч.

## Eval-план (главный дифференциатор)

**Test set:** 4 JD × 8 candidate response variations = **32 sessions**.

Variations кандидата (синтетика через Sonnet):
- 2× **strong** (full match, deep technical)
- 2× **medium** (partial match, gaps)
- 2× **weak** (off-topic, vague)
- 2× **edge** (lying/hallucinating, asking back questions)

**5 метрик:**

| Метрика | Цель | Как считаем |
|---|---|---|
| Question relevance to JD | mean ≥ 1.7/2 | Отдельный judge №2 оценивает каждый вопрос на соответствие JD requirements |
| Judge-vs-human agreement | ≥ 75% match | Андрей вручную ставит human_score 0/1/2 на 10 sessions, % совпадений с judge |
| Hallucination rate в report | < 5% | Judge №3: "содержит ли report claims, которых нет в responses?" |
| Cost per session | < $0.10 | Langfuse cost tracking (sum tokens × prices) |
| Latency p50/p95 | p95 < 60s | Langfuse автоматически |

## Risks + fallbacks

| Risk | Митигация / fallback |
|---|---|
| LangGraph кривая обучения | 1ч на quickstart перед стартом, держать граф плоским (без subgraphs) |
| Langfuse self-host на VPS падает | Fallback на Langfuse Cloud free tier (50k events/мес). Конфиг через `LANGFUSE_HOST` env var |
| FastAPI на HF Spaces глюки | Fallback на Railway free tier ($5 credit) |
| Judge bias (self-eval) | Split-judge: gpt-4o-mini agent + claude-sonnet-4.5 judge (разные families) |
| OR провайдер для конкретной модели падает | OR auto-routing на запасного провайдера; в крайнем — переключение на gemini-2.5-flash + qwen3-max |
| Малый JD corpus → плохой retrieval | Hybrid retrieval (BM25 + semantic) через rank-bm25 |
| State management между API запросами | LangGraph SqliteSaver checkpointer (built-in) |

## Критерии готовности

**MVP (минимум "зачёт", к ~2026-05-05):**
- [ ] LangGraph граф работает end-to-end локально
- [ ] FastAPI 3 endpoints отвечают
- [ ] Langfuse traces видны
- [ ] 1 пример session в `runs/` с screening report
- [ ] README с базовыми разделами

**Portfolio piece (полная цель, к 2026-05-12):**
- [ ] Live URL на HF Spaces, публично доступен
- [ ] README с реальными eval метриками (5 метрик, не "TBD")
- [ ] GIF demo
- [ ] Honest limitations секция
- [ ] Human labels проставлены, agreement посчитан
- [ ] Public GitHub repo (mirror как у P2)
- [ ] LinkedIn пост опубликован
- [ ] Self-hosted Langfuse работает на VPS
- [ ] Добавлен в `vault/01-profile/` showcase секцию

## Что переиспользуется из P1/P2

- LLM judge паттерн → `ai-portfolio/llm-eval/src/judges/`
- Provider-agnostic OpenAI SDK pattern (OR routing) → `ai-portfolio/rag-qa/src/generator.py`
- Eval методология (split-judge, honest limitations) → README обоих

## Что НЕ делаем в P3

- Аутентификация / пользователи / multi-tenant
- UI (только REST + curl/Postman примеры в README)
- Streaming responses
- Real candidate testing (только synthetic responses)

## Точки контроля

1. **После Этапа 2** — показать smoke test retrieval, согласовать качество синтетических JD
2. **После Этапа 3** — показать первый screening run end-to-end, согласовать формат screening report
3. **После Этапа 5** — показать eval results, решить deploy as-is или ещё итерация
4. **Перед публикацией LinkedIn поста** — Андрей читает финальный README

## Коммит scaffold

`P3 Screening Tool: spec + scaffold` — создание структуры, requirements, .env.example, пустых модулей, этого spec. Без логики. Без push.
