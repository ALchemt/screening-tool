"""Generate 8 synthetic job descriptions via OpenRouter (claude-sonnet-4.5).

One batch request, parses JSON array, writes .md files with frontmatter to
data/jd_corpus/. Costs ~$0.10-0.20 on Sonnet 4.5.

Roles cover automation / agentic / AI engineering JD layer (target market
per memory/project_ai_portfolio.md and 2026-04-25 JD analysis).

Usage:
    python scripts/gen_jd_corpus.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

ROLES = [
    {"id": "jd_001", "title": "AI Automation Engineer", "seniority": "junior",
     "focus": "n8n workflows + LLM API integration for SMB ops automation"},
    {"id": "jd_002", "title": "n8n Developer", "seniority": "mid",
     "focus": "self-hosted n8n, custom nodes, integrations with CRM/ERP"},
    {"id": "jd_003", "title": "AI Engineer (RAG)", "seniority": "junior",
     "focus": "LangChain/LlamaIndex, vector DBs, document QA pipelines"},
    {"id": "jd_004", "title": "Voice AI Engineer", "seniority": "mid",
     "focus": "Twilio + LLM conversational agents, IVR replacement, Vapi/Retell"},
    {"id": "jd_005", "title": "Agentic AI Developer", "seniority": "junior",
     "focus": "LangGraph, multi-agent orchestration, tool calling, MCP"},
    {"id": "jd_006", "title": "AI Integration Engineer", "seniority": "mid",
     "focus": "REST APIs, webhooks, OpenAI/Anthropic integration into existing systems"},
    {"id": "jd_007", "title": "LLM Application Developer", "seniority": "junior",
     "focus": "FastAPI, prompt engineering, RAG, basic eval, deploy to cloud"},
    {"id": "jd_008", "title": "AI Solutions Engineer", "seniority": "mid",
     "focus": "customer-facing, scoping AI automation projects, hybrid IC + consulting"},
]

SYSTEM_PROMPT = """You are a senior technical recruiter writing realistic job descriptions
for an AI/automation-focused startup. Style mirrors top postings on Remotive, Himalayas,
and Wellfound: concise, specific, no corporate fluff, no emoji.

Each JD must include:
- About the Company (2-3 sentences, fictional but plausible startup)
- Role (1 paragraph)
- Responsibilities (5-7 bullets, action verbs, specific)
- Required Qualifications (5-7 bullets, technical specifics — versions, tools, years)
- Nice-to-Have (3-5 bullets)
- What We Offer (3-4 bullets — remote, comp range, learning budget, etc.)

Use real tool names (n8n, LangChain, LangGraph, Pinecone, ChromaDB, FastAPI, Twilio, etc.)
where appropriate. Compensation: ranges in USD reflecting 2026 market for the seniority."""

USER_TEMPLATE = """Generate 8 distinct synthetic job descriptions, one per role below.
Companies must be fictional but plausible (different industries: HR tech, fintech, e-commerce
ops, healthcare ops, legal tech, real estate tech, customer support SaaS, dev tools).

Roles:
{roles_json}

Return a JSON array of 8 objects, each with these exact keys:
- "id": the jd_id from input
- "title": the role title (you may add company-specific suffix like "II" or "/ Specialist")
- "seniority": from input
- "company_name": fictional
- "company_blurb": 2-3 sentence company description
- "body_md": full JD markdown body starting with `## Role` (no h1, no frontmatter)

Output ONLY the raw JSON array, no markdown fences, no commentary."""


def main() -> int:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set", file=sys.stderr)
        return 1

    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    print(f"[gen] requesting 8 JDs from claude-sonnet-4.5 via OpenRouter...")
    resp = client.chat.completions.create(
        model="anthropic/claude-sonnet-4.5",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(
                roles_json=json.dumps(ROLES, indent=2))},
        ],
        max_tokens=12000,
        temperature=0.8,
    )

    content = resp.choices[0].message.content.strip()
    if content.startswith("```"):
        content = content.split("```", 2)[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.rsplit("```", 1)[0].strip()

    try:
        jds = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"ERROR: failed to parse JSON: {e}", file=sys.stderr)
        debug_path = ROOT / "data" / "jd_corpus" / "_raw_response.txt"
        debug_path.write_text(content)
        print(f"raw response saved to {debug_path}", file=sys.stderr)
        return 1

    out_dir = ROOT / "data" / "jd_corpus"
    out_dir.mkdir(parents=True, exist_ok=True)

    for jd in jds:
        fm = (
            "---\n"
            f"id: {jd['id']}\n"
            f"title: {jd['title']}\n"
            f"seniority: {jd['seniority']}\n"
            f"company: {jd['company_name']}\n"
            "source: synthetic (claude-sonnet-4.5, 2026-04-28)\n"
            "license: portfolio-use\n"
            "---\n\n"
            f"# {jd['title']} @ {jd['company_name']}\n\n"
            f"## About {jd['company_name']}\n\n{jd['company_blurb']}\n\n"
            f"{jd['body_md']}\n"
        )
        out_path = out_dir / f"{jd['id']}.md"
        out_path.write_text(fm)
        print(f"  wrote {out_path.name} ({len(fm)} chars)")

    usage = resp.usage
    print(f"[gen] done. tokens: {usage.prompt_tokens} in / {usage.completion_tokens} out")
    return 0


if __name__ == "__main__":
    sys.exit(main())
