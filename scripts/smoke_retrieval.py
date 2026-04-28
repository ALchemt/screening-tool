"""Smoke test for retrieval quality.

Runs a few representative queries and prints top-5 chunks to stdout.
Used as visual sanity-check at end of Stage 2.

    python scripts/smoke_retrieval.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))

from src.retrieval import retrieve  # noqa: E402

QUERIES = [
    "What Python and LLM framework experience is required?",
    "experience with n8n or workflow automation tools",
    "vector databases like Pinecone or ChromaDB for RAG",
    "Twilio voice integration and conversational AI",
    "LangGraph multi-agent orchestration",
    "FastAPI REST endpoint deployment",
    "what does the role offer in compensation and remote work",
]


def main() -> int:
    for q in QUERIES:
        print("=" * 78)
        print(f"Q: {q}")
        print("-" * 78)
        for r in retrieve(q, k=3):
            preview = r.text.replace("\n", " ")[:120]
            print(f"  [{r.score:.3f}] {r.metadata['jd_id']} / "
                  f"{r.metadata['section']:24s} :: {preview}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
