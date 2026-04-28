"""Build ChromaDB index from JD corpus.

Run after adding/updating JDs in data/jd_corpus/.

    python scripts/build_index.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))

from src.retrieval import index_corpus  # noqa: E402


def main() -> int:
    t0 = time.time()
    n = index_corpus(reset=True)
    print(f"indexed {n} chunks in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
