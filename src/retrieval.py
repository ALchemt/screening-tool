"""ChromaDB-backed retrieval over JD corpus.

- Custom EmbeddingFunction calls OpenRouter (text-embedding-3-small).
- Persistent client at data/chroma/.
- top-k semantic search with metadata filters.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from openai import OpenAI

from .jd_corpus import JDChunk, load_corpus

ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIR = Path(os.environ.get("CHROMA_DIR", ROOT / "data" / "chroma"))
COLLECTION = "jd_corpus"


class OpenRouterEmbedding(EmbeddingFunction[Documents]):
    """Embeds via OpenRouter using OpenAI-compatible API."""

    def __init__(self, model: str | None = None):
        self.model = model or os.environ.get(
            "EMBED_MODEL", "openai/text-embedding-3-small")
        self.client = OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
        )

    def __call__(self, input: Documents) -> Embeddings:
        resp = self.client.embeddings.create(model=self.model, input=list(input))
        return [d.embedding for d in resp.data]

    @staticmethod
    def name() -> str:
        return "openrouter"


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    metadata: dict
    score: float  # cosine distance, lower = more similar


def get_collection(reset: bool = False):
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if reset:
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass
    return client.get_or_create_collection(
        name=COLLECTION,
        embedding_function=OpenRouterEmbedding(),
        metadata={"hnsw:space": "cosine"},
    )


def index_corpus(chunks: list[JDChunk] | None = None, reset: bool = True) -> int:
    """Build ChromaDB index from JD corpus. Returns chunk count."""
    if chunks is None:
        chunks = load_corpus()
    coll = get_collection(reset=reset)
    coll.add(
        ids=[c.chunk_id for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[c.to_metadata() for c in chunks],
    )
    return len(chunks)


def retrieve(query: str, k: int = 5,
             where: dict | None = None) -> list[RetrievedChunk]:
    """Semantic top-k search. `where` accepts ChromaDB metadata filter."""
    coll = get_collection(reset=False)
    res = coll.query(query_texts=[query], n_results=k, where=where)
    out: list[RetrievedChunk] = []
    for i in range(len(res["ids"][0])):
        out.append(RetrievedChunk(
            chunk_id=res["ids"][0][i],
            text=res["documents"][0][i],
            metadata=res["metadatas"][0][i],
            score=res["distances"][0][i],
        ))
    return out
