"""JD corpus loader + chunker.

Loads .md files from data/jd_corpus/, parses YAML frontmatter, splits
each JD into section-level chunks (## h2 headings) for ChromaDB ingestion.

Each chunk preserves metadata: jd_id, title, seniority, company, section.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT / "data" / "jd_corpus"

FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)
SECTION_RE = re.compile(r"^##\s+(.+?)$", re.MULTILINE)


@dataclass
class JDChunk:
    chunk_id: str  # f"{jd_id}::{section_slug}"
    jd_id: str
    title: str
    seniority: str
    company: str
    section: str
    text: str

    def to_metadata(self) -> dict:
        return {
            "jd_id": self.jd_id,
            "title": self.title,
            "seniority": self.seniority,
            "company": self.company,
            "section": self.section,
        }


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def parse_jd_file(path: Path) -> tuple[dict, str]:
    """Return (frontmatter_dict, body_markdown)."""
    raw = path.read_text()
    m = FRONTMATTER_RE.match(raw)
    if not m:
        raise ValueError(f"{path.name}: no frontmatter found")
    fm = yaml.safe_load(m.group(1))
    body = m.group(2).lstrip("\n")
    return fm, body


def chunk_jd(fm: dict, body: str) -> list[JDChunk]:
    """Split JD body into section-level chunks by ## headings."""
    matches = list(SECTION_RE.finditer(body))
    if not matches:
        return [JDChunk(
            chunk_id=f"{fm['id']}::full",
            jd_id=fm["id"], title=fm["title"], seniority=fm["seniority"],
            company=fm["company"], section="full", text=body.strip(),
        )]

    chunks: list[JDChunk] = []
    for i, m in enumerate(matches):
        section = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        text = body[start:end].strip()
        if not text:
            continue
        # Include the section heading in chunk text — improves retrieval for
        # queries like "what are the responsibilities" by anchoring keyword.
        text_with_header = f"{fm['title']} — {section}\n\n{text}"
        chunks.append(JDChunk(
            chunk_id=f"{fm['id']}::{_slug(section)}",
            jd_id=fm["id"], title=fm["title"], seniority=fm["seniority"],
            company=fm["company"], section=section, text=text_with_header,
        ))
    return chunks


def load_corpus(corpus_dir: Path = CORPUS_DIR) -> list[JDChunk]:
    """Load all JDs and return flat list of chunks."""
    chunks: list[JDChunk] = []
    for path in sorted(corpus_dir.glob("jd_*.md")):
        fm, body = parse_jd_file(path)
        chunks.extend(chunk_jd(fm, body))
    return chunks


if __name__ == "__main__":
    chunks = load_corpus()
    print(f"loaded {len(chunks)} chunks from {len(set(c.jd_id for c in chunks))} JDs")
    by_section: dict[str, int] = {}
    for c in chunks:
        by_section[c.section] = by_section.get(c.section, 0) + 1
    for s, n in sorted(by_section.items()):
        print(f"  {s:30s} {n}")
