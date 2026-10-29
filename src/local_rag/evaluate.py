"""Evaluation: does retrieval find the passage that actually answers each question?

Without numbers, "it looks right" is the only signal. This module scores retrieval against
a hand-written QA set (eval/harbour_lane_qa.jsonl) so you can compare embedders and chunking.

A HIT = a retrieved chunk from the expected source that CONTAINS the evidence string.
Matching on the source file alone would hide chunking mistakes (right file, answer cut off).

Metrics for top-k retrieval:
  hit@1   share of questions whose first chunk is a hit
  hit@k   share with a hit anywhere in the top k (a.k.a. recall@k for one relevant passage)
  MRR@k   mean of 1/rank of the first hit (0 when none) - rewards ranking the answer high
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .config import Config
from .ingest import build_index
from .retrieve import retrieve
from .store import Store


@dataclass(frozen=True)
class QA:
    id: str
    lang: str
    q: str
    source: str
    evidence: str


@dataclass
class Result:
    embedder: str
    chunk_size: int
    chunk_overlap: int
    chunks: int
    hit1: float
    hitk: float
    mrr: float
    hitk_by_lang: dict[str, float] = field(default_factory=dict)
    misses: list[str] = field(default_factory=list)


def load_qa(path: str | Path) -> list[QA]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [QA(**json.loads(line)) for line in lines if line.strip()]


def first_hit_rank(hits, qa: QA) -> int | None:
    for rank, h in enumerate(hits, start=1):
        if h.source == qa.source and qa.evidence in h.text:
            return rank
    return None


def evaluate_config(folder, qa: list[QA], embedder, size: int, overlap: int, k: int = 3) -> Result:
    with tempfile.TemporaryDirectory() as tmp:
        store = Store(Path(tmp) / "eval.db", embedder.name, embedder.dim)
        try:
            n_chunks = build_index(folder, Config(chunk_size=size, chunk_overlap=overlap), embedder, store)
            ranks = {item.id: first_hit_rank(retrieve(store, embedder, item.q, k), item) for item in qa}
        finally:
            store.close()

    def share(items) -> float:
        items = list(items)
        return sum(1 for i in items if ranks[i.id]) / len(items) if items else 0.0

    return Result(
        embedder=embedder.name,
        chunk_size=size,
        chunk_overlap=overlap,
        chunks=n_chunks,
        hit1=sum(1 for i in qa if ranks[i.id] == 1) / len(qa),
        hitk=share(qa),
        mrr=sum(1 / ranks[i.id] for i in qa if ranks[i.id]) / len(qa),
        hitk_by_lang={lang: share(i for i in qa if i.lang == lang) for lang in sorted({i.lang for i in qa})},
        misses=[i.id for i in qa if not ranks[i.id]],
    )


def run_grid(folder, qa, embedders, sizes, overlap_fracs, k: int = 3, log=lambda _m: None) -> list[Result]:
    results = []
    for emb in embedders:
        for size in sizes:
            for frac in overlap_fracs:
                r = evaluate_config(folder, qa, emb, size, int(size * frac), k)
                log(f"  {emb.name:>60}  size={size:<4} overlap={r.chunk_overlap:<4} hit@{k}={r.hitk:.2f}")
                results.append(r)
    return results


def to_markdown(results: list[Result], k: int, n_questions: int) -> str:
    langs = sorted({lang for r in results for lang in r.hitk_by_lang})
    head = (
        f"| embedder | chunk_size | overlap | chunks | hit@1 | hit@{k} | MRR@{k} | "
        + " | ".join(f"hit@{k} {lang}" for lang in langs)
        + " |"
    )
    sep = "|" + "---|" * (7 + len(langs))
    rows = [
        f"| `{r.embedder}` | {r.chunk_size} | {r.chunk_overlap} | {r.chunks} | {r.hit1:.2f} | {r.hitk:.2f} | {r.mrr:.2f} | "
        + " | ".join(f"{r.hitk_by_lang.get(lang, 0):.2f}" for lang in langs)
        + " |"
        for r in results
    ]
    return f"{n_questions} questions, top-k = {k}.\n\n" + "\n".join([head, sep, *rows]) + "\n"
