"""Stage 5 - Retrieval.

Embed the question with the SAME embedder used at ingest time, then ask the store
for the k nearest chunks. Retrieval returns evidence only; it writes no answer.

Org filter (`department` / `entity`) is applied AFTER the vector search, so we over-fetch
(5x) when filtering; a very selective filter can still return fewer than k hits.
"""

from __future__ import annotations

from .store import Hit


def retrieve(store, embedder, question: str, k: int = 4, department: str | None = None, entity: str | None = None) -> list[Hit]:
    if store.count() == 0:
        return []
    vector = embedder.embed([question])[0]
    filtering = bool(department or entity)
    hits = store.search(vector, k * 5 if filtering else k)
    hits = [h for h in hits if (not department or h.department == department) and (not entity or h.entity == entity)]
    return hits[:k]
