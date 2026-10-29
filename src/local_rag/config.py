"""Configuration: one dataclass, loaded from rag.toml and overridden by env vars.

Chunking parameters live HERE, in the RAG layer. They are never passed to an
embedding API: embedding models only turn the text you hand them into a vector.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass
class Config:
    db_path: str = "rag.db"
    chunk_size: int = 800
    chunk_overlap: int = 120
    embed_provider: str = "local"
    embed_model: str = ""
    top_k: int = 4
    llm_provider: str = "auto"
    llm_model: str = ""


_ENV = {
    "RAG_DB": "db_path",
    "EMBED_PROVIDER": "embed_provider",
    "EMBED_MODEL": "embed_model",
    "LLM_PROVIDER": "llm_provider",
    "LLM_MODEL": "llm_model",
}


def load_config(path: str | os.PathLike | None = None) -> Config:
    path = Path(path or os.environ.get("RAG_CONFIG", "rag.toml"))
    values: dict = {}
    if path.exists():
        values = tomllib.loads(path.read_text(encoding="utf-8"))
    known = {f.name for f in fields(Config)}
    cfg = Config(**{k: v for k, v in values.items() if k in known})
    for env, attr in _ENV.items():
        if os.environ.get(env):
            setattr(cfg, attr, os.environ[env])
    return cfg
