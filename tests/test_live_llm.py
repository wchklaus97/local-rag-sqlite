"""Opt-in tests against a real LLM (OpenAI or any OpenAI-compatible API such as OpenRouter).

Skipped by default. Run with:
    RUN_LIVE_LLM=1 LIVE_MODEL=openai/gpt-4o-mini uv run pytest tests/test_live_llm.py
(needs OPENAI_API_KEY, and OPENAI_BASE_URL=https://openrouter.ai/api/v1 for OpenRouter)
"""

import os
from pathlib import Path

import pytest

from local_rag.config import Config
from local_rag.embed import HashEmbedder
from local_rag.generate import generate
from local_rag.ingest import build_index
from local_rag.retrieve import retrieve
from local_rag.store import Store

pytestmark = pytest.mark.skipif(
    not (os.environ.get("RUN_LIVE_LLM") and os.environ.get("OPENAI_API_KEY")),
    reason="live LLM test: set RUN_LIVE_LLM=1 and OPENAI_API_KEY",
)
MODEL = os.environ.get("LIVE_MODEL", "openai/gpt-4o-mini")
HK = Path(__file__).parent.parent / "examples" / "harbour-lane"


@pytest.fixture(scope="module")
def index(tmp_path_factory):
    emb = HashEmbedder()
    store = Store(tmp_path_factory.mktemp("live") / "rag.db", emb.name, emb.dim)
    build_index(HK, Config(), emb, store)
    return store, emb


def test_grounded_answer_with_citation(index):
    store, emb = index
    q = "年假申請要提前幾多個工作天？"
    answer = generate(q, retrieve(store, emb, q, k=2), "openai", MODEL)
    assert "5" in answer and "[" in answer


def test_department_filter_then_answer(index):
    store, emb = index
    q = "Who must approve a taxi fare over HKD 300?"
    answer = generate(q, retrieve(store, emb, q, k=2, department="finance"), "openai", MODEL)
    assert "Finance Manager" in answer


def test_says_unknown_when_context_lacks_answer(index):
    store, emb = index
    q = "What is the company's policy on cryptocurrency trading?"
    answer = generate(q, retrieve(store, emb, q, k=2), "openai", MODEL).lower()
    refusals = ("know", "not", "no information", "不知道", "唔知", "找不到", "搵唔到")
    assert any(r in answer for r in refusals)
    assert not any("\u4e00" <= ch <= "\u9fff" for ch in answer), "English question should get an English reply"
