import pytest

from local_rag.chunk import Chunk
from local_rag.store import Store, reset_db

CHUNKS = [Chunk("a.md", None, 0, "about cats"), Chunk("b.pdf", 2, 0, "about dogs"), Chunk("c.md", None, 0, "about fish")]
VECTORS = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


@pytest.fixture(params=[False, True], ids=["sqlite-vec", "python-fallback"])
def store(request, tmp_path):
    s = Store(tmp_path / "t.db", "test-embedder", 3, force_python=request.param)
    yield s
    s.close()


def test_roundtrip_returns_nearest_with_metadata(store):
    store.add(CHUNKS, VECTORS)
    hits = store.search([0.1, 0.9, 0.0], k=2)
    assert [h.text for h in hits] == ["about dogs", "about cats"]
    assert (hits[0].source, hits[0].page, hits[0].chunk_index) == ("b.pdf", 2, 0)
    assert hits[0].score > hits[1].score
    assert hits[0].score == pytest.approx(0.9 / (0.01 + 0.81) ** 0.5, abs=1e-4)


def test_delete_source_removes_chunk_and_vector(store):
    store.add(CHUNKS, VECTORS)
    store.delete_source("b.pdf")
    assert store.count() == 2
    assert "about dogs" not in [h.text for h in store.search([0.0, 1.0, 0.0], k=3)]


def test_backend_name(store):
    assert store.backend in {"sqlite-vec", "python-fallback"}


def test_refuses_to_mix_embedders(tmp_path):
    Store(tmp_path / "t.db", "model-a", 3).close()
    with pytest.raises(RuntimeError, match="reset"):
        Store(tmp_path / "t.db", "model-b", 3)


def test_reset_db(tmp_path):
    Store(tmp_path / "t.db", "m", 3).close()
    assert reset_db(tmp_path / "t.db") is True
    assert reset_db(tmp_path / "t.db") is False
