import pytest

from local_rag.chunk import split_text

TEXT = " ".join(f"Sentence number {i} talks about topic {i % 7}." for i in range(120))


def test_chunks_respect_size():
    chunks = split_text(TEXT, size=200, overlap=40)
    assert len(chunks) > 5
    assert all(0 < len(c) <= 200 for c in chunks)


def test_neighbouring_chunks_overlap():
    chunks = split_text(TEXT, size=200, overlap=40)
    for a, b in zip(chunks, chunks[1:]):
        # the head of the next chunk was already present at the tail of the previous one
        assert b[:15] in a[-120:]


def test_no_text_is_lost():
    chunks = split_text(TEXT, size=200, overlap=40)
    words = {w for c in chunks for w in c.split()}
    assert set(TEXT.split()) <= words


def test_short_text_is_one_chunk():
    assert split_text("hello world", size=800, overlap=120) == ["hello world"]


def test_empty_text():
    assert split_text("   ", size=100, overlap=10) == []


def test_no_overlap_allowed_and_terminates():
    assert "".join(split_text("a" * 1000, size=100, overlap=0)) == "a" * 1000


def test_cjk_without_spaces_terminates():
    chunks = split_text("你好。" * 500, size=100, overlap=20)
    assert chunks and all(len(c) <= 100 for c in chunks)


@pytest.mark.parametrize("size,overlap", [(0, 0), (100, 100), (100, -1), (100, 150)])
def test_invalid_params(size, overlap):
    with pytest.raises(ValueError):
        split_text("text", size, overlap)


def test_split_text_is_standalone():
    """chunk.py must stay copy-pasteable: importing it must not import the rest of the package."""
    import subprocess, sys
    code = "import sys, local_rag.chunk; print(any(m.startswith('local_rag.') and m != 'local_rag.chunk' for m in sys.modules))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "False"
