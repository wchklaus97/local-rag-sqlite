from pathlib import Path

from _minipdf import make_pdf
from local_rag.config import Config
from local_rag.embed import HashEmbedder
from local_rag.generate import build_prompt, generate
from local_rag.ingest import build_index, load_documents
from local_rag.retrieve import retrieve
from local_rag.store import Store

EXAMPLES = Path(__file__).parent.parent / "examples"
HK = EXAMPLES / "harbour-lane"  # layout: <entity>/<department>/file


def _index(tmp_path, folder=EXAMPLES, **cfg):
    embedder = HashEmbedder()
    store = Store(tmp_path / "rag.db", embedder.name, embedder.dim)
    build_index(folder, Config(chunk_size=300, chunk_overlap=50, **cfg), embedder, store)
    return embedder, store


def test_end_to_end_retrieval_finds_right_source(tmp_path):
    embedder, store = _index(tmp_path)
    hits = retrieve(store, embedder, "what does chunk overlap do for a sentence cut at a boundary", k=3)
    assert "learning/chunking-guide.md" in {h.source for h in hits}


def test_cjk_question_hits_cantonese_doc(tmp_path):
    embedder, store = _index(tmp_path)
    hits = retrieve(store, embedder, "切塊同重疊 overlap", k=3)
    assert "learning/rag-zh-hk.md" in {h.source for h in hits}


def test_reingest_replaces_instead_of_duplicating(tmp_path):
    embedder, store = _index(tmp_path)
    n = store.count()
    build_index(EXAMPLES, Config(chunk_size=300, chunk_overlap=50), embedder, store)
    assert store.count() == n


def test_pdf_pages_keep_page_numbers(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "two.pdf").write_bytes(make_pdf(["alpha apples are red", "beta bananas are yellow"]))
    docs = load_documents(folder)
    assert [(d.source, d.page) for d in docs] == [("two.pdf", 1), ("two.pdf", 2)]

    embedder, store = _index(tmp_path, folder)
    hit = retrieve(store, embedder, "yellow bananas", k=1)[0]
    assert (hit.source, hit.page) == ("two.pdf", 2)


def test_empty_index_returns_no_hits(tmp_path):
    embedder = HashEmbedder()
    assert retrieve(Store(tmp_path / "e.db", embedder.name, embedder.dim), embedder, "anything") == []


def test_retrieval_only_mode_without_llm(tmp_path, monkeypatch):
    for var in ("OPENAI_API_KEY", "GEMINI_API_KEY", "OLLAMA_MODEL"):
        monkeypatch.delenv(var, raising=False)
    embedder, store = _index(tmp_path)
    hits = retrieve(store, embedder, "what is rag", k=2)
    assert generate("what is rag", hits, "auto") is None
    prompt = build_prompt("what is rag", hits)
    assert "[1]" in prompt and "Question: what is rag" in prompt


def test_hk_company_question_in_cantonese_finds_hr_handbook(tmp_path):
    embedder, store = _index(tmp_path, HK)
    hits = retrieve(store, embedder, "年假 申請 要提前幾多日", k=3)
    assert hits[0].source == "hk-hq/hr/staff-handbook.md"
    assert (hits[0].entity, hits[0].department) == ("hk-hq", "hr")


def test_org_filter_limits_results_to_department_and_entity(tmp_path):
    embedder, store = _index(tmp_path, HK)
    fin = retrieve(store, embedder, "approval needed over HKD 300", k=3, department="finance")
    assert fin and all(h.department == "finance" for h in fin)
    sz = retrieve(store, embedder, "who approves purchase orders", k=3, entity="sz-subsidiary")
    assert sz and all(h.entity == "sz-subsidiary" for h in sz)
    assert retrieve(store, embedder, "anything", k=3, department="no-such-dept") == []
