from pathlib import Path

from local_rag.chunk import Chunk
from local_rag.embed import HashEmbedder
from local_rag.evaluate import QA, evaluate_config, first_hit_rank, load_qa, to_markdown
from local_rag.store import Hit

ROOT = Path(__file__).parent.parent
QA_FILE = ROOT / "eval" / "harbour_lane_qa.jsonl"
HK = ROOT / "examples" / "harbour-lane"


def test_qa_set_is_valid_and_every_evidence_exists_verbatim():
    qa = load_qa(QA_FILE)
    assert len(qa) >= 20 and len({q.id for q in qa}) == len(qa)
    assert {q.lang for q in qa} == {"zh", "en"}
    for q in qa:
        text = (HK / q.source).read_text(encoding="utf-8")
        assert q.evidence in text, f"{q.id}: evidence not in {q.source}"


def test_hit_requires_right_source_and_evidence():
    qa = QA("x", "en", "q", "a.md", "HKD 300")
    hit = lambda src, text: Hit(1, src, None, 0, text, 0.9)  # noqa: E731
    assert first_hit_rank([hit("b.md", "HKD 300"), hit("a.md", "taxi HKD 300")], qa) == 2
    assert first_hit_rank([hit("a.md", "answer was cut off here")], qa) is None


def test_evaluate_config_metrics_are_consistent():
    qa = load_qa(QA_FILE)
    r = evaluate_config(HK, qa, HashEmbedder(), size=400, overlap=60, k=3)
    assert 0 <= r.hit1 <= r.hitk <= 1 and r.hit1 <= r.mrr <= r.hitk
    assert set(r.hitk_by_lang) == {"en", "zh"} and r.chunks > 0
    assert len(r.misses) == round((1 - r.hitk) * len(qa))
    assert "| `hash:384` | 400 | 60 |" in to_markdown([r], 3, len(qa))


def test_overlap_rescues_evidence_cut_at_a_boundary():
    """With tiny chunks and no overlap some evidence strings get split; overlap brings them back."""
    qa = load_qa(QA_FILE)
    no_overlap = evaluate_config(HK, qa, HashEmbedder(), size=60, overlap=0, k=20)
    with_overlap = evaluate_config(HK, qa, HashEmbedder(), size=60, overlap=20, k=20)
    assert with_overlap.hitk >= no_overlap.hitk
