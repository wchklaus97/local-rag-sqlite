"""Optional UI:  pip install -e '.[ui]'  &&  streamlit run ui/streamlit_app.py"""

import streamlit as st

from local_rag.config import load_config
from local_rag.embed import get_embedder
from local_rag.generate import generate, resolve_provider
from local_rag.retrieve import retrieve
from local_rag.store import Store

cfg = load_config()
st.title("local-rag-sqlite")


@st.cache_resource
def _load():
    embedder = get_embedder(cfg.embed_provider, cfg.embed_model)
    return embedder, Store(cfg.db_path, embedder.name, embedder.dim, force_python=False)


embedder, store = _load()
st.caption(f"{store.count()} chunks · {store.embedder_info()} · {store.backend}")

question = st.text_input("Ask a question about your documents")
k = st.slider("top-k", 1, 10, cfg.top_k)
if question:
    hits = retrieve(store, embedder, question, k)
    answer = generate(question, hits, cfg.llm_provider, cfg.llm_model)
    if answer:
        st.subheader(f"Answer ({resolve_provider(cfg.llm_provider)})")
        st.write(answer)
    else:
        st.info("Retrieval-only mode: no LLM configured.")
    st.subheader("Retrieved chunks")
    for n, h in enumerate(hits, start=1):
        with st.expander(f"[{n}] {h.source}{f' p.{h.page}' if h.page else ''} · score {h.score:.3f}"):
            st.write(h.text)
