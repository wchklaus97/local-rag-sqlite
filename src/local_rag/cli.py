"""Command line: rag ingest <folder> | rag ask "<question>" | rag stats | rag reset."""

from __future__ import annotations

import argparse
import sys

from .config import load_config
from .embed import get_embedder
from .generate import generate, resolve_provider
from .ingest import build_index
from .retrieve import retrieve
from .store import Store, reset_db


def _open(cfg):
    embedder = get_embedder(cfg.embed_provider, cfg.embed_model)
    return embedder, Store(cfg.db_path, embedder.name, embedder.dim)


def cmd_ingest(args, cfg) -> int:
    embedder, store = _open(cfg)
    n = build_index(args.folder, cfg, embedder, store, log=print)
    print(f"Done: {n} chunks added, {store.count()} total in {cfg.db_path}")
    return 0


def cmd_ask(args, cfg) -> int:
    embedder, store = _open(cfg)
    hits = retrieve(store, embedder, args.question, args.k or cfg.top_k, args.dept, args.entity)
    if not hits:
        print("No matching chunks (index empty, or the --dept/--entity filter matched nothing). Run `rag ingest <folder>` first.")
        return 1

    print(f"Retrieved {len(hits)} chunks:\n")
    for n, h in enumerate(hits, start=1):
        where = h.source + (f" p.{h.page}" if h.page else "")
        org = "/".join(x for x in (h.entity, h.department) if x)
        where += f"  org={org}" if org else ""
        snippet = " ".join(h.text.split())
        print(f"[{n}] score={h.score:.3f}  {where}  (chunk {h.chunk_index})")
        print(f"    {snippet[:300]}{'...' if len(snippet) > 300 else ''}\n")

    provider = resolve_provider(cfg.llm_provider)
    answer = generate(args.question, hits, cfg.llm_provider, cfg.llm_model)
    if answer is None:
        print("-- retrieval-only mode: no LLM configured (set OPENAI_API_KEY, GEMINI_API_KEY or OLLAMA_MODEL) --")
    else:
        print(f"Answer ({provider}):\n{answer}")
    return 0


def cmd_stats(_args, cfg) -> int:
    embedder, store = _open(cfg)
    print(f"db: {cfg.db_path}  backend: {store.backend}  embedder: {store.embedder_info()}")
    for source, n in store.sources():
        print(f"  {n:4d} chunks  {source}")
    print(f"total: {store.count()}")
    return 0


def cmd_reset(args, cfg) -> int:
    print(f"Deleted {cfg.db_path}" if reset_db(cfg.db_path) else f"Nothing to delete at {cfg.db_path}")
    if args.rebuild:
        return cmd_ingest(args, cfg)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rag", description=__doc__)
    p.add_argument("--config", help="path to rag.toml (default: ./rag.toml or $RAG_CONFIG)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ingest", help="ingest a folder of PDF / Markdown / txt")
    s.add_argument("folder")
    s.set_defaults(fn=cmd_ingest)

    s = sub.add_parser("ask", help="ask a question against the index")
    s.add_argument("question")
    s.add_argument("-k", type=int, help="number of chunks to retrieve (default: top_k from config)")
    s.add_argument("--dept", help="only chunks from this department folder, e.g. hr")
    s.add_argument("--entity", help="only chunks from this entity folder, e.g. hk-hq")
    s.set_defaults(fn=cmd_ask)

    s = sub.add_parser("stats", help="show what is in the index")
    s.set_defaults(fn=cmd_stats)

    s = sub.add_parser("reset", help="delete the index; add --rebuild FOLDER to re-ingest")
    s.add_argument("--rebuild", metavar="FOLDER", help="re-ingest this folder after deleting")
    s.set_defaults(fn=cmd_reset)

    args = p.parse_args(argv)
    if args.cmd == "reset" and args.rebuild:
        args.folder = args.rebuild
    cfg = load_config(args.config)
    try:
        return args.fn(args, cfg)
    except (RuntimeError, FileNotFoundError, ValueError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
