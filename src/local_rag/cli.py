"""Command line: rag ingest <folder> | rag ask "<question>" | rag stats | rag reset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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


def cmd_eval(args, cfg) -> int:
    from .embed import HashEmbedder
    from .evaluate import load_qa, run_grid, to_markdown

    qa = load_qa(args.qa)
    embedders = []
    for name in args.embedders.split(","):
        emb = get_embedder(name.strip(), cfg.embed_model if name.strip() == cfg.embed_provider else "")
        if name.strip() == "local" and isinstance(emb, HashEmbedder):
            print("skipping 'local': sentence-transformers not installed (pip install -e '.[local]')", file=sys.stderr)
            continue
        embedders.append(emb)
    sizes = [int(x) for x in args.sizes.split(",")]
    overlaps = [float(x) for x in args.overlaps.split(",")]
    print(f"Evaluating {len(qa)} questions on {args.folder} ...")
    results = run_grid(args.folder, qa, embedders, sizes, overlaps, args.k, log=print)
    table = to_markdown(results, args.k, len(qa))
    print("\n" + table)
    if args.show_misses:
        for r in results:
            print(f"misses {r.embedder} size={r.chunk_size} overlap={r.chunk_overlap}: {', '.join(r.misses) or '-'}")
    if args.out:
        Path(args.out).write_text(table, encoding="utf-8")
        print(f"wrote {args.out}")
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

    s = sub.add_parser("eval", help="score retrieval on a QA set across embedders and chunk settings")
    s.add_argument("--qa", default="eval/harbour_lane_qa.jsonl")
    s.add_argument("--folder", default="examples/harbour-lane")
    s.add_argument("--embedders", default="hash", help="comma list, e.g. hash,local")
    s.add_argument("--sizes", default="100,200,400,800", help="chunk sizes (characters)")
    s.add_argument("--overlaps", default="0,0.15", help="overlap as a fraction of chunk size")
    s.add_argument("-k", type=int, default=3)
    s.add_argument("--out", help="write the markdown table here")
    s.add_argument("--show-misses", action="store_true")
    s.set_defaults(fn=cmd_eval)

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
