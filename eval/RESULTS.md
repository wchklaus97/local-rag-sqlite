# Retrieval evaluation - Harbour Lane QA set

25 questions, top-k = 3.

| embedder | chunk_size | overlap | chunks | hit@1 | hit@3 | MRR@3 | hit@3 en | hit@3 zh |
|---|---|---|---|---|---|---|---|---|
| `hash:384` | 100 | 0 | 23 | 0.48 | 0.68 | 0.55 | 0.53 | 1.00 |
| `hash:384` | 100 | 15 | 28 | 0.28 | 0.52 | 0.38 | 0.29 | 1.00 |
| `hash:384` | 200 | 0 | 13 | 0.36 | 0.64 | 0.48 | 0.47 | 1.00 |
| `hash:384` | 200 | 30 | 15 | 0.52 | 0.68 | 0.59 | 0.59 | 0.88 |
| `hash:384` | 400 | 0 | 7 | 0.48 | 0.68 | 0.57 | 0.53 | 1.00 |
| `hash:384` | 400 | 60 | 7 | 0.48 | 0.76 | 0.59 | 0.65 | 1.00 |
| `hash:384` | 800 | 0 | 4 | 0.52 | 0.92 | 0.69 | 0.88 | 1.00 |
| `hash:384` | 800 | 120 | 4 | 0.52 | 0.92 | 0.69 | 0.88 | 1.00 |
| `local:sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 100 | 0 | 23 | 0.56 | 0.84 | 0.67 | 0.82 | 0.88 |
| `local:sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 100 | 15 | 28 | 0.52 | 0.96 | 0.73 | 0.94 | 1.00 |
| `local:sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 200 | 0 | 13 | 0.40 | 0.92 | 0.61 | 0.88 | 1.00 |
| `local:sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 200 | 30 | 15 | 0.52 | 0.84 | 0.65 | 0.88 | 0.75 |
| `local:sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 400 | 0 | 7 | 0.56 | 0.96 | 0.75 | 0.94 | 1.00 |
| `local:sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 400 | 60 | 7 | 0.68 | 1.00 | 0.82 | 1.00 | 1.00 |
| `local:sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 800 | 0 | 4 | 0.80 | 0.96 | 0.87 | 1.00 | 0.88 |
| `local:sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 800 | 120 | 4 | 0.80 | 0.96 | 0.87 | 1.00 | 0.88 |

## What the numbers say

1. **The embedder matters most for a bilingual HK corpus.** English questions about Chinese-language policies: `hash` hit@3 en = 0.29–0.88, multilingual model = 0.82–1.00. A keyword embedder cannot cross languages; Chinese questions look perfect for `hash` (zh = 1.00) only because they share words with the documents. **Testing in one language would have hidden this.**
2. **Best setting here: multilingual, chunk_size 400, overlap 60** - hit@3 1.00, MRR 0.82.
3. **Trap: chunk_size 800 looks great but is nearly trivial.** Each document becomes one chunk (4 chunks total), so top-3 returns 75% of the corpus. Always read hit@k next to the chunk count; hit@1 and MRR are more honest here.
4. **Small eval sets are noisy.** One question = 0.04. Overlap sometimes "hurts" (hash 100/15, multilingual 200/30) - those gaps are 1–4 questions and mostly reflect which chunk boundary a phrase landed on. Treat differences under ~0.08 as noise; grow the QA set before tuning on it.

Reproduce: `uv sync --extra local && uv run rag eval --embedders hash,local --show-misses`
