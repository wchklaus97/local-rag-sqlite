# LEARNING — 經典 RAG 六步 / The classic RAG pipeline

Read the code in this order: `ingest.py → chunk.py → embed.py → store.py → retrieve.py → generate.py`.

---

## 繁體中文（zh-HK）

### 全貌
**Ingest → Chunking → Embedding → Vector store → Retrieval → Generation**
頭四步係「建索引」（做一次），後兩步係「回答問題」（每次提問都做）。

### 每一步：做咩、點解、常見錯誤

**1. Ingest（`ingest.py`）**
- 做咩：將 PDF／Markdown 讀成純文字，記低來源同頁碼。
- 點解：之後要引用出處，來源資料一定要喺呢度保留。
- 常見錯誤：掃描版 PDF 冇文字層，讀出嚟係空嘅（要 OCR）；Markdown 表格或 PDF 欄位次序亂咗。

**2. Chunking（`chunk.py`）— 最多人搞錯嘅一步**
- 做咩：將長文字切成適合檢索嘅段落，設定 `chunk_size` 同 `chunk_overlap`。
- 點解：檢索嘅最細單位就係 chunk。太細失去上下文；太大令向量模糊、prompt 又塞滿無關內容。重疊避免答案啱啱喺切口斷開。
- **常見錯誤：以為 chunk size／overlap 係 embedding API 嘅參數。**
  Embedding API 只做「一段字串 → 一個向量」，**唔會切文件**。字串超過 token 上限通常會被截斷或者報錯，即係超出部分根本搜尋唔到。所以 chunking 係 ingest／RAG 層嘅設計決定，要喺 embed 之前自己做（見 `rag.toml`）。
- 其他錯誤：喺字中間切開；overlap 太大導致結果重複。

**3. Embedding（`embed.py`）**
- 做咩：將每個 chunk 變成向量，意思相近嘅文字向量距離近。
- 點解：令「語意搜尋」得以進行，而唔係淨係比對關鍵字。
- 常見錯誤：問題同文件用唔同模型；換咗模型但冇重建索引（本 repo 會偵測並拒絕）；英文模型用嚟搜中文。`hash` embedder 淨係比對相同字詞，只係方便測試，唔係語意。

**4. Vector store（`store.py`）**
- 做咩：喺一個 SQLite 檔案入面存文字、元資料同向量，用 sqlite-vec 做最近鄰搜尋。
- 點解：細規模唔使另外起向量資料庫；一個檔案易備份、易刪除。
- 常見錯誤：向量維度同 `vec0` 表宣告唔一致；忘記用 cosine（未正規化向量用 L2 會偏差）；重新 ingest 同一檔案造成重複（本 repo 先刪舊 chunk）。

**5. Retrieval（`retrieve.py`）**
- 做咩：將問題 embed，攞最近嘅 k 個 chunk。
- 點解：LLM 睇唔到你成個資料庫，只可以睇你揀出嚟嗰幾段。檢索質素決定答案上限。
- 常見錯誤：k 太細漏答案、k 太大塞入噪音；只睇答案唔睇檢索結果。所以 CLI 會顯示每個 chunk 同分數，方便你除錯。

**6. Generation（`generate.py`）**
- 做咩：將問題同檢索結果放入 grounded prompt，叫 LLM 只根據內容作答並標註 [n]。
- 點解：減少幻覺，答案可追溯。
- 常見錯誤：prompt 冇叫模型「冇資料就話唔知」；冇 API key 就以為壞咗（其實會進入 retrieval-only 模式）。

### 公司結構 → metadata
資料夾 `hk-hq/finance/` 會變成 `entity=hk-hq`、`department=finance`，用 `rag ask --dept finance` 過濾。
陷阱：過濾係向量搜尋**之後**先做，所以會將候選放大 5 倍；部門太冷門仍可能少過 k 個結果。

### 練習
0. 用 `examples/harbour-lane` 問「年假申請要提前幾多日？」，再加 `--dept finance`，睇結果點變。
1. 將 `chunk_size` 改做 200／2000，重新 ingest，睇 `rag ask` 嘅檢索結果點變。
2. 將 `chunk_overlap` 改成 0，搵一條答案剛好喺切口嘅問題。
3. 裝 `.[local]`，用同一條問題比較 `hash` 同 sentence-transformers 嘅結果。

---

## English

### Big picture
**Ingest → Chunking → Embedding → Vector store → Retrieval → Generation.**
The first four build the index (once). The last two answer a question (every time).

### Each stage: what, why, common mistakes

**1. Ingest (`ingest.py`)**
- What: read PDF/Markdown into plain text, keeping source path and page number.
- Why: citations need provenance, and this is the only place it exists.
- Mistakes: scanned PDFs have no text layer (needs OCR); tables and multi-column layouts extract in a scrambled order.

**2. Chunking (`chunk.py`) — the most misunderstood stage**
- What: split long text into retrievable passages using `chunk_size` and `chunk_overlap`.
- Why: a chunk is the smallest thing retrieval can return. Too small loses context; too large blurs the vector and floods the prompt. Overlap keeps an answer from being cut at a boundary.
- **Mistake: thinking chunk size/overlap is an embedding-API setting.** An embedding endpoint maps one string to one vector. It does **not** split your document. Input beyond the model's token limit is truncated or rejected, so the tail is silently unsearchable. Chunking is an ingest/RAG-layer design decision you make *before* embedding (see `rag.toml`).
- Other mistakes: cutting mid-word; so much overlap that results are near-duplicates.

**3. Embedding (`embed.py`)**
- What: turn each chunk into a vector so similar meanings are near each other.
- Why: enables semantic search instead of keyword matching.
- Mistakes: embedding questions and documents with different models; changing models without rebuilding (this repo detects that and refuses); an English-only model on Chinese text. The `hash` embedder only matches shared words — it exists for tests and offline demos, not for meaning.

**4. Vector store (`store.py`)**
- What: keep text, metadata and vectors in one SQLite file; sqlite-vec does nearest-neighbour search.
- Why: no separate vector database needed at small scale; one file is easy to back up or delete.
- Mistakes: vector dimension not matching the `vec0` table; wrong distance metric for your vectors; duplicate chunks after re-ingesting (this repo deletes a file's old chunks first).

**5. Retrieval (`retrieve.py`)**
- What: embed the question, fetch the k nearest chunks.
- Why: the LLM only sees what you select. Retrieval quality caps answer quality.
- Mistakes: k too small misses the answer, too large adds noise; judging only the final answer instead of the retrieved chunks. The CLI prints every chunk with its score for exactly this reason.

**6. Generation (`generate.py`)**
- What: put the question and retrieved chunks in a grounded prompt; the LLM answers from them and cites `[n]`.
- Why: fewer hallucinations, traceable answers.
- Mistakes: a prompt that doesn't allow “I don't know”; assuming a missing API key is a bug (it just means retrieval-only mode).

### Company structure → metadata
A folder like `hk-hq/finance/` becomes `entity=hk-hq`, `department=finance`; filter with `rag ask --dept finance`.
Pitfall: the filter runs *after* vector search, so the candidate pool is widened 5x; a rare department can still leave fewer than k results.

### Exercises
0. Ingest `examples/harbour-lane`, ask the annual-leave question in Cantonese, then add `--dept finance` and see what changes.
1. Set `chunk_size` to 200, then 2000; re-ingest; compare what `rag ask` retrieves.
2. Set `chunk_overlap = 0` and find a question whose answer straddles a boundary.
3. Install `.[local]` and compare `hash` vs sentence-transformers on the same question.
