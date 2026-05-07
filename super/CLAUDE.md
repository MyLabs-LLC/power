# MyLabs Studio RAG Server

> Project root: `/home/lence/power/super/`
> Conda env: `rag`
> UI: React/FastAPI on `http://localhost:7860`
> LLM server: OpenAI-compatible `llama-server` on `http://localhost:8001`
> Last updated: 2026-05-07

MyLabs Studio is a compact local RAG application for research corpora. It runs a FastAPI backend, a no-build React single-page app, per-dataset Chroma vector stores, document ingestion, arXiv discovery, inline source links, and an optional SQLite RDBMS mode for exact SQL-backed questions.

## Current Architecture

```text
Browser :7860
  -> React SPA in web/
  -> FastAPI backend in app.py
       -> Dataset files in documents/<dataset>/
       -> ChromaDB collections in vectorstore/
       -> Optional SQLite RDBMS in documents/<dataset>/.rdbms/dataset.sqlite
       -> OpenAI-compatible llama-server :8001
```

The active model is `Nemotron-3-Nano-30B-A3B-Q8_0.gguf`, launched through the TurboQuant llama.cpp fork. The model has a native 1,048,576 token training context, but the server default is 262,144 tokens because that is the stable startup profile for the 2x RTX 3090 machine.

## Key Files

| Path | Purpose |
|---|---|
| `app.py` | FastAPI backend, React static serving, SSE chat, dataset APIs, document text routes, discovery endpoint, SQL RDBMS chat flow |
| `web/app.js` | React SPA with chat history, dataset controls, discovery controls, inline source links, SQL RDBMS mode |
| `web/index.html` | React UMD shell and cache-busted asset references |
| `web/styles.css` | MyLabs Studio UI styling |
| `rag.py` | `RAGEngine`: Chroma collection switching, retrieval, reranking, streaming generation |
| `ingest.py` | CLI and streaming document ingestion, PDF page extraction, chunking |
| `discover.py` | arXiv search, LLM query planning, adaptive query expansion, PDF download/filtering |
| `rdbms.py` | Dataset-to-SQL layer: normalized SQLite schema, profile generation, SQL validation/execution |
| `config.py` | Paths, model names, retrieval settings, generation settings, prompts |
| `start_server.sh` | Starts llama-server and the React/FastAPI app with health checks and PID files |
| `stop_server.sh` | Stops the UI and llama-server by PID file and port fallback |
| `test_rag.py` | RAG smoke/accuracy checks |
| `documents/<dataset>/` | User datasets and generated per-dataset artifacts |
| `vectorstore/` | Chroma persistent storage |

Runtime artifacts such as `documents/`, `vectorstore/`, `.run/`, `logs/`, and generated `.rdbms/` databases are local state. Do not delete user datasets unless explicitly asked.

## Running The Stack

Use the `rag` conda environment for Python commands.

```bash
cd /home/lence/power/super
./start_server.sh
```

Useful variants:

```bash
./start_server.sh --safe               # turbo3 V-cache, safer quality profile
./start_server.sh --baseline           # mainline llama.cpp, f16 cache, 8K context
./start_server.sh --ctx-size 524288    # 512K context
./start_server.sh --ctx-size 1048576   # full 1M native window, may OOM on startup
./stop_server.sh                       # stop both UI and llama-server
```

Health checks:

```bash
curl http://localhost:7860/api/health
curl http://localhost:8001/health
curl http://localhost:8001/v1/models
```

Run only the UI/backend:

```bash
conda activate rag
python app.py
```

## Dataset Operations

Datasets are folders under `documents/`. Dataset names are normalized by `clean_dataset_name()` in `app.py` before use.

CLI examples:

```bash
conda activate rag
python ingest.py                         # list datasets
python ingest.py CERN                    # ingest documents/CERN/
python ingest.py CERN --reset            # rebuild rag_cern collection
python ingest.py --all --reset           # rebuild all datasets
```

UI operations:

- Create dataset
- Upload and ingest files
- Re-index dataset
- Delete dataset
- Generate cached summary
- Discover papers from arXiv
- Generate RDBMS

The ingestion path preserves PDF page numbers where extractable. Chunks include `source`, `page`, `chunk_index`, and text metadata.

## Chat Modes

The current UI has four modes:

| Mode | Behavior |
|---|---|
| `RAG Only` | Retrieve from the active dataset and answer only from retrieved context |
| `RAG + Model` | Retrieve context, answer from documents first, and allow model background knowledge with clear distinction |
| `Model Only` | Bypass retrieval and call the local model directly |
| `SQL RDBMS` | Convert the question into validated read-only SQLite SQL for the selected dataset RDBMS |

Preserve server-sent event streaming from `POST /api/chat/stream`. The frontend expects `stats`, `sources`, `token`, `error`, and `done` events.

## Source Links And Document Text

RAG answers cite source chunks inline. The React app converts citations like:

```text
[source: paper.pdf, page 3, chunk 12]
```

into clickable links such as:

```text
/api/documents/paper.pdf/pages/3?chunk=12
```

Important document routes:

- `GET /api/documents/{source}/pages/{page}?chunk=N`
- `GET /api/documents/{source}/text?chunk=N`
- Dataset-scoped equivalents under `/api/datasets/{name}/documents/...`

The document view highlights cited chunk text in yellow. If the requested page is stale or no longer exists in the current PDF, the route now returns an HTML fallback showing the full extracted document text instead of JSON `404`. If the chunk still exists under another page, it still tries to highlight it.

## SQL RDBMS Feature

`Gen RDBMS` creates:

```text
documents/<dataset>/.rdbms/dataset.sqlite
documents/<dataset>/.rdbms/profile.json
```

The SQLite schema is normalized around:

- `documents`
- `pages`
- `chunks`
- `terms`
- `chunk_terms`
- `measurements`
- `citations`
- `sql_audit`

SQL mode behavior:

1. Loads the generated schema/profile.
2. Uses a strict SQL compiler prompt.
3. Extracts SQL from model output, including malformed `<think>` wrappers.
4. Validates read-only SQL only.
5. Dry-checks with SQLite `EXPLAIN QUERY PLAN`.
6. Repairs invalid SQL once.
7. Falls back to conservative evidence-search SQL if compilation fails.
8. Streams the generated SQL before execution.
9. Executes the query and returns exact rows.

Safety constraints:

- Only `SELECT` or `WITH`.
- No `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `CREATE`, `ATTACH`, `PRAGMA`, etc.
- No multi-statement SQL.
- Read-only SQLite connection for query execution.

For legal, scientific, or medical data, prefer returning `source_name`, `page_number`, `chunk_index`, `source_url`, and evidence text so results stay source-grounded.

## arXiv Discovery

Discovery is no longer a narrow literal query list. The current pipeline is:

1. Ask the local LLM to create a topic-specific arXiv query plan.
2. Force the first query to be the user's exact topic.
3. Include acronym expansions, synonyms, broader/narrower scientific terms, methods, treatments, measurements, legal concepts, datasets, or mechanisms as appropriate.
4. Search arXiv with conservative pacing.
5. After the first result batch, extract important title/abstract terms and append adaptive follow-up queries when there is room.
6. Score papers using the whole query plan, not just the original phrase.
7. Download PDFs, verify extractable text quality, then ingest downloaded documents.

The activity log shows:

- `Discovery query plan: ...`
- `Expanded query plan: ...`
- `Adaptive follow-up queries: ...` when generated
- arXiv cooldown and retry messages

Defaults:

- `DiscoveryRequest.max_papers = 15`
- `DiscoveryRequest.num_queries = 8`
- UI query count max is 12
- `RAG_DISCOVER_ARXIV_DELAY` defaults to 7 seconds
- 429 backoff is intentionally conservative

## Model And Context

Model path:

```text
/home/lence/models/nemotron-3-nano/Nemotron-3-Nano-30B-A3B-Q8_0.gguf
```

Current launch profile:

```text
--ctx-size 262144
-ctk q8_0
-ctv turbo2
--parallel 1
--batch-size 2048
--ubatch-size 512
--flash-attn on
--tensor-split 0.5,0.5
```

The model metadata reports `n_ctx_train = 1048576`, but `llama-server` currently runs `n_ctx = 262144` by default. `config.py` still exposes `LLM_CTX_SIZE = 1048576` for display/context accounting, so when exact running context matters, check `http://localhost:8001/props` or the llama-server log.

## API Endpoints

Core:

- `GET /`
- `GET /api/health`
- `GET /api/bootstrap`
- `GET /api/datasets`
- `POST /api/datasets`
- `POST /api/datasets/select`
- `DELETE /api/datasets/{name}`

Ingestion/discovery:

- `POST /api/datasets/{name}/upload`
- `POST /api/datasets/{name}/reindex`
- `GET /api/datasets/{name}/summary`
- `POST /api/datasets/{name}/summarize`
- `POST /api/discover`

RDBMS:

- `GET /api/datasets/{name}/rdbms`
- `POST /api/datasets/{name}/rdbms/generate`

Chat/documents:

- `POST /api/chat/stream`
- `GET /api/documents/{source}/pages/{page}`
- `GET /api/documents/{source}/text`
- `GET /api/datasets/{name}/documents/{source}/pages/{page}`
- `GET /api/datasets/{name}/documents/{source}/text`

## Frontend Notes

The React app is plain `React.createElement` with no build step. It is served from `web/` and loaded through CDN React UMD scripts.

Local browser chat state:

```text
localStorage key: mylabs-studio-chats-v2
```

The frontend defensively stringifies old object-shaped chat content and sanitizes chat history before sending it to the backend. This prevents `[object Object]` responses from malformed persisted history.

When editing `web/app.js` or `web/styles.css`, bump the query-string cache version in `web/index.html`.

## Testing And Validation

Light checks:

```bash
python -m py_compile app.py discover.py ingest.py rag.py rdbms.py config.py
node --check web/app.js
git diff --check
```

Runtime checks:

```bash
curl http://localhost:7860/api/health
curl http://localhost:8001/health
curl http://localhost:8001/props | head -c 1000
python test_rag.py
```

RDBMS smoke path:

1. Select a dataset.
2. Click `Gen RDBMS`.
3. Switch chat mode to `SQL RDBMS`.
4. Ask a question.
5. Confirm generated SQL appears inline before results.

Document-link smoke path:

```bash
curl -s -o /tmp/doc.html -w '%{http_code} %{content_type}\n' \
  'http://localhost:7860/api/documents/<source.pdf>/pages/<page>?chunk=<chunk>'
```

Expected result is `200 text/html` for valid pages and stale-page fallbacks.

## Operational Gotchas

- Use `./stop_server.sh` before restarting services.
- Do not reset/delete `documents/` or `vectorstore/` unless the user explicitly asks.
- `RAGEngine.collection` can be `None`; preserve guards around `engine.collection`.
- Existing chat modes plus `SQL RDBMS` should remain distinct.
- arXiv rate limits are real; avoid lowering pacing unless explicitly needed.
- The embedding/reranker load warnings about `position_ids` are benign.
- The UI has no bundler. Do not add a frontend build chain unless requested.
- The app may have dirty runtime data even when code is clean. Keep commits focused on source/config/docs.

## Git Notes

Recent integrated commit:

```text
85ac157 Add MyLabs Studio RAG workflows
```

That commit introduced the React/FastAPI workflow, RDBMS SQL mode, generalized discovery planning, source-link handling, launcher/stop scripts, and local artifact ignore rules.

Use short imperative commit messages. For UI/API changes, mention test commands and any model/dataset prerequisites in PR notes.
