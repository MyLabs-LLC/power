# Repository Guidelines

## Project Structure & Module Organization

Compact Python RAG server for MyLabs Studio:

- `app.py`: FastAPI backend for React, streaming chat, datasets, and discovery.
- `web/`: React single-page app served on port `7860`.
- `rag.py`: retrieval, reranking, and OpenAI-compatible generation.
- `ingest.py`: document ingestion CLI and helpers.
- `discover.py`: arXiv search/download.
- `config.py`: shared paths, model names, prompts, and retrieval settings.
- `test_rag.py`: RAG smoke/accuracy test.
- `start_server.sh`: launcher for llama-server plus React/FastAPI.
- `stop_server.sh`: stops both services by PID file and port fallback.

Runtime data lives in `documents/<dataset>/` and `vectorstore/`. Static assets are in `static/`. Treat `vectorstore/` and caches as local artifacts.

## Build, Test, and Development Commands

Use the `rag` conda environment before running Python commands:

```bash
conda activate rag
```

- `./start_server.sh`: start the TurboQuant llama-server and React UI.
- `./stop_server.sh`: stop the React UI and llama-server.
- `./start_server.sh --safe`: run with the safer TurboQuant cache setting.
- `python app.py`: run only the FastAPI/React app.
- `python ingest.py`: list available datasets under `documents/`.
- `python ingest.py CERN --reset`: rebuild `rag_cern` from `documents/CERN/`.
- `python test_rag.py`: run the smoke/accuracy suite.
- `curl http://localhost:8001/health`: verify server readiness.

## Coding Style & Naming Conventions

Write idiomatic Python with 4-space indentation, clear names, and small helpers. Existing modules use `snake_case` for functions and variables, and uppercase constants in `config.py`. `web/app.js` uses plain React without a build step. Keep comments focused on model-server assumptions or vector-store behavior.

No formatter or linter config is committed. Preserve the current style and avoid broad unrelated rewrites.

## Testing Guidelines

`test_rag.py` is a smoke/accuracy test, not a unit-test suite. It expects a populated Chroma store and reachable model server. Start the stack with `./start_server.sh` or confirm `:8001` is healthy. Add checks as `TESTS` entries with expected keywords and a concise topic label.

## Commit & Pull Request Guidelines

Git history uses short imperative summaries, for example `Add MyLabs Studio RAG server`. Keep commits focused and describe the user-visible or operational change. For PRs, include a summary, test commands, dataset/model prerequisites, and screenshots for UI changes. Mention port, model path, context-size, or vector-store changes explicitly.

## Agent-Specific Instructions

Read source files before editing; do not rely only on `CLAUDE.md`. Preserve the three chat modes, server-sent event streaming, and `RAGEngine.collection is None` guards. Use `./stop_server.sh` before restarting services. Do not delete or reset user datasets in `documents/` or `vectorstore/` unless requested.
