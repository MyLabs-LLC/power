# MyLabs Studio — Nemotron-3-Nano 30B-A3B RAG Server

> **Project root:** `/home/lence/power/super/`
> **Conda env:** `rag` (Python 3.13) — activate with `conda activate rag`
> **LLM server:** `llama-server` (TurboQuant fork) on `:8001`
> **Chat UI:** Gradio "MyLabs Studio" on `:7860`

---

## 1. Orientation (read first if resuming cold)

Local RAG stack running **NVIDIA Nemotron-3-Nano-30B-A3B** (Unsloth Q8_0 GGUF, ~34 GB) split across **2× RTX 3090** (48 GB VRAM total). Documents are chunked, embedded with BGE-base on CPU, stored in **per-dataset ChromaDB collections**, and retrieved with cross-encoder reranking. The chat UI is a Gradio app themed after **Unsloth Studio** (MyLabs-LLC fork, AGPL-3.0) with **MyLabs LLC branding**.

The llama.cpp server is a [TheTom/llama-cpp-turboquant fork](https://github.com/TheTom/llama-cpp-turboquant) that adds `turbo2/3/4` KV-cache compression types. This is what makes giant context windows fit on 24 GB consumer GPUs.

### 30-second mental model

```
Browser :7860  ──►  Gradio app (app.py in rag env)
                         │
                         ├─► ChromaDB (./vectorstore/, per-dataset collections)
                         │   embed: BAAI/bge-base-en-v1.5 (CPU)
                         │   rerank: BAAI/bge-reranker-base (CPU)
                         │
                         └─► OpenAI-compatible API on :8001
                             llama-server (turboquant build)
                             Nemotron-3-Nano Q8_0, tensor-split 0.5,0.5
                             -ctk q8_0 -ctv turbo2, 256K ctx default
```

---

## 2. Current state of the build (last touched 2026-04-19)

- ✅ llama.cpp turboquant fork built at `~/llama.cpp-turboquant/build/bin/llama-server`
- ✅ Mainline llama.cpp fallback at `~/llama.cpp.mainline/llama-server`
- ✅ Q8_0 GGUF downloaded to `~/models/nemotron-3-nano/`
- ✅ `rag` conda env with openai, chromadb, sentence-transformers, gradio 6.11, pypdf, python-docx, langchain-text-splitters
- ✅ `start_server.sh` launches llama-server **and** the Gradio app in one command (conda-activates `rag`, waits for `/health`, traps Ctrl-C to shut down both)
- ✅ `app.py` rewritten with Unsloth Studio visual design + MyLabs LLC branding (see §10)
- ✅ `static/mylabs-logo.png` = MyLabs LLC avatar (256×256 JPEG), used as header logo, chatbot avatar, and favicon
- ✅ Default ctx reduced from 1M to **256K** to avoid pipeline-parallel compute-buffer OOM on startup (pass `--ctx-size 1048576` for the full native window if VRAM allows)

### Implementation files (all live in project root)

| File | Role |
|---|---|
| `app.py` | Gradio UI — chat, dataset CRUD, paper discovery, pipeline stats |
| `rag.py` | `RAGEngine` class — retrieve → rerank → generate (streaming) |
| `ingest.py` | Document ingestion — CLI + `ingest_dataset` / `ingest_dataset_streaming` |
| `discover.py` | arXiv paper search + download for auto-ingestion |
| `config.py` | Shared settings (URLs, models, chunking, retrieval K, prompts) |
| `start_server.sh` | One-command launcher: llama-server + Gradio (with health check & trap cleanup) |
| `test_rag.py` | Smoke test for the RAG pipeline |
| `03_collect_data.py` | (related to the parent training project, not RAG) |
| `static/mylabs-logo.png` | Branding asset |
| `documents/<name>/` | One folder per dataset (ingested into `rag_<name>` Chroma collection) |
| `vectorstore/` | ChromaDB persistent storage |

Always **read the files directly** for code — don't reconstruct from this doc.

---

## 3. How to run it

### One-command launch (normal path)

```bash
cd ~/power/super
./start_server.sh
```

That script:
1. Kills any stale process on port 8001.
2. Launches `llama-server` in the background, PID captured.
3. `trap`s `EXIT INT TERM` to kill the server on Ctrl-C.
4. Polls `http://localhost:8001/health` for up to 4 min; bails if the server dies.
5. Sources `conda.sh`, activates the `rag` env, runs `python app.py` in the foreground.

Ctrl-C shuts down **both** the Gradio app and llama-server cleanly.

### Variants

```bash
./start_server.sh --ctx-size 1048576   # full 1M window (may trigger startup OOM-and-retry)
./start_server.sh --safe               # turbo3 V-cache (higher quality), 256K ctx
./start_server.sh --baseline           # mainline llama.cpp, f16 cache, 8K ctx
./start_server.sh --ctx-size 524288    # 512K
./start_server.sh --autostart on|off   # enable/disable systemd --user service
```

### Running just the LLM server (no UI)

```bash
~/llama.cpp-turboquant/build/bin/llama-server \
    --model ~/models/nemotron-3-nano/Nemotron-3-Nano-30B-A3B-Q8_0.gguf \
    --alias nemotron-3-nano --n-gpu-layers 99 --tensor-split 0.5,0.5 \
    -ctk q8_0 -ctv turbo2 --ctx-size 262144 --parallel 1 \
    --batch-size 2048 --ubatch-size 512 --flash-attn on \
    --port 8001 --host 0.0.0.0
```

### Running just the UI

```bash
conda activate rag
cd ~/power/super
python app.py
```

### Ingest docs from the CLI (alt to UI upload)

```bash
conda activate rag
cd ~/power/super
mkdir -p ./documents/CERN
cp /path/to/papers/* ./documents/CERN/
python ingest.py CERN              # ingest folder into rag_cern
python ingest.py CERN --reset      # wipe collection first
python ingest.py                   # list available datasets
```

### Smoke tests

```bash
curl http://localhost:8001/health
curl http://localhost:8001/v1/models    # should show "nemotron-3-nano"
python test_rag.py                       # RAG pipeline smoke test
```

---

## 4. Project layout

```
/home/lence/power/super/
├── CLAUDE.md              # this file
├── start_server.sh        # launcher (llama-server + Gradio, conda + health check + trap)
├── config.py              # URLs, model names, chunking/retrieval params, system prompts
├── rag.py                 # RAGEngine: switch_dataset, retrieve, generate_stream(_direct)
├── ingest.py              # ingest_dataset, ingest_dataset_streaming, CLI entry
├── discover.py            # arXiv search + download; title_to_filename helper
├── app.py                 # Gradio UI (MyLabs Studio design system)
├── test_rag.py            # smoke test
├── static/
│   └── mylabs-logo.png    # MyLabs LLC branding (256×256 JPEG)
├── documents/             # per-dataset subfolders; each → rag_<name> Chroma collection
│   └── <dataset>/
└── vectorstore/           # ChromaDB PersistentClient data dir
```

Adjacent: `/home/lence/power/CLAUDE.md` covers the **pretraining pipeline** (unrelated project; don't mix them up).

---

## 5. Non-obvious gotchas (must-know before editing)

These are things that cost time to debug in earlier sessions. Respect them.

### 5.1 Gradio 6.11 API differences

Gradio 6 is the installed version. It differs from Gradio 4/5 in ways our code hits:

- **`gr.Chatbot` does NOT accept `type="messages"`.** The messages format is the default in v6 — passing `type=` raises `TypeError`. Our chatbot uses the messages dict format (`{"role": "user"|"assistant", "content": str}`) implicitly.
- **`theme` and `css` must be passed to `.launch()`, NOT `gr.Blocks()` constructor.** Gradio 6.0 moved these. `gr.Blocks(theme=..., css=...)` emits a deprecation warning and may fail.
- **`gr.Chatbot(avatar_images=...)` must be a file path, not a data URI.** Gradio validates the arg via `Path.exists()` which `OSError: [Errno 36] File name too long`s on base64 data URIs. Pass `LOGO_PATH` (a filesystem path).
- **`allowed_paths=[...]` in `.launch()` is required** to serve files from `static/` — we use it so the favicon resolves.

### 5.2 The dataset dropdown must tolerate empty choices

`gr.Dropdown(choices=get_dataset_choices(), allow_custom_value=True)` — without `allow_custom_value=True`, if `./documents/` is empty on first launch the dropdown's `.change` event fires with value `""` against choices `[]` and Gradio throws `Value:  is not in the list of choices: []`. The `allow_custom_value=True` bypasses that validation.

### 5.3 Default context is 256K, not 1M

The model's **native training window is 1,048,576 tokens**, but on startup llama.cpp first tries to allocate a **pipeline-parallel compute buffer** (~9.3 GiB at 1M), which OOMs on the 3090s after the Q8_0 weights load. It auto-retries without pipeline parallelism and succeeds with smaller buffers (~3 GB + ~1 GB + ~2 GB host) — so 1M *works* but boots with noisy retry logs. **Default is now 256K** in `start_server.sh`. Pass `--ctx-size 1048576` to opt into 1M.

### 5.4 The Gradio app takes ~10-15s to start

It loads BGE-base (embed) + BGE-reranker-base (rerank) on CPU at import time, both of which log `UNEXPECTED embeddings.position_ids` warnings. **These are benign** (sentence-transformers ships a `position_ids` tensor the newer HF architecture doesn't use) — do not "fix" them.

### 5.5 `RAGEngine.collection` can be `None`

After deleting the active dataset, `engine.collection` is set to `None`. The chat handler checks `engine.collection is not None and engine.collection.count() > 0` before retrieving. Preserve that guard if refactoring.

### 5.6 Three chat modes

`mode_selector` (pill-tab Radio) has three values: **"RAG Only"**, **"RAG + Model"**, **"Model Only"**. Only the first two retrieve from the vector store; Model Only bypasses retrieval entirely and calls `engine.generate_stream_direct`. Don't flatten these into a boolean.

### 5.7 Chat history is stored in `gr.State`

`all_chats` is a list of `{"name": str, "messages": list[dict]}`; `current_idx` indexes into it. The left-column Radio shows `"<idx+1>. <name>"` labels. Chat name auto-derives from the first 6 words of the first user message.

### 5.8 No mainline llama.cpp at `/usr/local` — use the prebuilt binaries

- Primary: `~/llama.cpp-turboquant/build/bin/llama-server` (supports `-ctv turbo2|3|4`)
- Fallback: `~/llama.cpp.mainline/llama-server` (f16/q8_0 cache only, used by `--baseline`)

### 5.9 Port already in use errors

If `./start_server.sh` fails with "port in use", run `lsof -ti:7860 :8001 | xargs -r kill` and retry. The launcher auto-frees 8001 at the top, but 7860 is the Gradio port and relies on a previous clean shutdown.

---

## 6. Architecture

```
┌─────────────────────────────────────────────────────┐
│           MyLabs Studio — Gradio UI (:7860)          │
│   chat history │ chatbot + composer │ dataset/arXiv  │
├─────────────────────────────────────────────────────┤
│                  RAG Orchestrator                    │
│   query → embed → vector search → rerank → generate  │
├──────────────────────┬──────────────────────────────┤
│   Vector Store       │   LLM Inference Server        │
│   ChromaDB           │   llama-server (turboquant)   │
│   per-dataset cols   │   Nemotron-3-Nano Q8_0         │
│   name = rag_<x>     │   2× RTX 3090, tensor split   │
│   Embed: bge-base    │   OpenAI API on :8001          │
│   Rerank: bge-base   │   256K ctx (1M optional)      │
├──────────────────────┴──────────────────────────────┤
│              Document Ingestion Pipeline             │
│   PDF/DOCX/TXT/MD → chunk (1024 tok, 128 overlap)   │
│   → embed (BGE) → upsert to dataset collection      │
└─────────────────────────────────────────────────────┘
```

### RAG pipeline stages (for the stats panel)

1. **Embed query** — BGE-base-en-v1.5 on CPU
2. **Vector search** — top `RETRIEVAL_K=20` candidates by cosine distance
3. **Rerank** — BGE-reranker-base cross-encoder → top `TOP_K=8`
4. **LLM generation** — llama-server streaming via OpenAI client

Each stage reports its time; final stats include TTFT, prefill tok/s, decode tok/s, context utilization.

### Per-dataset isolation

Folders under `./documents/` map 1:1 to ChromaDB collections named `rag_<lowercased_name_with_underscores>`. The `CERN` dataset never bleeds into `moon_landing`. The UI's dropdown picks the active collection; `RAGEngine.switch_dataset` swaps `self.collection`.

---

## 7. Hardware & model setup (reference, done already)

- **GPUs:** 2× NVIDIA RTX 3090 (24 GB each, 48 GB total)
- **Model:** `unsloth/Nemotron-3-Nano-30B-A3B-GGUF` variant `Q8_0` (~34 GB single file)
- **OS:** Ubuntu 24.04
- **RAM:** ≥64 GB recommended

### System deps (one-time)

```bash
sudo apt update && sudo apt install -y \
    build-essential cmake git curl wget libcurl4-openssl-dev \
    python3 python3-pip python3-venv
```

### Build llama.cpp (one-time)

```bash
# Mainline fallback
git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp.mainline
cmake ~/llama.cpp.mainline -B ~/llama.cpp.mainline/build \
    -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON
cmake --build ~/llama.cpp.mainline/build --config Release -j$(nproc) \
    --target llama-cli llama-server
cp ~/llama.cpp.mainline/build/bin/llama-* ~/llama.cpp.mainline/

# TurboQuant fork (primary)
git clone https://github.com/TheTom/llama-cpp-turboquant ~/llama.cpp-turboquant
cd ~/llama.cpp-turboquant
git checkout feature/turboquant-kv-cache
cmake -B build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc) --target llama-cli llama-server
```

> If GCC 13+ errors with `invalid use of 'extern' in linkage specification` in `ggml/src/ggml-cpu/ops.cpp`, change `extern "C" GGML_API int turbo3_cpu_wht_group_size;` to `extern "C" int turbo3_cpu_wht_group_size;` and rebuild.

### Download model (one-time)

```bash
pip install huggingface_hub hf_transfer
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download \
    unsloth/Nemotron-3-Nano-30B-A3B-GGUF \
    --local-dir ~/models/nemotron-3-nano --include "*Q8_0*"
```

### Create the `rag` conda env (one-time)

```bash
conda create -n rag python=3.13 -y
conda activate rag
pip install uv
uv pip install openai chromadb sentence-transformers gradio pypdf \
    python-docx tiktoken langchain langchain-community langchain-text-splitters
```

---

## 8. llama.cpp server reference

### Critical flags

- `--n-gpu-layers 99` — offload all layers to GPU
- `--tensor-split 0.5,0.5` — even split across the two 3090s
- `-ctk q8_0 -ctv turbo2` — asymmetric KV cache. K=q8_0 (precision-critical for attention scoring), V=turbo2 (~6.4× compression for memory)
- `--ctx-size 262144` — **default 256K** (see §5.3)
- `--parallel 1` — single-sequence mode; dedicates the full KV cache to one request
- `--flash-attn on` — memory-efficient attention
- `--batch-size 2048 --ubatch-size 512` — smaller batch/ubatch → smaller compute buffer → more room for KV cache

### TurboQuant cache types

| Type | Compression | Quality | Use case |
|---|---|---|---|
| `turbo4` | 3.8× | Safest | Quality-critical |
| `turbo3` | 5.1× | Balanced | `--safe` default |
| `turbo2` | 6.4× | Aggressive | Max-context default |

Why this matters: the Nemotron-Nano hybrid has only **6 of 53 layers using attention** (the rest are Mamba-2 with constant-size state), so KV cache is already cheap per token. `q8_0 K + turbo2 V` costs **~2 KB/token** — a full 1M window is ~2 GB of KV.

### llama-server endpoints

- `GET /health` — liveness check
- `GET /v1/models` — lists `nemotron-3-nano`
- `POST /v1/chat/completions` — OpenAI-compatible chat
- `POST /v1/completions` — OpenAI-compatible text completion

References:
- Fork: https://github.com/TheTom/llama-cpp-turboquant
- TurboQuant plus research: https://github.com/TheTom/turboquant_plus
- Mainline tracking: https://github.com/ggml-org/llama.cpp/discussions/20969
- Paper: https://arxiv.org/abs/2504.19874

---

## 9. OpenAI-compatible API for external agents

llama-server speaks OpenAI's chat-completions wire format. Any agent framework that targets OpenAI can drive Nemotron-3-Nano locally without a shim.

### Connection

| Setting | Value |
|---|---|
| Base URL | `http://localhost:8001/v1` |
| API key | any non-empty string; convention `sk-no-key-required` |
| Model name | `nemotron-3-nano` (matches `--alias`) |
| Context | whatever `--ctx-size` was set to (default 256K) |

### Env vars (most OpenAI clients)

```bash
export OPENAI_BASE_URL="http://localhost:8001/v1"
export OPENAI_API_KEY="sk-no-key-required"
export OPENAI_MODEL="nemotron-3-nano"
```

### curl smoke test

```bash
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"nemotron-3-nano","messages":[{"role":"user","content":"Say hello."}],"max_tokens":32}'
```

### Framework pointers

- **LangChain:** `ChatOpenAI(base_url="http://localhost:8001/v1", api_key="sk-no-key-required", model="nemotron-3-nano")`
- **LlamaIndex:** `OpenAILike(api_base="http://localhost:8001/v1", api_key="sk-no-key-required", model="nemotron-3-nano", is_chat_model=True)`
- **AutoGen:** `{"base_url": "http://localhost:8001/v1", "api_key": "sk-no-key-required", "model": "nemotron-3-nano"}`

### Caveats

- **Single-request.** llama.cpp is not vLLM — a second concurrent request queues.
- **Prompt caching is implicit.** Identical prefixes hit the KV cache; keep system prompts stable.
- **Tool/function calling** is supported via the model's chat template. Watch the tool-schema size against the ctx budget.
- **LAN access.** Server binds `0.0.0.0:8001`. **No auth.** Only expose on trusted networks.

---

## 10. UI design system — MyLabs Studio

The UI is a Gradio Blocks app themed to match **Unsloth Studio** (React/Tailwind in its native form at [MyLabs-LLC/unsloth](https://github.com/MyLabs-LLC/unsloth), AGPL-3.0). We port only the **visual language** (design tokens, typography, layout), not the code.

### Branding

- **Header logo & favicon:** `static/mylabs-logo.png` (MyLabs LLC avatar). Embedded as base64 data URI in the header HTML; passed as a file path to `gr.Chatbot(avatar_images=...)` (see §5.1).
- **Header text:** "MyLabs Studio — Nemotron-3-Nano RAG · 2× RTX 3090"
- **Status pill:** green "Online" indicator (pulses with primary color glow)
- **Context chip:** shows `Q8_0 · <N>K ctx` in the header

### Design tokens (in `app.py` CSS block)

```css
--mylabs-primary:     oklch(0.6929 0.1396 166.5513);  /* mint-teal — Unsloth brand */
--mylabs-primary-soft: oklch(0.6929 0.1396 166.5513 / 0.15);
--mylabs-bg:          oklch(0.24 0 0);
--mylabs-card:        oklch(0.28 0 0);
--mylabs-sidebar:     oklch(0.21 0 0);
--mylabs-border:      oklch(0.38 0 0);
--mylabs-muted:       oklch(0.33 0 0);
--mylabs-muted-fg:    oklch(0.72 0 0);
--mylabs-radius:      14px;
--mylabs-radius-lg:   18px;
```

Fonts loaded from Google Fonts: **Inter** (body), **Space Grotesk** (headings), **JetBrains Mono** (code/stats).

### Layout

```
┌─── HEADER (logo + title + ctx chip + Online pill) ─────────────┐
│                                                                 │
│  LEFT          │      MAIN                    │  RIGHT          │
│  (scale 1)     │      (scale 4)               │  (scale 2)      │
│                │                              │                 │
│  + New Chat    │  Chatbot (560px, markdown)   │  Dataset        │
│  Conversations │                              │    dropdown     │
│  · chat 1      │  Mode pills: RAG / R+M / M   │    info box     │
│  · chat 2      │                              │  ▸ New Dataset  │
│  · …           │  Composer (textarea + Send)  │  ▾ Upload Docs  │
│                │                              │    (drag-drop)  │
│  Delete Chat   │  ▸ Pipeline stats            │  ▸ Discover     │
│                │                              │    (arXiv)      │
└────────────────┴──────────────────────────────┴─────────────────┘
```

### Gradio theme

```python
THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.emerald,
    font=[gr.themes.GoogleFont("Inter"), ...],
    radius_size=gr.themes.sizes.radius_lg,
).set(...)  # OKLCH overrides match the design tokens above
```

Theme + CSS are passed to `.launch()`, not `gr.Blocks()` (see §5.1).

---

## 11. Tuning guide

| Problem | Fix |
|---|---|
| llama-server OOM on startup | Default is already 256K. Further reduce `--ctx-size` (e.g. 131072), or fall back to `--baseline` (8K f16). |
| Slow generation | `nvidia-smi`; if one GPU maxes, adjust `--tensor-split` (e.g. `0.45,0.55`). |
| Bad retrieval quality | Raise `RETRIEVAL_K` (overfetch for the reranker) or bump `CHUNK_OVERLAP`. |
| Reranker too slow | Lower `RETRIEVAL_K` — rerank cost ≈ linear in K. |
| Hallucinations | Lower `TEMPERATURE` to 0.3, raise `TOP_K` to 12, use `SYSTEM_PROMPT_RAG_ONLY` in config.py. |
| Cross-dataset bleed | Datasets are collection-isolated — check which `rag_<name>` the UI is pointed at. |
| Want full 1M ctx | `./start_server.sh --ctx-size 1048576` (accepts the startup OOM-retry noise). |
| TurboQuant quality issues | `-ctv turbo4` (3.8×, safest) or `--baseline`. |
| Gradio `Value:  is not in the list of choices: []` | Add `allow_custom_value=True` to the offending dropdown (see §5.2). |
| Port 7860 in use | `lsof -ti:7860 | xargs -r kill` then re-run `./start_server.sh`. |

---

## 12. Monitoring

```bash
# GPU utilization + VRAM
watch -n 1 nvidia-smi

# llama-server health
curl http://localhost:8001/health

# Chunk counts per dataset
conda activate rag
python -c "
import chromadb
db = chromadb.PersistentClient(path='./vectorstore')
for c in db.list_collections():
    print(f'{c.name}: {c.count()} chunks')
"
```

---

## 13. History of recent changes (this session)

- **app.py**: full UI rewrite — MyLabs Studio branding, Unsloth Studio design tokens, 3-column layout (chat history / chatbot / dataset+discovery), pill-tab mode selector, composer with shadow surface, collapsible pipeline stats. Handler logic unchanged.
- **app.py**: added `allow_custom_value=True` to dataset dropdown to tolerate empty initial state.
- **app.py**: adapted for Gradio 6.11 — removed `type="messages"` from `gr.Chatbot`, moved `theme`/`css` to `.launch()`, used file path (not data URI) for `avatar_images`.
- **start_server.sh**: now launches llama-server **and** the Gradio app. Server runs in background, trap handler cleans up on exit, polls `/health` before starting app.py. Sources conda and activates `rag` env.
- **start_server.sh**: default `--ctx-size` lowered 1048576 → 262144 (256K) to avoid pipeline-parallel compute-buffer OOM on startup.
- **static/mylabs-logo.png**: added (MyLabs LLC GitHub org avatar, 256×256 JPEG saved as .png).
