"""Gradio chat UI for RAG inference — MyLabs Studio (Nemotron-3-Nano RAG).

Visual design adapted from Unsloth Studio (MyLabs-LLC/unsloth, AGPL-3.0):
we replicate the design-token system (OKLCH palette, Inter + Space Grotesk
typography, 1.1rem radius, chat-composer surface) in pure CSS so a Gradio
frontend can carry the same look without the React/Tailwind stack.
"""

import os
import base64
import time
import shutil
import threading
import queue
import gradio as gr
from rag import RAGEngine
from ingest import ingest_dataset, ingest_dataset_streaming
from discover import discover_and_download, title_to_filename
from config import DOCUMENTS_DIR, LLM_CTX_SIZE

engine = RAGEngine()

APP_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(APP_DIR, "static", "mylabs-logo.png")


def _logo_data_uri() -> str:
    if not os.path.isfile(LOGO_PATH):
        return ""
    with open(LOGO_PATH, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


LOGO_URI = _logo_data_uri()


# ── Helpers ──────────────────────────────────────────────────────────────────

def get_dataset_choices() -> list[str]:
    os.makedirs(DOCUMENTS_DIR, exist_ok=True)
    return sorted(
        d for d in os.listdir(DOCUMENTS_DIR)
        if os.path.isdir(os.path.join(DOCUMENTS_DIR, d)) and not d.startswith(".")
    )


def dataset_info(name: str) -> str:
    if not name:
        return ""
    doc_dir = os.path.join(DOCUMENTS_DIR, name)
    files = [f for f in os.listdir(doc_dir) if not f.startswith(".")] if os.path.isdir(doc_dir) else []
    chunks = engine.switch_dataset(name)
    file_list = "\n".join(f"  • {f}" for f in sorted(files)) if files else "  (no files)"
    return f"{name}\n{len(files)} files · {chunks} chunks\n\n{file_list}"


# ── Dataset management ───────────────────────────────────────────────────────

def on_select_dataset(name: str):
    if not name:
        return "No dataset selected.", gr.update()
    engine.switch_dataset(name)
    return dataset_info(name), gr.update(value=name)


def on_page_load():
    choices = get_dataset_choices()
    selected = choices[0] if choices else ""
    if selected:
        engine.switch_dataset(selected)
    info = dataset_info(selected) if selected else ""
    return gr.update(choices=choices, value=selected), info


def on_create_dataset(name: str):
    name = name.strip().replace(" ", "_")
    if not name:
        return "Enter a dataset name.", gr.update(), ""
    os.makedirs(os.path.join(DOCUMENTS_DIR, name), exist_ok=True)
    engine.switch_dataset(name)
    choices = get_dataset_choices()
    return f"Created dataset '{name}'.", gr.update(choices=choices, value=name), ""


def on_upload_files(files, dataset_name: str):
    if not dataset_name:
        yield "Select a dataset first.", ""
        return
    if not files:
        yield "No files uploaded.", ""
        return

    doc_dir = os.path.join(DOCUMENTS_DIR, dataset_name)
    os.makedirs(doc_dir, exist_ok=True)

    existing = set(f for f in os.listdir(doc_dir) if not f.startswith("."))
    paths = []
    renames = []

    for f in files:
        orig_name = os.path.basename(f)
        ext = os.path.splitext(orig_name)[1].lower()
        raw_title = os.path.splitext(orig_name)[0]
        new_name = title_to_filename(raw_title, ext=ext, existing=existing)
        existing.add(new_name)

        dest = os.path.join(doc_dir, new_name)
        shutil.copy2(f, dest)
        paths.append(dest)
        if new_name != orig_name:
            renames.append(f"{orig_name} -> {new_name}")

    yield f"Copied {len(paths)} file(s). Ingesting...", ""

    log_lines = [f"Uploaded {len(paths)} file(s):"]
    if renames:
        for r in renames:
            log_lines.append(f"  Renamed: {r}")

    for update in ingest_dataset_streaming(dataset_name, paths, reset=False):
        log_lines.append(update)
        yield "\n".join(log_lines), ""

    engine.switch_dataset(dataset_name)
    log_lines.append("Done.")
    yield "\n".join(log_lines), dataset_info(dataset_name)


def on_reindex_dataset(dataset_name: str):
    if not dataset_name:
        yield "Select a dataset first.", ""
        return

    doc_dir = os.path.join(DOCUMENTS_DIR, dataset_name)
    paths = [
        os.path.join(doc_dir, f) for f in sorted(os.listdir(doc_dir))
        if not f.startswith(".")
    ] if os.path.isdir(doc_dir) else []

    if not paths:
        yield f"No files in {dataset_name}/.", ""
        return

    log_lines = [f"Re-indexing {len(paths)} files..."]
    yield "\n".join(log_lines), ""

    for update in ingest_dataset_streaming(dataset_name, paths, reset=True):
        log_lines.append(update)
        yield "\n".join(log_lines), ""

    engine.switch_dataset(dataset_name)
    log_lines.append("Re-index complete.")
    yield "\n".join(log_lines), dataset_info(dataset_name)


def on_delete_dataset(dataset_name: str):
    if not dataset_name:
        return "Select a dataset first.", gr.update(), ""

    doc_dir = os.path.join(DOCUMENTS_DIR, dataset_name)
    if os.path.isdir(doc_dir):
        shutil.rmtree(doc_dir)
    engine.delete_dataset(dataset_name)
    engine.collection = None

    choices = get_dataset_choices()
    new_val = choices[0] if choices else ""
    if new_val:
        engine.switch_dataset(new_val)

    return (
        f"Deleted '{dataset_name}'.",
        gr.update(choices=choices, value=new_val),
        dataset_info(new_val) if new_val else "",
    )


# ── Paper discovery ──────────────────────────────────────────────────────────

def on_discover_papers(topic: str, dataset_name: str, max_papers: int, num_queries: int):
    if not topic.strip():
        yield "Enter a search topic.", ""
        return
    if not dataset_name:
        yield "Select or create a dataset first.", ""
        return

    log_lines = []
    progress_q = queue.Queue()

    def progress_cb(msg):
        log_lines.append(msg)
        progress_q.put(msg)

    result_holder = {}

    def _run_discovery():
        result_holder["stats"] = discover_and_download(
            topic=topic.strip(),
            dataset_name=dataset_name,
            max_papers=int(max_papers),
            num_queries=int(num_queries),
            progress_cb=progress_cb,
        )

    thread = threading.Thread(target=_run_discovery)
    thread.start()

    while thread.is_alive():
        try:
            while True:
                progress_q.get_nowait()
        except queue.Empty:
            pass
        yield "\n".join(log_lines), ""
        time.sleep(0.3)

    thread.join()
    stats = result_holder.get("stats", {})

    report_lines = [
        f"── Discovery Complete ──",
        f"Topic: {stats.get('topic', topic)}",
        f"Papers found: {stats.get('searched', 0)}",
        f"PDFs downloaded: {stats.get('downloaded', 0)}",
        f"Failed/skipped: {stats.get('failed', 0)}",
        "",
        "Papers:",
    ]
    for p in stats.get("papers", []):
        status = "OK" if p["has_pdf"] else "SKIP"
        report_lines.append(f"  [{status}] {p['title'][:70]}  (rel: {p['relevance']:.2f})")

    download_report = "\n".join(report_lines)

    if stats.get("downloaded", 0) == 0:
        yield download_report + "\n\nNo papers downloaded. Try broadening the search.", ""
        return

    doc_dir = os.path.join(DOCUMENTS_DIR, dataset_name)
    paths = [
        os.path.join(doc_dir, f) for f in sorted(os.listdir(doc_dir))
        if not f.startswith(".") and f.endswith(".pdf")
    ]

    ingest_log = [download_report, "", "── Ingesting Papers ──"]
    for update in ingest_dataset_streaming(dataset_name, paths, reset=True):
        ingest_log.append(update)
        yield "\n".join(ingest_log), ""

    engine.switch_dataset(dataset_name)
    ingest_log.append(f"\nIngestion complete.")
    yield "\n".join(ingest_log), dataset_info(dataset_name)


# ── Chat history management ──────────────────────────────────────────────────

def make_empty_chat():
    return {"name": "New Chat", "messages": []}


def chat_name_from_message(message: str) -> str:
    words = message.strip().split()
    name = " ".join(words[:6])
    if len(words) > 6:
        name += "..."
    return name


def get_chat_list_choices(all_chats):
    return [f"{i+1}. {c['name']}" for i, c in enumerate(all_chats)]


def on_new_chat(all_chats, current_idx):
    all_chats = all_chats + [make_empty_chat()]
    new_idx = len(all_chats) - 1
    choices = get_chat_list_choices(all_chats)
    return (
        all_chats,
        new_idx,
        [],
        "",
        gr.update(choices=choices, value=choices[new_idx]),
    )


def on_select_chat(selected_label, all_chats):
    if not selected_label:
        return all_chats, 0, [], "", gr.update()
    try:
        idx = int(selected_label.split(".")[0]) - 1
    except (ValueError, IndexError):
        idx = 0
    idx = max(0, min(idx, len(all_chats) - 1))
    return (
        all_chats,
        idx,
        all_chats[idx]["messages"],
        "",
        gr.update(),
    )


def on_delete_chat(all_chats, current_idx):
    if len(all_chats) <= 1:
        all_chats = [make_empty_chat()]
        choices = get_chat_list_choices(all_chats)
        return all_chats, 0, [], "", gr.update(choices=choices, value=choices[0])

    all_chats = all_chats[:current_idx] + all_chats[current_idx + 1:]
    new_idx = min(current_idx, len(all_chats) - 1)
    choices = get_chat_list_choices(all_chats)
    return (
        all_chats,
        new_idx,
        all_chats[new_idx]["messages"],
        "",
        gr.update(choices=choices, value=choices[new_idx]),
    )


# ── Chat ─────────────────────────────────────────────────────────────────────

def format_stats(stages):
    lines = []
    for label, value in stages:
        if label.startswith("─"):
            lines.append(value)
        else:
            lines.append(f"{label}: {value}")
    return "\n".join(lines)


def _build_final_stats(stages, llm_usage, token_count, first_token_time, t_gen_start, t_start):
    t_gen_total = time.time() - t_gen_start
    t_total = time.time() - t_start
    ttft = first_token_time or 0
    decode_time = t_gen_total - ttft
    tps = token_count / decode_time if decode_time > 0 else 0

    prompt_tokens = llm_usage["prompt_tokens"] if llm_usage else "?"
    completion_tokens = llm_usage["completion_tokens"] if llm_usage else token_count
    prefill_tps = f"{prompt_tokens / ttft:.0f} tok/s" if llm_usage and ttft > 0 else "n/a"
    ctx_usage = f"{llm_usage['total_tokens']}/{LLM_CTX_SIZE} ({llm_usage['total_tokens']*100/LLM_CTX_SIZE:.0f}%)" if llm_usage else "?"

    stages[-1] = (stages[-1][0], f"{t_gen_total:.1f}s")
    stages.append(("─", f"   TTFT (time to first token): {ttft:.2f}s"))
    stages.append(("─", f"   Prefill: {prompt_tokens} prompt tokens @ {prefill_tps}"))
    stages.append(("─", f"   Decode:  {completion_tokens} tokens in {decode_time:.1f}s @ {tps:.1f} tok/s"))
    stages.append(("─", f"   Context: {ctx_usage}"))
    stages.append(("─", ""))
    stages.append(("Total", f"{t_total:.1f}s"))


def respond_streaming(message: str, chat_history: list, all_chats: list, current_idx: int, mode: str):
    if not message.strip():
        yield "", chat_history, "", all_chats, current_idx, gr.update()
        return

    is_first_message = len(chat_history) == 0
    if is_first_message:
        all_chats[current_idx]["name"] = chat_name_from_message(message)

    use_rag = mode in ("RAG Only", "RAG + Model")

    has_dataset = engine.collection is not None and engine.collection.count() > 0
    if use_rag and not has_dataset:
        chat_history = chat_history + [{"role": "user", "content": message}]
        chat_history = chat_history + [{"role": "assistant", "content":
            "No dataset selected or dataset is empty. Select a dataset and upload/discover documents first."}]
        all_chats[current_idx]["messages"] = chat_history
        choices = get_chat_list_choices(all_chats)
        yield "", chat_history, "", all_chats, current_idx, gr.update(choices=choices, value=choices[current_idx])
        return

    t_start = time.time()
    stages = []

    oai_history = []
    for msg in chat_history[-8:]:
        oai_history.append({"role": msg["role"], "content": msg["content"]})

    chat_history = chat_history + [{"role": "user", "content": message}]
    choices = get_chat_list_choices(all_chats)
    chat_list_update = gr.update(choices=choices, value=choices[current_idx]) if is_first_message else gr.update()

    if mode == "Model Only":
        chat_history = chat_history + [{"role": "assistant", "content": ""}]
        stages.append(("Mode", "Model Only (no RAG)"))
        stages.append(("1. LLM generation", "waiting for first token..."))
        all_chats[current_idx]["messages"] = chat_history
        yield "", chat_history, format_stats(stages), all_chats, current_idx, chat_list_update

        t_gen_start = time.time()
        llm_usage = None
        first_token_time = None
        token_count = 0

        for delta, usage in engine.generate_stream_direct(message, history=oai_history):
            if usage:
                llm_usage = usage
            if delta:
                token_count += 1
                if first_token_time is None:
                    first_token_time = time.time() - t_gen_start
                chat_history[-1] = {"role": "assistant", "content": chat_history[-1]["content"] + delta}
                elapsed = time.time() - t_gen_start
                decode_time = elapsed - first_token_time
                tps = token_count / decode_time if decode_time > 0 else 0
                stages[-1] = ("1. LLM generation", f"TTFT {first_token_time:.1f}s | {token_count} tok @ {tps:.1f} tok/s")
                all_chats[current_idx]["messages"] = chat_history
                yield "", chat_history, format_stats(stages), all_chats, current_idx, gr.update()

        _build_final_stats(stages, llm_usage, token_count, first_token_time, t_gen_start, t_start)
        all_chats[current_idx]["messages"] = chat_history
        yield "", chat_history, format_stats(stages), all_chats, current_idx, gr.update()
        return

    # RAG modes: retrieve first
    chat_history = chat_history + [{"role": "assistant", "content": "*Searching documents...*"}]
    stages.append(("Mode", f"{mode}"))
    stages.append(("1. Embed query", "running..."))
    all_chats[current_idx]["messages"] = chat_history
    yield "", chat_history, format_stats(stages), all_chats, current_idx, chat_list_update

    chunks, timings = engine.retrieve(message)
    sources = set(c["source"] for c in chunks)

    retrieval_k = engine._last_retrieval_k
    total_chunks = engine.collection.count()
    stages[-1] = ("1. Embed query", f"{timings['embed']:.2f}s")
    stages.append(("2. Vector search", f"{timings['search']:.2f}s  ({min(retrieval_k, total_chunks)} candidates from {total_chunks} chunks)"))
    stages.append(("3. Rerank", f"{timings['rerank']:.2f}s  (top {len(chunks)} of {min(retrieval_k, total_chunks)} reranked)"))

    gen_step = 4
    stages.append((f"{gen_step}. LLM generation", "waiting for first token..."))

    chat_history[-1] = {"role": "assistant", "content": ""}
    yield "", chat_history, format_stats(stages), all_chats, current_idx, gr.update()

    t_gen_start = time.time()
    first_token_time = None
    token_count = 0
    llm_usage = None

    for delta, usage in engine.generate_stream(message, chunks, history=oai_history, mode=mode):
        if usage:
            llm_usage = usage
        if delta:
            token_count += 1
            if first_token_time is None:
                first_token_time = time.time() - t_gen_start

            chat_history[-1] = {"role": "assistant", "content": chat_history[-1]["content"] + delta}

            elapsed = time.time() - t_gen_start
            decode_time = elapsed - first_token_time
            tps = token_count / decode_time if decode_time > 0 else 0
            stages[-1] = (f"{gen_step}. LLM generation", f"TTFT {first_token_time:.1f}s | {token_count} tok @ {tps:.1f} tok/s")
            all_chats[current_idx]["messages"] = chat_history
            yield "", chat_history, format_stats(stages), all_chats, current_idx, gr.update()

    source_line = f"\n\n---\n*Sources: {', '.join(sources)}*"
    chat_history[-1] = {"role": "assistant", "content": chat_history[-1]["content"] + source_line}

    _build_final_stats(stages, llm_usage, token_count, first_token_time, t_gen_start, t_start)
    all_chats[current_idx]["messages"] = chat_history
    yield "", chat_history, format_stats(stages), all_chats, current_idx, gr.update()


# ── Styling ──────────────────────────────────────────────────────────────────

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.emerald,
    neutral_hue=gr.themes.colors.neutral,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
    radius_size=gr.themes.sizes.radius_lg,
).set(
    body_background_fill="#1f1f1f",
    body_background_fill_dark="#1f1f1f",
    background_fill_primary="#242424",
    background_fill_primary_dark="#242424",
    background_fill_secondary="#2b2b2b",
    background_fill_secondary_dark="#2b2b2b",
    border_color_primary="#3a3a3a",
    border_color_primary_dark="#3a3a3a",
    body_text_color="#f5f5f5",
    body_text_color_dark="#f5f5f5",
    color_accent_soft="#10b981",
    button_primary_background_fill="*primary_500",
    button_primary_background_fill_hover="*primary_400",
    button_primary_text_color="#ffffff",
    block_background_fill="#242424",
    block_background_fill_dark="#242424",
    block_border_width="1px",
    block_radius="14px",
    input_background_fill="#2b2b2b",
    input_background_fill_dark="#2b2b2b",
    input_border_color="#3a3a3a",
    input_border_color_dark="#3a3a3a",
)

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --mylabs-primary: oklch(0.6929 0.1396 166.5513);
    --mylabs-primary-soft: oklch(0.6929 0.1396 166.5513 / 0.15);
    --mylabs-bg: oklch(0.24 0 0);
    --mylabs-card: oklch(0.28 0 0);
    --mylabs-sidebar: oklch(0.21 0 0);
    --mylabs-border: oklch(0.38 0 0);
    --mylabs-muted: oklch(0.33 0 0);
    --mylabs-muted-fg: oklch(0.72 0 0);
    --mylabs-radius: 14px;
    --mylabs-radius-lg: 18px;
    --mylabs-font-heading: 'Space Grotesk', 'Inter', ui-sans-serif, system-ui, sans-serif;
}

html, body, .gradio-container {
    background: var(--mylabs-bg) !important;
    letter-spacing: -0.01em;
}

.gradio-container {
    max-width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
}

.gradio-container > .main, .gradio-container .main {
    gap: 0 !important;
    padding: 0 !important;
}

footer, .footer { display: none !important; }

/* ── Header bar ─────────────────────────────────────────────────────────── */
.app-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 14px 22px;
    border-bottom: 1px solid var(--mylabs-border);
    background: var(--mylabs-sidebar);
}
.app-header .logo {
    width: 36px;
    height: 36px;
    border-radius: 10px;
    object-fit: cover;
    border: 1px solid var(--mylabs-border);
}
.app-header .brand-text { display: flex; flex-direction: column; line-height: 1.15; }
.app-header .brand-title {
    font-family: var(--mylabs-font-heading);
    font-size: 16px;
    font-weight: 600;
    letter-spacing: -0.02em;
    color: #f5f5f5;
}
.app-header .brand-sub {
    font-size: 12px;
    color: var(--mylabs-muted-fg);
    margin-top: 2px;
}
.app-header .header-spacer { flex: 1; }
.app-header .status-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 5px 12px;
    background: var(--mylabs-primary-soft);
    color: var(--mylabs-primary);
    border-radius: 999px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    border: 1px solid color-mix(in oklch, var(--mylabs-primary) 30%, transparent);
}
.app-header .status-pill::before {
    content: "";
    width: 6px; height: 6px;
    border-radius: 999px;
    background: var(--mylabs-primary);
    box-shadow: 0 0 0 3px color-mix(in oklch, var(--mylabs-primary) 25%, transparent);
}
.app-header .ctx-chip {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 11px;
    color: var(--mylabs-muted-fg);
    background: var(--mylabs-card);
    padding: 5px 10px;
    border-radius: 8px;
    border: 1px solid var(--mylabs-border);
}

/* ── Section labels ─────────────────────────────────────────────────────── */
.section-label {
    font-family: var(--mylabs-font-heading);
    font-size: 10.5px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: var(--mylabs-muted-fg);
    padding: 14px 4px 6px 4px;
}

/* ── Column wrappers (sidebars vs main) ─────────────────────────────────── */
.left-col {
    background: var(--mylabs-sidebar) !important;
    border-right: 1px solid var(--mylabs-border);
    min-height: calc(100vh - 70px);
    padding: 14px !important;
}
.right-col {
    background: var(--mylabs-sidebar) !important;
    border-left: 1px solid var(--mylabs-border);
    min-height: calc(100vh - 70px);
    padding: 14px !important;
}
.main-col {
    padding: 20px 24px !important;
    min-height: calc(100vh - 70px);
}

/* ── Blocks look like cards ─────────────────────────────────────────────── */
.gradio-container .block,
.gradio-container .form {
    background: var(--mylabs-card) !important;
    border: 1px solid var(--mylabs-border) !important;
    border-radius: var(--mylabs-radius) !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.15);
}
.left-col .block, .right-col .block {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* ── Chatbot ────────────────────────────────────────────────────────────── */
.chatbot-main {
    border: 1px solid var(--mylabs-border) !important;
    border-radius: var(--mylabs-radius-lg) !important;
    background: var(--mylabs-card) !important;
    box-shadow: 0 6px 14px rgba(0,0,0,0.12), 0 18px 40px rgba(0,0,0,0.12);
}
.chatbot-main .message,
.chatbot-main .message-wrap {
    background: transparent !important;
}
.chatbot-main .bot, .chatbot-main [data-testid="bot"] {
    background: var(--mylabs-muted) !important;
    border-radius: 12px !important;
}
.chatbot-main .user, .chatbot-main [data-testid="user"] {
    background: var(--mylabs-primary-soft) !important;
    border: 1px solid color-mix(in oklch, var(--mylabs-primary) 25%, transparent) !important;
    border-radius: 12px !important;
}

/* ── Composer ───────────────────────────────────────────────────────────── */
.composer-surface {
    padding: 10px !important;
    border: 1px solid var(--mylabs-border) !important;
    border-radius: var(--mylabs-radius-lg) !important;
    background: var(--mylabs-card) !important;
    box-shadow:
        0 1px 2px rgba(0,0,0,0.12),
        0 6px 14px rgba(0,0,0,0.10),
        0 18px 40px rgba(0,0,0,0.10);
}
.composer-surface textarea {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    font-size: 14.5px !important;
    min-height: 44px !important;
    resize: none !important;
}
.composer-surface textarea:focus { outline: none !important; box-shadow: none !important; }

/* ── Buttons ────────────────────────────────────────────────────────────── */
button.primary, .gr-button-primary,
button[variant="primary"] {
    background: var(--mylabs-primary) !important;
    color: #ffffff !important;
    border: none !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em !important;
    border-radius: 10px !important;
    transition: filter 150ms ease;
}
button.primary:hover, .gr-button-primary:hover { filter: brightness(1.08); }

button.stop, .gr-button-stop {
    background: oklch(0.6368 0.2078 25.3313) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
}
button.secondary, .gr-button-secondary {
    background: var(--mylabs-muted) !important;
    color: #f5f5f5 !important;
    border: 1px solid var(--mylabs-border) !important;
    border-radius: 10px !important;
}

/* ── Inputs ─────────────────────────────────────────────────────────────── */
input, textarea, select, .wrap .input-container {
    background: var(--mylabs-card) !important;
    color: #f5f5f5 !important;
    border-color: var(--mylabs-border) !important;
    border-radius: 10px !important;
    font-family: 'Inter', ui-sans-serif, sans-serif !important;
}
input:focus, textarea:focus, select:focus {
    border-color: var(--mylabs-primary) !important;
    box-shadow: 0 0 0 3px color-mix(in oklch, var(--mylabs-primary) 22%, transparent) !important;
}

/* ── Chat history radio as list items ───────────────────────────────────── */
.chat-history-list .wrap {
    display: flex !important;
    flex-direction: column !important;
    gap: 2px !important;
    max-height: calc(100vh - 280px);
    overflow-y: auto;
}
.chat-history-list .wrap label {
    background: transparent !important;
    border: 1px solid transparent !important;
    border-radius: 10px !important;
    padding: 9px 11px !important;
    margin: 0 !important;
    cursor: pointer;
    color: #e5e5e5 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    transition: background 120ms ease;
    width: 100%;
    display: flex !important;
    align-items: center;
    gap: 10px;
    min-height: 34px !important;
}
.chat-history-list .wrap label:hover { background: var(--mylabs-muted) !important; }
.chat-history-list .wrap label:has(input:checked) {
    background: var(--mylabs-muted) !important;
    border-color: var(--mylabs-border) !important;
    color: #ffffff !important;
}
.chat-history-list .wrap label input[type="radio"] { display: none !important; }

/* ── Mode pill selector ─────────────────────────────────────────────────── */
.mode-pills .wrap {
    display: inline-flex !important;
    flex-direction: row !important;
    gap: 4px !important;
    padding: 4px !important;
    background: var(--mylabs-muted) !important;
    border-radius: 10px !important;
    border: 1px solid var(--mylabs-border) !important;
    flex-wrap: wrap !important;
}
.mode-pills .wrap label {
    background: transparent !important;
    border: none !important;
    border-radius: 7px !important;
    padding: 5px 12px !important;
    margin: 0 !important;
    cursor: pointer;
    color: var(--mylabs-muted-fg) !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em !important;
    transition: all 150ms ease;
}
.mode-pills .wrap label:hover { color: #f5f5f5 !important; }
.mode-pills .wrap label:has(input:checked) {
    background: var(--mylabs-card) !important;
    color: var(--mylabs-primary) !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.2);
}
.mode-pills .wrap label input[type="radio"] { display: none !important; }

/* ── Pipeline stats panel ───────────────────────────────────────────────── */
.stats-panel textarea {
    font-family: 'JetBrains Mono', ui-monospace, monospace !important;
    font-size: 11.5px !important;
    line-height: 1.55 !important;
    background: var(--mylabs-sidebar) !important;
    color: var(--mylabs-muted-fg) !important;
    border: 1px solid var(--mylabs-border) !important;
    border-radius: 10px !important;
}

/* ── Accordion styling ──────────────────────────────────────────────────── */
.gradio-container .accordion {
    background: var(--mylabs-card) !important;
    border: 1px solid var(--mylabs-border) !important;
    border-radius: var(--mylabs-radius) !important;
    overflow: hidden;
}
.gradio-container .accordion > button,
.gradio-container .accordion .label-wrap {
    font-family: var(--mylabs-font-heading) !important;
    font-size: 12.5px !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    padding: 12px 14px !important;
    color: #f5f5f5 !important;
}

/* ── File upload ────────────────────────────────────────────────────────── */
.gradio-container .file-preview, .gradio-container [data-testid="file"] {
    background: var(--mylabs-card) !important;
    border: 1.5px dashed var(--mylabs-border) !important;
    border-radius: var(--mylabs-radius) !important;
}

/* ── Scrollbars ─────────────────────────────────────────────────────────── */
* {
    scrollbar-width: thin;
    scrollbar-color: oklch(0.5 0 0 / 0.5) transparent;
}
*::-webkit-scrollbar { width: 8px; height: 8px; }
*::-webkit-scrollbar-track { background: transparent; }
*::-webkit-scrollbar-thumb {
    background: oklch(0.5 0 0 / 0.5);
    border-radius: 999px;
}

/* Tighten inner label margins */
label > span { font-weight: 500 !important; color: var(--mylabs-muted-fg) !important; font-size: 12px !important; letter-spacing: 0.02em !important; }

/* Remove extra Gradio block paddings inside sidebars */
.left-col .form, .right-col .form { padding: 0 !important; }
"""


HEADER_HTML = f"""
<div class="app-header">
  <img class="logo" src="{LOGO_URI}" alt="MyLabs LLC" />
  <div class="brand-text">
    <span class="brand-title">MyLabs Studio</span>
    <span class="brand-sub">Nemotron-3-Nano RAG · 2× RTX 3090</span>
  </div>
  <div class="header-spacer"></div>
  <span class="ctx-chip">Q8_0 · {LLM_CTX_SIZE // 1024}K ctx</span>
  <span class="status-pill">Online</span>
</div>
"""


# ── UI Layout ────────────────────────────────────────────────────────────────

with gr.Blocks(title="MyLabs Studio — Nemotron RAG", fill_height=True) as app:
    gr.HTML(HEADER_HTML)

    all_chats = gr.State([make_empty_chat()])
    current_idx = gr.State(0)

    with gr.Row(equal_height=False):
        # ── Left: Chats ─────────────────────────────────────────────────
        with gr.Column(scale=1, min_width=230, elem_classes="left-col"):
            new_chat_btn = gr.Button("+ New Chat", variant="primary", size="sm")
            gr.HTML('<div class="section-label">Conversations</div>')
            chat_list = gr.Radio(
                choices=["1. New Chat"],
                value="1. New Chat",
                label="",
                interactive=True,
                container=False,
                elem_classes="chat-history-list",
                show_label=False,
            )
            gr.HTML('<div style="flex:1"></div>')
            delete_chat_btn = gr.Button("Delete Chat", variant="stop", size="sm")

        # ── Middle: Chat ────────────────────────────────────────────────
        with gr.Column(scale=4, elem_classes="main-col"):
            chatbot = gr.Chatbot(
                height=560,
                label="",
                show_label=False,
                elem_classes="chatbot-main",
                avatar_images=(None, LOGO_PATH if os.path.isfile(LOGO_PATH) else None),
                render_markdown=True,
            )

            with gr.Row(equal_height=True):
                mode_selector = gr.Radio(
                    choices=["RAG Only", "RAG + Model", "Model Only"],
                    value="RAG Only",
                    label="",
                    show_label=False,
                    container=False,
                    elem_classes="mode-pills",
                    scale=0,
                )

            with gr.Group(elem_classes="composer-surface"):
                with gr.Row(equal_height=True):
                    msg = gr.Textbox(
                        placeholder="Ask a question about your documents…",
                        label="",
                        show_label=False,
                        container=False,
                        scale=6,
                        lines=1,
                        max_lines=6,
                        autofocus=True,
                    )
                    submit = gr.Button("Send", variant="primary", scale=0, min_width=90)

            with gr.Accordion("Pipeline stats", open=False):
                stats = gr.Textbox(
                    label="",
                    show_label=False,
                    interactive=False,
                    lines=6,
                    max_lines=14,
                    elem_classes="stats-panel",
                    container=False,
                )

        # ── Right: Datasets + Discovery ─────────────────────────────────
        with gr.Column(scale=2, min_width=320, elem_classes="right-col"):
            gr.HTML('<div class="section-label">Dataset</div>')

            dataset_dropdown = gr.Dropdown(
                choices=get_dataset_choices(),
                label="Active",
                interactive=True,
                allow_custom_value=True,
                container=True,
            )
            dataset_info_box = gr.Textbox(
                label="",
                show_label=False,
                interactive=False,
                lines=5,
                max_lines=12,
                container=False,
                elem_classes="stats-panel",
            )

            with gr.Accordion("New Dataset", open=False):
                new_dataset_name = gr.Textbox(label="Name", placeholder="e.g. autoimmune")
                create_btn = gr.Button("Create", variant="primary", size="sm")

            with gr.Accordion("Upload Documents", open=True):
                file_upload = gr.File(
                    label="Drop PDF / TXT / MD / DOCX",
                    file_count="multiple",
                    file_types=[".pdf", ".txt", ".md", ".docx", ".csv", ".json"],
                    height=120,
                )
                upload_btn = gr.Button("Upload & Ingest", variant="primary", size="sm")
                upload_status = gr.Textbox(
                    label="",
                    show_label=False,
                    interactive=False,
                    lines=2,
                    max_lines=4,
                    container=False,
                    elem_classes="stats-panel",
                )
                with gr.Row():
                    reindex_btn = gr.Button("Re-index", variant="secondary", size="sm")
                    delete_btn = gr.Button("Delete", variant="stop", size="sm")

            with gr.Accordion("Discover Papers (arXiv)", open=False):
                discover_topic = gr.Textbox(
                    label="Topic",
                    placeholder="e.g. particle physics Higgs boson",
                )
                with gr.Row():
                    discover_max = gr.Slider(5, 100, value=30, step=5, label="Max papers")
                    discover_queries = gr.Slider(2, 16, value=8, step=1, label="Queries")
                discover_btn = gr.Button("Search & Download", variant="primary", size="sm")
                discover_log = gr.Textbox(
                    label="",
                    show_label=False,
                    interactive=False,
                    lines=10,
                    max_lines=20,
                    container=False,
                    elem_classes="stats-panel",
                )

    # ── Wire events ─────────────────────────────────────────────────────
    new_chat_btn.click(
        on_new_chat, [all_chats, current_idx],
        [all_chats, current_idx, chatbot, stats, chat_list],
    )
    chat_list.change(
        on_select_chat, [chat_list, all_chats],
        [all_chats, current_idx, chatbot, stats, chat_list],
    )
    delete_chat_btn.click(
        on_delete_chat, [all_chats, current_idx],
        [all_chats, current_idx, chatbot, stats, chat_list],
    )

    submit.click(
        respond_streaming, [msg, chatbot, all_chats, current_idx, mode_selector],
        [msg, chatbot, stats, all_chats, current_idx, chat_list],
    )
    msg.submit(
        respond_streaming, [msg, chatbot, all_chats, current_idx, mode_selector],
        [msg, chatbot, stats, all_chats, current_idx, chat_list],
    )

    dataset_dropdown.change(on_select_dataset, [dataset_dropdown], [dataset_info_box, dataset_dropdown])
    create_btn.click(
        on_create_dataset, [new_dataset_name],
        [upload_status, dataset_dropdown, new_dataset_name],
    )
    upload_btn.click(
        on_upload_files, [file_upload, dataset_dropdown],
        [upload_status, dataset_info_box],
    )
    reindex_btn.click(
        on_reindex_dataset, [dataset_dropdown],
        [upload_status, dataset_info_box],
    )
    delete_btn.click(
        on_delete_dataset, [dataset_dropdown],
        [upload_status, dataset_dropdown, dataset_info_box],
    )
    discover_btn.click(
        on_discover_papers,
        [discover_topic, dataset_dropdown, discover_max, discover_queries],
        [discover_log, dataset_info_box],
    )

    app.load(on_page_load, [], [dataset_dropdown, dataset_info_box])


if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=THEME,
        css=CSS,
        favicon_path=LOGO_PATH if os.path.isfile(LOGO_PATH) else None,
        allowed_paths=[os.path.join(APP_DIR, "static")],
    )
