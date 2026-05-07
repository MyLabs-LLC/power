"""React/FastAPI UI for MyLabs Studio RAG inference."""

from __future__ import annotations

import json
import html
import os
import queue
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import DOCUMENTS_DIR, LLM_CTX_SIZE, LLM_MODEL, MAX_TOKENS, TEMPERATURE, TOP_P
from discover import discover_and_download, generate_queries, title_to_filename
from ingest import ingest_dataset_streaming, iter_pdf_page_texts, load_file
from rag import RAGEngine
from rdbms import (
    build_rdbms,
    check_readonly_sql,
    execute_readonly_sql,
    extract_sql,
    read_profile,
    rdbms_exists,
    rdbms_info,
    rows_markdown,
    schema_summary,
    validate_readonly_sql,
)

APP_DIR = Path(__file__).resolve().parent
WEB_DIR = APP_DIR / "web"
STATIC_DIR = APP_DIR / "static"
LOGO_PATH = STATIC_DIR / "mylabs-logo.png"
SUMMARY_FILENAME = ".dataset_summary.md"

engine = RAGEngine()
engine_lock = threading.Lock()
active_dataset: str | None = None

app = FastAPI(title="MyLabs Studio", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/ui", StaticFiles(directory=WEB_DIR), name="ui")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class DatasetCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)


class DatasetSelect(BaseModel):
    name: str


class ChatRequest(BaseModel):
    message: str
    mode: str = "RAG + Model"
    dataset: str | None = None
    history: list[dict[str, Any]] = Field(default_factory=list)


class DiscoveryRequest(BaseModel):
    topic: str
    dataset: str
    max_papers: int = 15
    num_queries: int = 8


class DatasetSummaryRequest(BaseModel):
    max_chunks_per_call: int = 18


def clean_dataset_name(name: str) -> str:
    """Normalize dataset names for safe folder and Chroma collection use."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    cleaned = cleaned.strip("._-")
    if not cleaned:
        raise HTTPException(status_code=400, detail="Dataset name is required.")
    return cleaned[:80]


def dataset_names() -> list[str]:
    os.makedirs(DOCUMENTS_DIR, exist_ok=True)
    return sorted(
        d
        for d in os.listdir(DOCUMENTS_DIR)
        if os.path.isdir(os.path.join(DOCUMENTS_DIR, d)) and not d.startswith(".")
    )


def collection_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for collection in engine.list_collections():
        counts[collection["name"]] = int(collection["chunks"])
    return counts


def dataset_detail(name: str | None) -> dict[str, Any] | None:
    if not name:
        return None
    doc_dir = Path(DOCUMENTS_DIR) / name
    if not doc_dir.is_dir():
        return None
    files = sorted(p.name for p in doc_dir.iterdir() if p.is_file() and not p.name.startswith("."))
    summary_file = summary_path(name)
    with engine_lock:
        chunks = engine.switch_dataset(name)
    rdbms = rdbms_info(name)
    return {
        "name": name,
        "files": files,
        "file_count": len(files),
        "chunks": chunks,
        "summary_available": summary_file.is_file(),
        "summary_updated": summary_file.stat().st_mtime if summary_file.is_file() else None,
        "rdbms_available": rdbms.get("available", False),
        "rdbms_updated": rdbms.get("updated"),
        "rdbms_domain": rdbms.get("profile", {}).get("domain") if rdbms.get("available") else None,
    }


def dataset_summary() -> list[dict[str, Any]]:
    counts = collection_counts()
    return [
        {
            "name": name,
            "file_count": len(
                [
                    p
                    for p in (Path(DOCUMENTS_DIR) / name).iterdir()
                    if p.is_file() and not p.name.startswith(".")
                ]
            ),
            "chunks": counts.get(name.lower().replace(" ", "_"), 0),
            "summary_available": summary_path(name).is_file(),
            "rdbms_available": rdbms_exists(name),
        }
        for name in dataset_names()
    ]


def set_active_dataset(name: str | None) -> dict[str, Any] | None:
    global active_dataset
    if not name:
        active_dataset = None
        engine.collection = None
        return None
    normalized = clean_dataset_name(name)
    if normalized not in dataset_names():
        raise HTTPException(status_code=404, detail=f"Dataset '{normalized}' not found.")
    active_dataset = normalized
    return dataset_detail(normalized)


def summary_path(dataset: str) -> Path:
    return Path(DOCUMENTS_DIR) / dataset / SUMMARY_FILENAME


def read_dataset_summary(dataset: str) -> str | None:
    path = summary_path(dataset)
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8", errors="ignore")


def write_dataset_summary(dataset: str, content: str) -> None:
    path = summary_path(dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def stream_worker(
    work,
    *,
    finish=None,
    heartbeat_message: str = "Still processing...",
    heartbeat_seconds: float = 5.0,
):
    progress_q: queue.Queue[tuple[str, dict[str, Any]]] = queue.Queue()
    result_holder: dict[str, Any] = {}

    def emit(event: str, data: dict[str, Any]) -> None:
        progress_q.put((event, data))

    def run() -> None:
        try:
            result_holder["result"] = work(emit)
        except Exception as exc:
            result_holder["error"] = str(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    last_heartbeat = time.time()

    while thread.is_alive():
        emitted = False
        try:
            while True:
                event, data = progress_q.get_nowait()
                emitted = True
                yield sse(event, data)
        except queue.Empty:
            pass

        now = time.time()
        if heartbeat_message and now - last_heartbeat >= heartbeat_seconds:
            last_heartbeat = now
            yield sse("log", {"message": heartbeat_message})
        elif emitted:
            last_heartbeat = now

        time.sleep(0.25)

    thread.join()

    try:
        while True:
            event, data = progress_q.get_nowait()
            yield sse(event, data)
    except queue.Empty:
        pass

    if result_holder.get("error"):
        yield sse("error", {"message": result_holder["error"]})
        return

    if finish:
        yield from finish(result_holder.get("result"))


def stream_ingestion(
    dataset: str,
    paths: list[str],
    *,
    reset: bool,
    intro_messages: list[str] | None = None,
    done_data: dict[str, Any] | None = None,
):
    def work(emit) -> None:
        for message in intro_messages or []:
            emit("log", {"message": message})
        for update in ingest_dataset_streaming(dataset, paths, reset=reset):
            emit("log", {"message": update})
        if paths:
            try:
                build_dataset_summary(dataset, emit)
            except Exception as exc:
                emit("log", {"message": f"Summary cache failed: {exc}"})

    def finish(_result):
        if not paths:
            return
        set_active_dataset(dataset)
        payload = {"datasets": dataset_summary(), "dataset_detail": dataset_detail(dataset)}
        if done_data:
            payload.update(done_data)
        yield sse("done", payload)

    yield from stream_worker(work, finish=finish, heartbeat_message="Still ingesting or summarizing documents...")


def format_stats(stages: list[tuple[str, str]]) -> list[dict[str, str]]:
    return [{"label": label, "value": value} for label, value in stages]


def safe_document_path(dataset: str, source: str) -> Path:
    source_name = Path(source).name
    if not source_name or source_name.startswith("."):
        raise HTTPException(status_code=404, detail="Document not found.")
    path = Path(DOCUMENTS_DIR) / dataset / source_name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Document not found.")
    return path


def find_document_path(source: str) -> tuple[str, Path]:
    source_name = Path(source).name
    if not source_name or source_name.startswith("."):
        raise HTTPException(status_code=404, detail="Document not found.")

    matches: list[tuple[str, Path]] = []
    for dataset in dataset_names():
        path = Path(DOCUMENTS_DIR) / dataset / source_name
        if path.is_file():
            matches.append((dataset, path))

    if not matches:
        raise HTTPException(status_code=404, detail="Document not found.")
    if len(matches) > 1:
        raise HTTPException(status_code=409, detail=f"Document name exists in multiple datasets: {source_name}")
    return matches[0]


def document_text_url(dataset: str, source: str, page: int | None = None, chunk_index: int | None = None) -> str:
    source_part = quote(Path(source).name, safe="")
    chunk_query = f"?chunk={int(chunk_index)}" if chunk_index is not None else ""
    if page:
        return f"/api/documents/{source_part}/pages/{int(page)}{chunk_query}"
    return f"/api/documents/{source_part}/text{chunk_query}"


def format_source_ref(dataset: str, chunk: dict[str, Any]) -> dict[str, Any]:
    source = Path(chunk.get("source", "unknown")).name
    page = chunk.get("page")
    chunk_index = chunk.get("chunk_index")
    label = f"{source} p. {page}" if page else source
    if chunk_index is not None:
        label = f"{label} (chunk {chunk_index})"
    return {
        "label": label,
        "url": document_text_url(dataset, source, page, chunk_index),
        "source": source,
        "page": page,
        "chunk_index": chunk_index,
    }


def format_source_refs(dataset: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    refs: dict[tuple[str, int | None], dict[str, Any]] = {}
    for chunk in chunks:
        source = Path(chunk.get("source", "unknown")).name
        page = chunk.get("page")
        key = (source, page)
        chunk_index = chunk.get("chunk_index")
        if key not in refs:
            refs[key] = format_source_ref(dataset, chunk)
            refs[key]["chunk_indexes"] = []
        if chunk_index is not None:
            refs[key]["chunk_indexes"].append(chunk_index)

    for ref in refs.values():
        indexes = sorted({idx for idx in ref.pop("chunk_indexes", []) if idx is not None})
        if indexes:
            chunks_label = ", ".join(str(idx) for idx in indexes)
            base = f"{ref['source']} p. {ref['page']}" if ref.get("page") else ref["source"]
            ref["label"] = f"{base} (chunks {chunks_label})"

    return sorted(
        refs.values(),
        key=lambda item: (item.get("source") or "", item.get("page") or 0, item.get("chunk_index") or -1),
    )


def get_indexed_chunk_text(dataset: str, source: str, page: int | None, chunk_index: int | None) -> str | None:
    if chunk_index is None:
        return None
    with engine_lock:
        grouped = engine.all_chunks_by_source(dataset)
    source_name = Path(source).name
    chunk_match = None
    for chunk in grouped.get(source_name, []):
        if chunk.get("chunk_index") != chunk_index:
            continue
        chunk_match = chunk
        if page is not None and chunk.get("page") != page:
            continue
        text = chunk.get("text") or ""
        return text.strip() or None
    if chunk_match:
        text = chunk_match.get("text") or ""
        return text.strip() or None
    return None


def highlight_text_html(text: str, highlight: str | None) -> tuple[str, bool]:
    if not highlight:
        return html.escape(text or "No extractable text found."), False
    haystack = text or ""
    needle = highlight.strip()
    if not needle:
        return html.escape(haystack or "No extractable text found."), False

    index = haystack.find(needle)
    if index < 0:
        return html.escape(haystack or "No extractable text found."), False

    before = html.escape(haystack[:index])
    matched = html.escape(haystack[index : index + len(needle)])
    after = html.escape(haystack[index + len(needle) :])
    return f'{before}<mark id="reference">{matched}</mark>{after}', True


def render_text_document(title: str, subtitle: str, text: str, highlight: str | None = None) -> HTMLResponse:
    escaped_title = html.escape(title)
    escaped_subtitle = html.escape(subtitle)
    rendered_text, highlight_found = highlight_text_html(text, highlight)
    highlight_block = ""
    if highlight and not highlight_found:
        highlight_block = (
            '<section class="reference-block" id="reference">'
            "<h2>Cited reference text</h2>"
            f"<pre>{html.escape(highlight)}</pre>"
            "</section>"
        )
    return HTMLResponse(
        f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escaped_title}</title>
  <style>
    body {{ margin: 0; background: #f7f7f4; color: #181a17; font: 16px/1.58 system-ui, sans-serif; }}
    header {{ position: sticky; top: 0; background: #ffffff; border-bottom: 1px solid #d8ddd3; padding: 16px 22px; }}
    h1 {{ font-size: 18px; margin: 0 0 4px; }}
    .subtitle {{ color: #5f685b; font-size: 13px; }}
    main {{ max-width: 980px; margin: 0 auto; padding: 24px; }}
    mark {{ background: #fff176; color: inherit; padding: 2px 0; }}
    .reference-block {{ background: #fff9bf; border: 1px solid #e4cd46; margin: 0 0 22px; padding: 14px 16px; }}
    .reference-block h2 {{ font-size: 14px; margin: 0 0 8px; }}
    pre {{ white-space: pre-wrap; word-wrap: break-word; font: 14px/1.65 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
  </style>
</head>
<body>
  <header>
    <h1>{escaped_title}</h1>
    <div class="subtitle">{escaped_subtitle}</div>
  </header>
  <main>{highlight_block}<pre>{rendered_text}</pre></main>
  <script>document.getElementById("reference")?.scrollIntoView({{ block: "center" }});</script>
</body>
</html>"""
    )


def render_unavailable_page_document(
    dataset: str,
    path: Path,
    requested_page: int,
    total_pages: int,
    text: str,
    highlight: str | None,
) -> HTMLResponse:
    subtitle = (
        f"{dataset} · requested page {requested_page} is unavailable; "
        f"showing full extracted text from {total_pages} pages"
    )
    return render_text_document(path.name, subtitle, text, highlight)


@app.get("/api/datasets/{name}/documents/{source}/pages/{page}")
def get_document_page_text(name: str, source: str, page: int, chunk: int | None = None) -> HTMLResponse:
    dataset = clean_dataset_name(name)
    if page < 1:
        raise HTTPException(status_code=400, detail="Page must be 1 or greater.")
    path = safe_document_path(dataset, source)
    return document_page_response(dataset, path, page, chunk)


def document_page_response(dataset: str, path: Path, page: int, chunk: int | None = None) -> HTMLResponse:
    highlight = get_indexed_chunk_text(dataset, path.name, page, chunk)
    if path.suffix.lower() != ".pdf":
        if page != 1:
            return render_unavailable_page_document(dataset, path, page, 1, load_file(str(path)), highlight)
        return render_text_document(path.name, f"{dataset} · full text", load_file(str(path)), highlight)

    pages = list(iter_pdf_page_texts(str(path)))
    for _extractor, total_pages, page_num, text in pages:
        if page_num == page:
            return render_text_document(path.name, f"{dataset} · page {page} of {total_pages}", text, highlight)

    total_pages = pages[-1][1] if pages else 0
    full_text = "\n\n".join(text for _extractor, _total_pages, _page_num, text in pages if text.strip())
    if not full_text:
        full_text = load_file(str(path))
    return render_unavailable_page_document(dataset, path, page, total_pages, full_text, highlight)


@app.get("/api/datasets/{name}/documents/{source}/text")
def get_document_text(name: str, source: str, chunk: int | None = None) -> HTMLResponse:
    dataset = clean_dataset_name(name)
    path = safe_document_path(dataset, source)
    highlight = get_indexed_chunk_text(dataset, path.name, None, chunk)
    return render_text_document(path.name, f"{dataset} · full text", load_file(str(path)), highlight)


@app.get("/api/documents/{source}/pages/{page}")
def get_document_page_text_global(source: str, page: int, chunk: int | None = None) -> HTMLResponse:
    if page < 1:
        raise HTTPException(status_code=400, detail="Page must be 1 or greater.")
    dataset, path = find_document_path(source)
    return document_page_response(dataset, path, page, chunk)


@app.get("/api/documents/{source}/text")
def get_document_text_global(source: str, chunk: int | None = None) -> HTMLResponse:
    dataset, path = find_document_path(source)
    highlight = get_indexed_chunk_text(dataset, path.name, None, chunk)
    return render_text_document(path.name, f"{dataset} · full text", load_file(str(path)), highlight)


def trim_incomplete_tail(text: str) -> str:
    text = text.strip()
    if not text or text.endswith((".", "!", "?", "```")):
        return text
    sentence_end = max(text.rfind(". "), text.rfind("! "), text.rfind("? "))
    if sentence_end > max(120, len(text) // 2):
        return text[: sentence_end + 1].strip()
    return text


def summary_fallback(source: str) -> str:
    return f"No substantive summary could be generated from the indexed excerpts for {source}."
    if page:
        return f"{chunk['source']} p. {page}"
    return chunk["source"]


def build_final_stats(
    stages: list[tuple[str, str]],
    llm_usage: dict[str, int] | None,
    token_count: int,
    first_token_time: float | None,
    t_gen_start: float,
    t_start: float,
) -> None:
    t_gen_total = time.time() - t_gen_start
    t_total = time.time() - t_start
    ttft = first_token_time or 0
    decode_time = t_gen_total - ttft
    tps = token_count / decode_time if decode_time > 0 else 0
    prompt_tokens = llm_usage["prompt_tokens"] if llm_usage else "?"
    completion_tokens = llm_usage["completion_tokens"] if llm_usage else token_count
    prefill_tps = f"{prompt_tokens / ttft:.0f} tok/s" if llm_usage and ttft > 0 else "n/a"
    ctx_usage = (
        f"{llm_usage['total_tokens']}/{LLM_CTX_SIZE} "
        f"({llm_usage['total_tokens'] * 100 / LLM_CTX_SIZE:.0f}%)"
        if llm_usage
        else "?"
    )

    stages[-1] = (stages[-1][0], f"{t_gen_total:.1f}s")
    stages.extend(
        [
            ("TTFT", f"{ttft:.2f}s"),
            ("Prefill", f"{prompt_tokens} prompt tokens @ {prefill_tps}"),
            ("Decode", f"{completion_tokens} chunks in {decode_time:.1f}s @ {tps:.1f}/s"),
            ("Context", ctx_usage),
            ("Total", f"{t_total:.1f}s"),
        ]
    )


def chunk_context(chunks: list[dict[str, Any]], max_chars: int = 18000) -> list[list[dict[str, Any]]]:
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_chars = 0
    for chunk in chunks:
        text_len = len(chunk.get("text") or "")
        if current and current_chars + text_len > max_chars:
            batches.append(current)
            current = []
            current_chars = 0
        current.append(chunk)
        current_chars += text_len
    if current:
        batches.append(current)
    return batches


def format_chunk_batch(chunks: list[dict[str, Any]]) -> str:
    parts = []
    for chunk in chunks:
        page = f", page {chunk['page']}" if chunk.get("page") else ""
        parts.append(f"[{chunk['source']}{page}, chunk {chunk.get('chunk_index', '?')}]\n{chunk.get('text', '')}")
    return "\n\n---\n\n".join(parts)


def llm_complete(system_prompt: str, user_prompt: str, max_tokens: int = 900) -> str:
    response = engine.llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=min(max_tokens, MAX_TOKENS),
        temperature=min(TEMPERATURE, 0.3),
        top_p=TOP_P,
    )
    content = (response.choices[0].message.content or "").strip()
    if not content:
        return ""
    if response.choices[0].finish_reason == "length":
        content = trim_incomplete_tail(content)
    return content


def sql_compiler_prompt(dataset: str, question: str) -> tuple[str, str]:
    profile = read_profile(dataset)
    profile_json = json.dumps(profile, ensure_ascii=False, indent=2)[:6000]
    schema = schema_summary(dataset)
    system_prompt = (
        "You are a careful SQLite query compiler for exact, source-grounded datasets. "
        "Return exactly one read-only SELECT or WITH query and no prose, no markdown, and no thinking text. "
        "Scientific, medical, and legal answers must be traceable to rows, pages, chunks, and source_url. "
        "Never invent schema fields and never write data-changing SQL."
    )
    user_prompt = f"""Dataset: {dataset}

Dataset profile:
{profile_json}

SQLite schema:
{schema}

Rules:
- Use only the tables and columns shown in the schema.
- Prefer evidence queries that include documents.source_name, pages.page_number, chunks.chunk_index, chunks.source_url, and a short text/context column.
- Join through foreign keys: documents -> pages/chunks, chunks -> terms/chunk_terms/measurements/citations.
- For text search, use lower(column) LIKE lower('%term%') and search chunks.text, pages.text, documents.title, or terms.term as appropriate.
- Only use measurements when the question explicitly asks for a number, unit, dosage, percentage, year, sample size, rate, or measured value.
- Never infer "best", "recommended", or "most effective" by sorting measurements.numeric_value. Use source evidence rows instead.
- For counts, rankings, and coverage questions, aggregate explicitly and name computed columns.
- Add LIMIT 80 unless the query is a single aggregate result.
- If the question cannot be answered exactly from structured tables, retrieve the closest source-grounded evidence rows.
- Do not use aliases in JOIN conditions before they are declared.
- Do not use ILIKE, REGEXP, JSON functions, or full-text search. This is plain SQLite.

Question:
{question}

SQL only:"""
    return system_prompt, user_prompt


def sql_repair_prompt(dataset: str, question: str, bad_sql: str, error: str) -> tuple[str, str]:
    schema = schema_summary(dataset)
    system_prompt = (
        "You repair invalid SQLite. Return exactly one corrected read-only SELECT or WITH query. "
        "Return SQL only, with no markdown and no explanation."
    )
    user_prompt = f"""Dataset: {dataset}

SQLite schema:
{schema}

Question:
{question}

The previous SQL failed:
{bad_sql}

SQLite error:
{error}

Return one corrected SQLite SELECT/WITH query only:"""
    return system_prompt, user_prompt


SQL_QUESTION_STOPWORDS = {
    "about", "after", "again", "against", "also", "among", "because", "being", "best",
    "does", "from", "have", "into", "many", "more", "most", "should", "than", "that", "the",
    "their", "there", "these", "this", "tested", "were", "what", "when", "where", "which",
    "with", "would", "how", "is", "of", "in", "on", "or", "to",
}


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def fallback_evidence_sql(question: str) -> str:
    words = [word.lower() for word in re.findall(r"[A-Za-z0-9][A-Za-z0-9-]{1,}", question)]
    keywords = [word for word in words if word not in SQL_QUESTION_STOPWORDS]
    years = [word for word in keywords if re.fullmatch(r"(?:19|20)\d{2}", word)]
    groups: list[list[str]] = []
    if years:
        groups.append(years[:3])
    if any(word in {"ra", "rheumatoid", "arthritis"} for word in keywords):
        groups.append(["rheumatoid", "arthritis"])
    if any(word in {"drug", "drugs", "medication", "medications"} for word in keywords):
        groups.append(["drug", "drugs", "medication", "medications", "treatment", "treatments", "therapy", "methotrexate", "dmard", "biologic"])
    if any(word in {"effective", "efficacy", "better", "best"} for word in words):
        groups.append(["effective", "efficacy", "response", "improvement", "superior"])

    used = {term for group in groups for term in group}
    remaining = [word for word in keywords if word not in used and not re.fullmatch(r"(?:19|20)\d{2}", word)]
    groups.extend([remaining[i : i + 3] for i in range(0, min(len(remaining), 6), 3)])
    if not groups:
        groups = [["rheumatoid", "arthritis"]]

    conditions = " AND ".join(
        "(" + " OR ".join(f"lower(c.text) LIKE '%' || lower({sql_literal(word)}) || '%'" for word in group) + ")"
        for group in groups
        if group
    )
    return f"""SELECT
  d.source_name,
  p.page_number,
  c.chunk_index,
  c.source_url,
  substr(c.text, 1, 900) AS evidence
FROM chunks c
JOIN documents d ON d.document_id = c.document_id
LEFT JOIN pages p ON p.page_id = c.page_id
WHERE {conditions}
ORDER BY d.source_name, p.page_number, c.chunk_index
LIMIT 80"""


def prefer_evidence_sql(question: str) -> bool:
    words = {word.lower() for word in re.findall(r"[A-Za-z0-9][A-Za-z0-9-]{1,}", question)}
    medication_words = {"drug", "drugs", "medication", "medications", "treatment", "therapy", "therapies"}
    judgement_words = {"best", "recommended", "recommend", "safest", "effective", "efficacy", "better"}
    tested_words = {"tested", "trial", "trials", "study", "studies"}
    has_year = any(re.fullmatch(r"(?:19|20)\d{2}", word) for word in words)
    return bool(words & medication_words and (words & judgement_words or has_year or words & tested_words))


def compile_dataset_sql(dataset: str, question: str) -> tuple[str, list[str]]:
    notes: list[str] = []
    if prefer_evidence_sql(question):
        notes.append("Used source-evidence SQL because this asks about medications, trials, or recommendations.")
        return check_readonly_sql(dataset, fallback_evidence_sql(question)), notes

    system_prompt, user_prompt = sql_compiler_prompt(dataset, question)
    generated = llm_complete(system_prompt, user_prompt, max_tokens=900)
    sql = extract_sql(generated)

    try:
        return check_readonly_sql(dataset, sql), notes
    except Exception as first_error:
        notes.append(f"Initial SQL rejected: {first_error}")
        if sql:
            repair_system, repair_user = sql_repair_prompt(dataset, question, sql, str(first_error))
            repaired = extract_sql(llm_complete(repair_system, repair_user, max_tokens=900))
            try:
                notes.append("Used repaired SQL after SQLite validation.")
                return check_readonly_sql(dataset, repaired), notes
            except Exception as repair_error:
                notes.append(f"Repair rejected: {repair_error}")

    fallback = fallback_evidence_sql(question)
    notes.append("Used conservative evidence-search SQL fallback.")
    return check_readonly_sql(dataset, fallback), notes


def format_generated_sql(sql: str, notes: list[str] | None = None) -> str:
    note_text = ""
    if notes:
        note_text = "\n".join(f"- {note}" for note in notes) + "\n\n"
    return (
        "Generated SQL\n\n"
        f"{note_text}"
        "```sql\n"
        f"{sql}\n"
        "```\n\n"
        "Running query...\n\n"
    )


def format_sql_answer(result: dict[str, Any], *, include_sql: bool = True) -> str:
    limited_note = " Results were capped at 80 rows." if result.get("limited") else ""
    sql_block = (
        "SQL RDBMS answer (exact rows from the generated database)\n\n"
        "```sql\n"
        f"{result['sql']}\n"
        "```\n\n"
    ) if include_sql else "SQL results\n\n"
    return sql_block + (
        f"Rows returned: {result['row_count']}.{limited_note}\n\n"
        f"{rows_markdown(result.get('rows', []), result.get('columns', []))}"
    )


def parse_discovery_query_plan(raw: str, topic: str, limit: int) -> list[str]:
    cleaned = re.sub(r"</?think>", " ", raw or "", flags=re.IGNORECASE)
    queries: list[str] = []

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            candidates = parsed
        elif isinstance(parsed, dict):
            candidates = parsed.get("queries") or parsed.get("searches") or []
        else:
            candidates = []
    except json.JSONDecodeError:
        candidates = []

    if not candidates:
        candidates = []
        for line in cleaned.splitlines():
            line = re.sub(r"^(?:[-*]|\d+[.)])\s*", "", line.strip())
            if line:
                candidates.append(line)

    planned: list[str] = []

    def add_query(value: Any) -> None:
        text = str(value).strip().strip("\"'")
        text = re.sub(r"\s+", " ", text)
        if not text or len(text) < 3 or len(text) > 140:
            return
        if text.lower() in {item.lower() for item in planned}:
            return
        planned.append(text)

    add_query(topic)
    for candidate in candidates:
        add_query(candidate)
        if len(planned) >= limit:
            break
    return planned[:limit]


def build_discovery_query_plan(topic: str, num_queries: int, emit=None) -> list[str]:
    limit = max(1, min(int(num_queries), 12))
    system_prompt = (
        "You are a research librarian generating arXiv search queries. "
        "Return only a JSON array of concise search strings. "
        "The first query must be the user's exact topic. "
        "Include acronym expansions, synonyms, broader terms, narrower technical terms, and question-style variants. "
        "Do not include explanations."
    )
    user_prompt = f"""Topic: {topic}

Create {limit} arXiv search queries for this topic.

Rules:
- Query 1 must be exactly: {topic}
- Make the remaining queries meaningfully different, not just adding "review" repeatedly.
- Expand domain acronyms when likely.
- Include related mechanisms, methods, treatments, measurements, datasets, legal concepts, or scientific subtopics as appropriate.
- Keep each query under 12 words.

Return JSON array only."""
    try:
        raw = llm_complete(system_prompt, user_prompt, max_tokens=700)
        queries = parse_discovery_query_plan(raw, topic, limit)
    except Exception as exc:
        if emit:
            emit(f"LLM query planner failed; using built-in expansion: {exc}")
        queries = generate_queries(topic, limit)

    if len(queries) < 2:
        for query in generate_queries(topic, limit):
            if query.lower() not in {item.lower() for item in queries}:
                queries.append(query)
            if len(queries) >= limit:
                break

    if emit:
        emit("Discovery query plan: " + " | ".join(queries))
    return queries


def summarize_document(source: str, chunks: list[dict[str, Any]], emit, max_chunks_per_call: int) -> str:
    batches = []
    for chunk_batch in chunk_context(chunks):
        for i in range(0, len(chunk_batch), max_chunks_per_call):
            batches.append(chunk_batch[i : i + max_chunks_per_call])

    partials: list[str] = []
    for idx, batch in enumerate(batches, start=1):
        if len(batches) > 1:
            emit("log", {"message": f"    part {idx}/{len(batches)} ({len(batch)} chunks)"})
        context = format_chunk_batch(batch)
        partials.append(
            llm_complete(
                "You summarize research documents from provided excerpts. Use only the provided text. "
                "Capture the research topic, methods or evidence, key findings, limitations, and notable numbers. "
                "If the excerpt is metadata or bibliography rather than article content, say that plainly. "
                "Write complete sentences, avoid tables, and stay under 220 words.",
                f"Document: {source}\n\nExcerpts:\n{context}\n\nWrite a concise, factual summary of this part.",
                max_tokens=520,
            )
        )

    if len(partials) == 1:
        return partials[0] or summary_fallback(source)

    combined = llm_complete(
        "You combine partial summaries of one research document. Use only the partial summaries. "
        "Remove duplication and preserve concrete findings. Write complete sentences, avoid tables, "
        "and stay under 280 words.",
        f"Document: {source}\n\nPartial summaries:\n" + "\n\n---\n\n".join(partials)
        + "\n\nWrite one clear summary of the full document.",
        max_tokens=680,
    )
    return combined or summary_fallback(source)


def synthesize_dataset_summary(document_summaries: list[dict[str, str]]) -> str:
    compact = "\n\n".join(f"## {item['source']}\n{item['summary']}" for item in document_summaries)
    return llm_complete(
        "You synthesize a corpus-level literature summary from per-document summaries. "
        "Identify major themes, recurring methods, important findings, disagreements, and gaps. "
        "Do not claim access to documents beyond the provided summaries. Avoid tables and stay under 900 words.",
        f"Per-document summaries:\n\n{compact}\n\nWrite an executive summary of the research corpus.",
        max_tokens=1200,
    ) or "No corpus-level summary could be generated from the document summaries."


def build_dataset_summary(dataset: str, emit, max_chunks_per_call: int = 18) -> str:
    with engine_lock:
        chunks_available = engine.switch_dataset(dataset)
        grouped = engine.all_chunks_by_source(dataset)
    if chunks_available <= 0 or not grouped:
        raise ValueError("Selected dataset is empty. Upload or re-index documents first.")

    emit("log", {"message": f"Building cached summary for {len(grouped)} documents ({chunks_available} chunks)..."})
    document_summaries: list[dict[str, str]] = []
    for idx, (source, chunks) in enumerate(grouped.items(), start=1):
        emit("log", {"message": f"[{idx}/{len(grouped)}] Summarizing {source} ({len(chunks)} chunks)"})
        summary = summarize_document(
            source,
            chunks,
            emit,
            max(1, min(int(max_chunks_per_call), 40)),
        )
        document_summaries.append({"source": source, "summary": summary})

    emit("log", {"message": "Synthesizing cached corpus summary..."})
    overview = synthesize_dataset_summary(document_summaries)
    body = [
        f"# Dataset Summary: {dataset}",
        "",
        "## Overall Themes",
        overview,
        "",
        "## Document Summaries",
    ]
    for item in document_summaries:
        body.extend(["", f"### {item['source']}", item["summary"]])

    content = "\n".join(body)
    write_dataset_summary(dataset, content)
    emit("log", {"message": f"Cached summary saved to {SUMMARY_FILENAME}."})
    return content


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/api/bootstrap")
def bootstrap() -> dict[str, Any]:
    names = dataset_names()
    selected = active_dataset or (names[0] if names else None)
    detail = set_active_dataset(selected) if selected else None
    return {
        "app": "MyLabs Studio",
        "model": "nemotron-3-nano",
        "ctx_size": LLM_CTX_SIZE,
        "logo": "/static/mylabs-logo.png" if LOGO_PATH.is_file() else "",
        "modes": ["RAG Only", "RAG + Model", "Model Only", "SQL RDBMS"],
        "datasets": dataset_summary(),
        "active_dataset": selected,
        "dataset_detail": detail,
    }


@app.get("/api/datasets")
def list_datasets() -> dict[str, Any]:
    return {
        "datasets": dataset_summary(),
        "active_dataset": active_dataset,
        "dataset_detail": dataset_detail(active_dataset),
    }


@app.post("/api/datasets")
def create_dataset(payload: DatasetCreate) -> dict[str, Any]:
    name = clean_dataset_name(payload.name)
    os.makedirs(Path(DOCUMENTS_DIR) / name, exist_ok=True)
    detail = set_active_dataset(name)
    return {"message": f"Created dataset '{name}'.", "datasets": dataset_summary(), "dataset_detail": detail}


@app.post("/api/datasets/select")
def select_dataset(payload: DatasetSelect) -> dict[str, Any]:
    detail = set_active_dataset(payload.name)
    return {"active_dataset": active_dataset, "dataset_detail": detail}


@app.delete("/api/datasets/{name}")
def delete_dataset(name: str) -> dict[str, Any]:
    normalized = clean_dataset_name(name)
    doc_dir = Path(DOCUMENTS_DIR) / normalized
    if doc_dir.is_dir():
        shutil.rmtree(doc_dir)
    with engine_lock:
        engine.delete_dataset(normalized)
        engine.collection = None

    names = dataset_names()
    detail = set_active_dataset(names[0]) if names else None
    return {"message": f"Deleted '{normalized}'.", "datasets": dataset_summary(), "dataset_detail": detail}


@app.post("/api/datasets/{name}/upload")
async def upload_files(name: str, files: list[UploadFile] = File(...)) -> StreamingResponse:
    dataset = clean_dataset_name(name)
    doc_dir = Path(DOCUMENTS_DIR) / dataset
    doc_dir.mkdir(parents=True, exist_ok=True)

    existing = {p.name for p in doc_dir.iterdir() if p.is_file() and not p.name.startswith(".")}
    paths: list[str] = []
    renames: list[str] = []

    for uploaded in files:
        original = Path(uploaded.filename or "document").name
        ext = Path(original).suffix.lower()
        new_name = title_to_filename(Path(original).stem, ext=ext, existing=existing)
        existing.add(new_name)
        dest = doc_dir / new_name
        with dest.open("wb") as out:
            while chunk := await uploaded.read(1024 * 1024):
                out.write(chunk)
        await uploaded.close()
        paths.append(str(dest))
        if new_name != original:
            renames.append(f"{original} -> {new_name}")

    def stream():
        intro = [f"Uploaded {len(paths)} file(s)."]
        intro.extend(f"Renamed: {rename}" for rename in renames)
        yield from stream_ingestion(dataset, paths, reset=False, intro_messages=intro)

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/api/datasets/{name}/reindex")
def reindex_dataset(name: str) -> StreamingResponse:
    dataset = clean_dataset_name(name)
    doc_dir = Path(DOCUMENTS_DIR) / dataset
    paths = sorted(str(p) for p in doc_dir.iterdir() if p.is_file() and not p.name.startswith(".")) if doc_dir.is_dir() else []

    def stream():
        if not paths:
            yield sse("error", {"message": f"No files in {dataset}/."})
            return
        yield from stream_ingestion(
            dataset,
            paths,
            reset=True,
            intro_messages=[f"Re-indexing {len(paths)} files..."],
        )

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/api/datasets/{name}/summary")
def get_dataset_summary(name: str) -> dict[str, Any]:
    dataset = clean_dataset_name(name)
    summary = read_dataset_summary(dataset)
    if not summary:
        raise HTTPException(status_code=404, detail="No cached summary exists yet. Re-index or upload documents to build one.")
    path = summary_path(dataset)
    return {
        "dataset": dataset,
        "summary": summary,
        "summary_updated": path.stat().st_mtime,
    }


@app.get("/api/datasets/{name}/rdbms")
def get_dataset_rdbms(name: str) -> dict[str, Any]:
    dataset = clean_dataset_name(name)
    info = rdbms_info(dataset)
    if not info.get("available"):
        return {"dataset": dataset, "available": False}
    return {"dataset": dataset, **info}


@app.post("/api/datasets/{name}/rdbms/generate")
def generate_dataset_rdbms(name: str) -> StreamingResponse:
    dataset = clean_dataset_name(name)

    def work(emit) -> dict[str, Any]:
        emit("log", {"message": f"Preparing normalized RDBMS for {dataset}..."})
        with engine_lock:
            chunks_available = engine.switch_dataset(dataset)
            grouped = engine.all_chunks_by_source(dataset)
        if chunks_available <= 0 or not grouped:
            raise ValueError("Selected dataset is empty. Upload, discover, or re-index documents first.")

        emit(
            "log",
            {"message": f"Building SQLite 3NF database from {len(grouped)} documents and {chunks_available} chunks..."},
        )
        result = build_rdbms(dataset, grouped, emit=emit)
        profile = result.get("profile", {})
        emit(
            "log",
            {
                "message": (
                    f"RDBMS ready: {result.get('documents', 0)} documents, {result.get('pages', 0)} pages, "
                    f"{result.get('chunks', 0)} chunks, domain={profile.get('domain', 'general_documents')}."
                )
            },
        )
        return result

    def finish(result):
        set_active_dataset(dataset)
        yield sse(
            "done",
            {
                "datasets": dataset_summary(),
                "dataset_detail": dataset_detail(dataset),
                "rdbms": result,
            },
        )

    return StreamingResponse(
        stream_worker(work, finish=finish, heartbeat_message="Still normalizing dataset into SQL tables..."),
        media_type="text/event-stream",
    )


@app.post("/api/datasets/{name}/summarize")
def summarize_dataset(name: str, payload: DatasetSummaryRequest | None = None) -> StreamingResponse:
    dataset = clean_dataset_name(name)

    def stream():
        summary = read_dataset_summary(dataset)
        if not summary:
            yield sse("error", {"message": "No cached summary exists yet. Re-index or upload documents to build one."})
            return

        yield sse(
            "done",
            {
                "answer": summary,
                "stats": format_stats([("Mode", "Cached dataset summary")]),
                "sources": [],
            },
        )

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/api/discover")
def discover_papers(payload: DiscoveryRequest) -> StreamingResponse:
    topic = payload.topic.strip()
    dataset = clean_dataset_name(payload.dataset)
    if not topic:
        raise HTTPException(status_code=400, detail="Discovery topic is required.")

    def stream():
        progress_q: queue.Queue[str] = queue.Queue()
        result_holder: dict[str, Any] = {}

        def progress_cb(message: str) -> None:
            progress_q.put(message)

        def run() -> None:
            query_plan = build_discovery_query_plan(
                topic,
                int(payload.num_queries),
                emit=progress_cb,
            )
            result_holder["stats"] = discover_and_download(
                topic=topic,
                dataset_name=dataset,
                max_papers=int(payload.max_papers),
                num_queries=int(payload.num_queries),
                query_plan=query_plan,
                progress_cb=progress_cb,
            )

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        while thread.is_alive():
            try:
                while True:
                    yield sse("log", {"message": progress_q.get_nowait()})
            except queue.Empty:
                pass
            time.sleep(0.25)
        thread.join()

        try:
            while True:
                yield sse("log", {"message": progress_q.get_nowait()})
        except queue.Empty:
            pass

        stats = result_holder.get("stats", {})
        yield sse("log", {"message": f"Downloaded {stats.get('downloaded', 0)} of {stats.get('searched', 0)} papers."})
        if stats.get("downloaded", 0):
            doc_dir = Path(DOCUMENTS_DIR) / dataset
            paths = sorted(str(p) for p in doc_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")
            yield from stream_ingestion(
                dataset,
                paths,
                reset=True,
                intro_messages=["Ingesting downloaded papers..."],
                done_data={"stats": stats},
            )
            return

        yield sse("done", {"stats": stats, "datasets": dataset_summary(), "dataset_detail": dataset_detail(dataset)})

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/api/chat/stream")
def chat_stream(payload: ChatRequest) -> StreamingResponse:
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message is required.")

    mode = payload.mode if payload.mode in {"RAG Only", "RAG + Model", "Model Only", "SQL RDBMS"} else "RAG + Model"
    dataset = clean_dataset_name(payload.dataset) if payload.dataset else active_dataset
    history = [
        {"role": item.get("role", ""), "content": item.get("content", "")}
        for item in payload.history[-8:]
        if item.get("role") in {"user", "assistant"} and item.get("content")
    ]

    def stream():
        use_rag = mode in {"RAG Only", "RAG + Model"}
        if mode == "SQL RDBMS":
            if not dataset:
                yield sse("error", {"message": "Select a dataset before using SQL RDBMS mode."})
                return
            if not rdbms_exists(dataset):
                yield sse("error", {"message": "RDBMS is not generated for this dataset. Click Gen RDBMS first."})
                return
        elif use_rag:
            if not dataset:
                yield sse("error", {"message": "Select a dataset or use Model Only mode."})
                return
            with engine_lock:
                chunks_available = engine.switch_dataset(dataset)
            if chunks_available <= 0:
                yield sse("error", {"message": "Selected dataset is empty. Upload or discover documents first."})
                return

        t_start = time.time()
        stages: list[tuple[str, str]] = [("Mode", mode)]
        yield sse("stats", {"stats": format_stats(stages)})

        try:
            if mode == "SQL RDBMS":
                stages.append(("Schema load", "ready"))
                stages.append(("SQL compile", "running..."))
                yield sse("stats", {"stats": format_stats(stages)})

                t_sql_start = time.time()
                sql, compile_notes = compile_dataset_sql(dataset, message)
                stages[-1] = ("SQL compile", f"{time.time() - t_sql_start:.2f}s")
                stages.append(("SQL execute", "running..."))
                yield sse("token", {"delta": format_generated_sql(sql, compile_notes)})
                yield sse("stats", {"stats": format_stats(stages)})

                t_exec_start = time.time()
                result = execute_readonly_sql(dataset, sql, question=message)
                stages[-1] = ("SQL execute", f"{time.time() - t_exec_start:.2f}s ({result['row_count']} rows)")
                stages.append(("Total", f"{time.time() - t_start:.1f}s"))
                answer = format_sql_answer(result, include_sql=False)
                yield sse("token", {"delta": answer})
                yield sse("done", {"stats": format_stats(stages), "sources": []})
                return

            if mode == "Model Only":
                stages.append(("LLM generation", "waiting for first token..."))
                yield sse("stats", {"stats": format_stats(stages)})
                t_gen_start = time.time()
                first_token_time = None
                llm_usage = None
                token_count = 0

                for delta, usage in engine.generate_stream_direct(message, history=history):
                    if usage:
                        llm_usage = usage
                    if delta:
                        token_count += 1
                        if first_token_time is None:
                            first_token_time = time.time() - t_gen_start
                        elapsed = time.time() - t_gen_start
                        decode_time = elapsed - first_token_time
                        tps = token_count / decode_time if decode_time > 0 else 0
                        stages[-1] = ("LLM generation", f"TTFT {first_token_time:.1f}s | {token_count} chunks @ {tps:.1f}/s")
                        yield sse("token", {"delta": delta})
                        yield sse("stats", {"stats": format_stats(stages)})

                build_final_stats(stages, llm_usage, token_count, first_token_time, t_gen_start, t_start)
                yield sse("done", {"stats": format_stats(stages), "sources": []})
                return

            stages.append(("Embed query", "running..."))
            yield sse("stats", {"stats": format_stats(stages)})
            with engine_lock:
                chunks, timings = engine.retrieve(message)
                total_chunks = engine.collection.count() if engine.collection is not None else 0

            retrieval_k = engine._last_retrieval_k
            stages[-1] = ("Embed query", f"{timings.get('embed', 0):.2f}s")
            stages.append(("Vector search", f"{timings.get('search', 0):.2f}s ({min(retrieval_k, total_chunks)} candidates from {total_chunks})"))
            stages.append(("Rerank", f"{timings.get('rerank', 0):.2f}s (top {len(chunks)})"))
            stages.append(("LLM generation", "waiting for first token..."))
            sources = format_source_refs(dataset, chunks)
            yield sse("sources", {"sources": sources})
            yield sse("stats", {"stats": format_stats(stages)})

            t_gen_start = time.time()
            first_token_time = None
            llm_usage = None
            token_count = 0

            for delta, usage in engine.generate_stream(message, chunks, history=history, mode=mode):
                if usage:
                    llm_usage = usage
                if delta:
                    token_count += 1
                    if first_token_time is None:
                        first_token_time = time.time() - t_gen_start
                    elapsed = time.time() - t_gen_start
                    decode_time = elapsed - first_token_time
                    tps = token_count / decode_time if decode_time > 0 else 0
                    stages[-1] = ("LLM generation", f"TTFT {first_token_time:.1f}s | {token_count} chunks @ {tps:.1f}/s")
                    yield sse("token", {"delta": delta})
                    yield sse("stats", {"stats": format_stats(stages)})

            build_final_stats(stages, llm_usage, token_count, first_token_time, t_gen_start, t_start)
            yield sse("done", {"stats": format_stats(stages), "sources": sources})
        except Exception as exc:
            yield sse("error", {"message": str(exc)})

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    ui_port = int(os.environ.get("MYLABS_UI_PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=ui_port, log_level="info")
