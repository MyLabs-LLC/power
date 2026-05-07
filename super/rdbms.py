"""Dataset-to-SQL layer for exact, source-grounded RDBMS querying."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import quote

from config import DOCUMENTS_DIR
from ingest import iter_pdf_page_texts, load_file

RDBMS_DIRNAME = ".rdbms"
DB_FILENAME = "dataset.sqlite"
PROFILE_FILENAME = "profile.json"
SCHEMA_VERSION = 1
MAX_SQL_ROWS = 80

STOPWORDS = {
    "about", "after", "again", "against", "also", "because", "been", "before", "being",
    "between", "both", "could", "does", "each", "from", "have", "into", "more", "most",
    "other", "over", "same", "should", "such", "than", "that", "their", "there", "these",
    "this", "through", "under", "using", "were", "where", "which", "while", "with", "would",
}

MEASUREMENT_RE = re.compile(
    r"(?P<value>[+-]?\d+(?:,\d{3})*(?:\.\d+)?(?:\s*[x\u00d7]\s*10\^?[+-]?\d+)?)\s*"
    r"(?P<unit>%|percent|s|ms|kg|g|m|cm|mm|km|Hz|kHz|MHz|GHz|K|eV|keV|MeV|GeV|TeV|"
    r"m/s|km/s|cm\^-?2|cm-2|cm\u00b2|yr|years?|days?|hours?)",
    re.IGNORECASE,
)
CITATION_RE = re.compile(r"\[(?:\d+(?:,\s*\d+)*|[A-Z][A-Za-z]+(?:\s+et\s+al\.)?,?\s+\d{4})\]")
TERM_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+-]{2,}")


def dataset_rdbms_dir(dataset: str) -> Path:
    return Path(DOCUMENTS_DIR) / dataset / RDBMS_DIRNAME


def dataset_db_path(dataset: str) -> Path:
    return dataset_rdbms_dir(dataset) / DB_FILENAME


def dataset_profile_path(dataset: str) -> Path:
    return dataset_rdbms_dir(dataset) / PROFILE_FILENAME


def rdbms_exists(dataset: str) -> bool:
    return dataset_db_path(dataset).is_file()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def word_count(text: str) -> int:
    return len(re.findall(r"\w+", text or ""))


def estimate_tokens(text: str) -> int:
    return max(1, len(text or "") // 4)


def clean_title(filename: str) -> str:
    return Path(filename).stem.replace("-", " ").replace("_", " ").strip().title()


def source_url(source: str, page: int | None = None, chunk_index: int | None = None) -> str:
    source_part = quote(Path(source).name, safe="")
    chunk_query = f"?chunk={int(chunk_index)}" if chunk_index is not None else ""
    if page:
        return f"/api/documents/{source_part}/pages/{int(page)}{chunk_query}"
    return f"/api/documents/{source_part}/text{chunk_query}"


def connect_rw(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def connect_ro(path: Path) -> sqlite3.Connection:
    uri = f"file:{path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA foreign_keys = ON;

        CREATE TABLE meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE documents (
            document_id INTEGER PRIMARY KEY,
            dataset_name TEXT NOT NULL,
            source_name TEXT NOT NULL UNIQUE,
            title TEXT NOT NULL,
            file_ext TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            byte_size INTEGER NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE pages (
            page_id INTEGER PRIMARY KEY,
            document_id INTEGER NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
            page_number INTEGER NOT NULL,
            text TEXT NOT NULL,
            char_count INTEGER NOT NULL,
            word_count INTEGER NOT NULL,
            UNIQUE(document_id, page_number)
        );

        CREATE TABLE chunks (
            chunk_id INTEGER PRIMARY KEY,
            document_id INTEGER NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
            page_id INTEGER REFERENCES pages(page_id) ON DELETE SET NULL,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            char_count INTEGER NOT NULL,
            token_estimate INTEGER NOT NULL,
            source_url TEXT NOT NULL,
            UNIQUE(document_id, chunk_index)
        );

        CREATE TABLE terms (
            term_id INTEGER PRIMARY KEY,
            term TEXT NOT NULL UNIQUE,
            term_kind TEXT NOT NULL DEFAULT 'keyword'
        );

        CREATE TABLE chunk_terms (
            chunk_id INTEGER NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
            term_id INTEGER NOT NULL REFERENCES terms(term_id) ON DELETE CASCADE,
            frequency INTEGER NOT NULL,
            PRIMARY KEY (chunk_id, term_id)
        );

        CREATE TABLE measurements (
            measurement_id INTEGER PRIMARY KEY,
            chunk_id INTEGER NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
            raw_value TEXT NOT NULL,
            numeric_value REAL,
            unit TEXT NOT NULL,
            context TEXT NOT NULL
        );

        CREATE TABLE citations (
            citation_id INTEGER PRIMARY KEY,
            chunk_id INTEGER NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
            citation_text TEXT NOT NULL
        );

        CREATE TABLE sql_audit (
            audit_id INTEGER PRIMARY KEY,
            question TEXT NOT NULL,
            generated_sql TEXT NOT NULL,
            row_count INTEGER NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE INDEX idx_pages_document ON pages(document_id, page_number);
        CREATE INDEX idx_chunks_document ON chunks(document_id, chunk_index);
        CREATE INDEX idx_chunks_page ON chunks(page_id);
        CREATE INDEX idx_terms_term ON terms(term);
        CREATE INDEX idx_measurements_unit ON measurements(unit);
        """
    )


def reset_database(path: Path) -> sqlite3.Connection:
    if path.exists():
        path.unlink()
    conn = connect_rw(path)
    init_schema(conn)
    conn.execute("INSERT INTO meta(key, value) VALUES (?, ?)", ("schema_version", str(SCHEMA_VERSION)))
    return conn


def extract_terms(text: str, max_terms: int = 18) -> list[tuple[str, int]]:
    counts: Counter[str] = Counter()
    for raw in TERM_RE.findall(text or ""):
        term = raw.strip("_+-").lower()
        if len(term) < 3 or term in STOPWORDS or term.isdigit():
            continue
        counts[term] += 1
    return counts.most_common(max_terms)


def measurement_rows(text: str) -> list[tuple[str, float | None, str, str]]:
    rows = []
    for match in MEASUREMENT_RE.finditer(text or ""):
        raw_value = match.group("value")
        unit = match.group("unit")
        try:
            numeric = float(raw_value.replace(",", "").split()[0])
        except ValueError:
            numeric = None
        start = max(0, match.start() - 90)
        end = min(len(text), match.end() + 90)
        context = re.sub(r"\s+", " ", text[start:end]).strip()
        rows.append((raw_value, numeric, unit, context))
    return rows[:20]


def citation_rows(text: str) -> list[str]:
    return list(dict.fromkeys(CITATION_RE.findall(text or "")))[:20]


def insert_pages(conn: sqlite3.Connection, document_id: int, path: Path) -> dict[int, int]:
    page_ids: dict[int, int] = {}
    if path.suffix.lower() == ".pdf":
        for _extractor, _total_pages, page_num, text in iter_pdf_page_texts(str(path)):
            text = text or ""
            cur = conn.execute(
                """
                INSERT INTO pages(document_id, page_number, text, char_count, word_count)
                VALUES (?, ?, ?, ?, ?)
                """,
                (document_id, page_num, text, len(text), word_count(text)),
            )
            page_ids[page_num] = int(cur.lastrowid)
        return page_ids

    text = load_file(str(path)) or ""
    cur = conn.execute(
        """
        INSERT INTO pages(document_id, page_number, text, char_count, word_count)
        VALUES (?, 1, ?, ?, ?)
        """,
        (document_id, text, len(text), word_count(text)),
    )
    page_ids[1] = int(cur.lastrowid)
    return page_ids


def build_domain_profile(dataset: str, conn: sqlite3.Connection, counts: dict[str, int]) -> dict[str, Any]:
    top_terms = [
        row["term"]
        for row in conn.execute(
            """
            SELECT t.term, SUM(ct.frequency) AS freq
            FROM terms t
            JOIN chunk_terms ct ON ct.term_id = t.term_id
            GROUP BY t.term_id
            ORDER BY freq DESC, t.term
            LIMIT 40
            """
        )
    ]
    titles = [row["title"] for row in conn.execute("SELECT title FROM documents ORDER BY title LIMIT 30")]
    signal = " ".join(top_terms + titles).lower()
    if any(word in signal for word in ("statute", "court", "legal", "contract", "regulation", "liability", "rights")):
        domain = "legal"
    elif any(word in signal for word in ("trial", "patient", "disease", "clinical", "treatment", "medical")):
        domain = "medical_science"
    elif any(word in signal for word in ("gravity", "quantum", "measurement", "experiment", "model", "wave")):
        domain = "scientific_literature"
    elif any(word in signal for word in ("api", "server", "platform", "configuration", "user", "guide")):
        domain = "technical_documentation"
    else:
        domain = "general_documents"

    prompt = (
        "Generate SQLite SELECT queries only. Prefer exact evidence from chunks/pages over inference. "
        "For legal, science, and medical questions, return source_name, page_number, chunk_index, and source_url "
        "whenever a claim depends on text. Use measurements for numeric/unit questions, terms/chunk_terms for keyword "
        "coverage, and chunks joined to documents/pages for textual evidence. Never write INSERT/UPDATE/DELETE/DDL. "
        "If the schema cannot answer exactly, query the closest evidence and state the limitation."
    )
    return {
        "dataset": dataset,
        "domain": domain,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "counts": counts,
        "top_terms": top_terms,
        "prompt_engineering": prompt,
    }


def build_rdbms(dataset: str, grouped_chunks: dict[str, list[dict]], emit=None) -> dict[str, Any]:
    if not grouped_chunks:
        raise ValueError("No indexed chunks are available for this dataset.")

    db_path = dataset_db_path(dataset)
    conn = reset_database(db_path)
    counts = {"documents": 0, "pages": 0, "chunks": 0, "terms": 0, "measurements": 0, "citations": 0}
    doc_root = Path(DOCUMENTS_DIR) / dataset

    try:
        with conn:
            for idx, (source, chunks) in enumerate(grouped_chunks.items(), start=1):
                path = doc_root / Path(source).name
                if not path.is_file():
                    continue
                if emit:
                    emit("log", {"message": f"[{idx}/{len(grouped_chunks)}] Normalizing {source} into 3NF tables"})

                cur = conn.execute(
                    """
                    INSERT INTO documents(dataset_name, source_name, title, file_ext, sha256, byte_size, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        dataset,
                        Path(source).name,
                        clean_title(source),
                        path.suffix.lower() or "unknown",
                        file_sha256(path),
                        path.stat().st_size,
                        time.strftime("%Y-%m-%d %H:%M:%S"),
                    ),
                )
                document_id = int(cur.lastrowid)
                counts["documents"] += 1
                page_ids = insert_pages(conn, document_id, path)
                counts["pages"] += len(page_ids)

                for chunk in chunks:
                    page = chunk.get("page")
                    text = chunk.get("text") or ""
                    chunk_index = int(chunk.get("chunk_index", -1))
                    cur = conn.execute(
                        """
                        INSERT INTO chunks(document_id, page_id, chunk_index, text, char_count, token_estimate, source_url)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            document_id,
                            page_ids.get(int(page)) if page else None,
                            chunk_index,
                            text,
                            len(text),
                            estimate_tokens(text),
                            source_url(Path(source).name, page, chunk_index),
                        ),
                    )
                    chunk_id = int(cur.lastrowid)
                    counts["chunks"] += 1

                    for term, frequency in extract_terms(text):
                        term_cur = conn.execute(
                            "INSERT OR IGNORE INTO terms(term, term_kind) VALUES (?, 'keyword')",
                            (term,),
                        )
                        if term_cur.rowcount:
                            counts["terms"] += 1
                        term_id = conn.execute("SELECT term_id FROM terms WHERE term = ?", (term,)).fetchone()["term_id"]
                        conn.execute(
                            "INSERT OR REPLACE INTO chunk_terms(chunk_id, term_id, frequency) VALUES (?, ?, ?)",
                            (chunk_id, term_id, frequency),
                        )

                    for raw_value, numeric, unit, context in measurement_rows(text):
                        conn.execute(
                            """
                            INSERT INTO measurements(chunk_id, raw_value, numeric_value, unit, context)
                            VALUES (?, ?, ?, ?, ?)
                            """,
                            (chunk_id, raw_value, numeric, unit, context),
                        )
                        counts["measurements"] += 1

                    for citation in citation_rows(text):
                        conn.execute(
                            "INSERT INTO citations(chunk_id, citation_text) VALUES (?, ?)",
                            (chunk_id, citation),
                        )
                        counts["citations"] += 1

        profile = build_domain_profile(dataset, conn, counts)
        dataset_profile_path(dataset).write_text(json.dumps(profile, indent=2), encoding="utf-8")
        return {"path": str(db_path), "profile": profile, **counts}
    finally:
        conn.close()


def rdbms_info(dataset: str) -> dict[str, Any]:
    path = dataset_db_path(dataset)
    profile_path = dataset_profile_path(dataset)
    if not path.is_file():
        return {"available": False}
    profile = {}
    if profile_path.is_file():
        profile = json.loads(profile_path.read_text(encoding="utf-8", errors="ignore"))
    return {
        "available": True,
        "path": str(path),
        "updated": path.stat().st_mtime,
        "profile": profile,
    }


def schema_summary(dataset: str) -> str:
    path = dataset_db_path(dataset)
    if not path.is_file():
        raise FileNotFoundError("RDBMS has not been generated for this dataset.")
    conn = connect_ro(path)
    try:
        lines = []
        for table in ("documents", "pages", "chunks", "terms", "chunk_terms", "measurements", "citations"):
            cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
            col_text = ", ".join(f"{row['name']} {row['type']}" for row in cols)
            lines.append(f"{table}({col_text})")
        return "\n".join(lines)
    finally:
        conn.close()


def read_profile(dataset: str) -> dict[str, Any]:
    info = rdbms_info(dataset)
    return info.get("profile", {}) if info.get("available") else {}


def extract_sql(text: str) -> str:
    text = re.sub(r"</?think>", " ", text or "", flags=re.IGNORECASE).strip()
    match = re.search(r"```sql\s*(.*?)```", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"\b(select|with)\b[\s\S]*", text, re.IGNORECASE)
    if not match:
        return ""
    sql = match.group(0).strip()
    sql = re.sub(r"^(sql\s*:|query\s*:)\s*", "", sql, flags=re.IGNORECASE).strip()
    semicolon = sql.find(";")
    if semicolon >= 0:
        sql = sql[:semicolon]
    return sql.strip()


def validate_readonly_sql(sql: str) -> str:
    sql = sql.strip().rstrip(";")
    if not sql:
        raise ValueError("No SQL was generated.")
    lowered = re.sub(r"\s+", " ", sql.lower())
    if not (lowered.startswith("select ") or lowered.startswith("with ")):
        raise ValueError("Only read-only SELECT/WITH queries are allowed.")
    forbidden = re.search(
        r"\b(insert|update|delete|drop|alter|create|replace|truncate|attach|detach|vacuum|pragma|reindex)\b",
        lowered,
    )
    if forbidden:
        raise ValueError(f"Unsafe SQL keyword is not allowed: {forbidden.group(1)}")
    if ";" in sql:
        raise ValueError("Only one SQL statement is allowed.")
    return sql


def execute_readonly_sql(dataset: str, sql: str, question: str = "") -> dict[str, Any]:
    sql = validate_readonly_sql(sql)
    path = dataset_db_path(dataset)
    if not path.is_file():
        raise FileNotFoundError("RDBMS has not been generated for this dataset.")

    conn = connect_ro(path)
    try:
        cur = conn.execute(sql)
        rows = cur.fetchmany(MAX_SQL_ROWS + 1)
        columns = [desc[0] for desc in cur.description or []]
        limited = len(rows) > MAX_SQL_ROWS
        rows = rows[:MAX_SQL_ROWS]
        data = [dict(row) for row in rows]
    finally:
        conn.close()

    audit = connect_rw(path)
    try:
        with audit:
            audit.execute(
                "INSERT INTO sql_audit(question, generated_sql, row_count, created_at) VALUES (?, ?, ?, ?)",
                (question, sql, len(data), time.strftime("%Y-%m-%d %H:%M:%S")),
            )
    finally:
        audit.close()

    return {"sql": sql, "columns": columns, "rows": data, "row_count": len(data), "limited": limited}


def check_readonly_sql(dataset: str, sql: str) -> str:
    sql = validate_readonly_sql(sql)
    path = dataset_db_path(dataset)
    if not path.is_file():
        raise FileNotFoundError("RDBMS has not been generated for this dataset.")

    conn = connect_ro(path)
    try:
        conn.execute(f"EXPLAIN QUERY PLAN {sql}").fetchall()
    finally:
        conn.close()
    return sql


def rows_markdown(rows: list[dict[str, Any]], columns: list[str], limit: int = 12) -> str:
    if not rows or not columns:
        return "_No rows returned._"
    shown = rows[:limit]
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in shown:
        values = []
        for col in columns:
            value = row.get(col)
            text = "" if value is None else str(value)
            text = text.replace("\n", " ")[:220]
            values.append(text)
        body.append("| " + " | ".join(values) + " |")
    extra = f"\n\n_Showing {len(shown)} of {len(rows)} rows._" if len(rows) > len(shown) else ""
    return "\n".join([header, sep, *body]) + extra
