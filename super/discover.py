"""Automated research paper discovery, download, and ingestion.

Given a topic/subject, searches arXiv for relevant papers, downloads PDFs,
extracts text, scores for relevance, and ingests into a dataset.
"""

import os
import re
import time
import hashlib
from pathlib import Path
from typing import Optional

import arxiv
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from config import DOCUMENTS_DIR, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP

PDF_DOWNLOAD_DELAY = 3.0
MAX_PDF_MB = 20
MIN_TEXT_LEN = 500

# arXiv asks for ≥3 s between requests. We space inter-query sleeps at 3 s and
# on a 429 ("Too Many Requests") back off much harder: 30, 60, 120 s.
ARXIV_INTER_QUERY_DELAY = 3.0
ARXIV_429_BACKOFF = (30.0, 60.0, 120.0)
ARXIV_OTHER_BACKOFF = (2.0, 5.0, 10.0)


def title_to_filename(title: str, ext: str = ".pdf", existing: set[str] = None) -> str:
    """Convert a paper title to a short, readable filename.

    Uses the first 3 words, lowercased, joined by hyphens.
    Appends -1, -2, etc. if the name collides with existing filenames.
    """
    if existing is None:
        existing = set()

    # Clean title: strip non-alphanumeric except spaces
    clean = re.sub(r"[^a-zA-Z0-9\s]", "", title)
    words = clean.split()[:3]
    if not words:
        words = ["untitled"]

    base = "-".join(w.lower() for w in words)
    candidate = f"{base}{ext}"

    if candidate not in existing:
        return candidate

    # Append incrementing number
    i = 1
    while True:
        candidate = f"{base}-{i}{ext}"
        if candidate not in existing:
            return candidate
        i += 1


# ── arXiv search ─────────────────────────────────────────────────────────────

def generate_queries(topic: str, num_queries: int = 8) -> list[str]:
    """Generate multiple search queries for a topic to maximize coverage."""
    base = topic.strip()
    queries = [
        base,
        f"{base} review",
        f"{base} recent advances",
        f"{base} measurement results",
        f"{base} experimental",
        f"{base} analysis",
        f"{base} 2024",
        f"{base} 2023",
        f"{base} new results",
        f"{base} overview",
        f"{base} study",
        f"{base} survey",
    ]
    return queries[:num_queries]


def _arxiv_search_raw(client, query, max_results):
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    return list(client.results(search))


def _arxiv_search_with_retry(client, query, max_results, progress_cb=None):
    """Call arXiv with a 429-aware exponential backoff.

    On 429 ("Too Many Requests") waits 30 → 60 → 120 s before retrying.
    On other transient errors waits 2 → 5 → 10 s. Raises after the last attempt.
    """
    last_exc = None
    for attempt in range(len(ARXIV_429_BACKOFF) + 1):
        try:
            return _arxiv_search_raw(client, query, max_results)
        except Exception as e:
            last_exc = e
            if attempt >= len(ARXIV_429_BACKOFF):
                break
            msg = str(e)
            is_429 = "429" in msg or "Too Many Requests" in msg
            wait = ARXIV_429_BACKOFF[attempt] if is_429 else ARXIV_OTHER_BACKOFF[attempt]
            if progress_cb:
                why = "rate-limited (429)" if is_429 else f"error: {type(e).__name__}"
                progress_cb(f"  arXiv {why}; retrying in {wait:.0f}s (attempt {attempt+2}/{len(ARXIV_429_BACKOFF)+1})")
            time.sleep(wait)
    raise last_exc


def search_arxiv_papers(topic: str, max_per_query: int = 25, num_queries: int = 8,
                        progress_cb=None) -> list[dict]:
    """Search arXiv for papers related to the topic."""
    queries = generate_queries(topic, num_queries)
    # delay_seconds: minimum gap the client itself enforces between page fetches.
    # page_size: 100 = arXiv's max per page, so max_results<=100 is a single fetch.
    client = arxiv.Client(delay_seconds=ARXIV_INTER_QUERY_DELAY, num_retries=1)
    papers = []
    seen_ids = set()

    for i, q in enumerate(queries):
        if progress_cb:
            progress_cb(f"Searching arXiv: query {i+1}/{len(queries)} — '{q}'")
        try:
            results = _arxiv_search_with_retry(client, q, max_per_query, progress_cb=progress_cb)
            for r in results:
                aid = r.entry_id.split("/")[-1]
                if aid in seen_ids:
                    continue
                seen_ids.add(aid)

                if not r.summary or len(r.summary) < 100:
                    continue

                papers.append({
                    "arxiv_id": aid,
                    "title": r.title,
                    "abstract": r.summary,
                    "authors": [a.name for a in r.authors[:5]],
                    "published": r.published.strftime("%Y-%m-%d") if r.published else "",
                    "pdf_url": f"https://arxiv.org/pdf/{aid}.pdf",
                })
        except Exception as e:
            if progress_cb:
                progress_cb(f"Query failed after retries: {e}")

        # Polite spacing between queries regardless of success/failure.
        if i < len(queries) - 1:
            time.sleep(ARXIV_INTER_QUERY_DELAY)

    return papers


# ── Relevance scoring ────────────────────────────────────────────────────────

def relevance_score(paper: dict, topic: str) -> float:
    """Score how relevant a paper is to the topic (0-1)."""
    text = (paper.get("title", "") + " " + paper.get("abstract", "")).lower()
    if not text.strip():
        return 0.0

    topic_words = [w.lower() for w in topic.split() if len(w) > 2]
    word_count = max(len(text.split()), 1)

    hits = 0
    for word in topic_words:
        hits += text.count(word) * 2

    # Also check for the full topic phrase
    if topic.lower() in text:
        hits += 10

    return min(hits / (word_count * 0.05), 1.0)


# ── PDF download & text extraction ───────────────────────────────────────────

def download_pdf(pdf_url: str, title: str, dataset_dir: str, existing_files: set[str] = None) -> Optional[str]:
    """Download PDF to dataset directory with a title-based filename. Returns path or None."""
    filename = title_to_filename(title, ext=".pdf", existing=existing_files)
    pdf_path = os.path.join(dataset_dir, filename)

    if os.path.exists(pdf_path):
        return pdf_path

    try:
        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
        def _fetch():
            return requests.get(
                pdf_url, timeout=60,
                headers={"User-Agent": "RAG-Research/1.0 (research paper collection)"},
            )

        r = _fetch()
        if r.status_code != 200:
            return None

        size_mb = len(r.content) / 1024 / 1024
        if size_mb > MAX_PDF_MB:
            return None
        if len(r.content) < 1024:
            return None

        # Verify it's actually a PDF
        if not r.content[:5].startswith(b"%PDF"):
            return None

        with open(pdf_path, "wb") as f:
            f.write(r.content)
        return pdf_path

    except Exception:
        return None


def extract_pdf_text(pdf_path: str) -> Optional[str]:
    """Extract text from PDF using pymupdf."""
    try:
        import fitz
        doc = fitz.open(pdf_path)
        pages = [page.get_text() for page in doc]
        doc.close()
        text = "\n\n".join(p for p in pages if p.strip())
        return text if len(text) >= MIN_TEXT_LEN else None
    except Exception:
        return None


def is_quality_pdf(text: str) -> bool:
    """Check if extracted text is usable (not garbled OCR, etc)."""
    if not text or len(text) < MIN_TEXT_LEN:
        return False
    alpha_ratio = sum(1 for c in text if c.isalpha()) / max(len(text), 1)
    return alpha_ratio >= 0.50


# ── Main discovery pipeline ──────────────────────────────────────────────────

def discover_and_download(
    topic: str,
    dataset_name: str,
    max_papers: int = 30,
    num_queries: int = 8,
    max_per_query: int = 25,
    min_relevance: float = 0.05,
    progress_cb=None,
) -> dict:
    """
    Full pipeline: search → score → download → extract.

    Returns stats dict with counts of papers found, downloaded, etc.
    """
    dataset_dir = os.path.join(DOCUMENTS_DIR, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    # Step 1: Search
    if progress_cb:
        progress_cb(f"Searching arXiv for '{topic}'...")
    papers = search_arxiv_papers(topic, max_per_query=max_per_query,
                                 num_queries=num_queries, progress_cb=progress_cb)
    if progress_cb:
        progress_cb(f"Found {len(papers)} unique papers")

    # Step 2: Score and filter
    for p in papers:
        p["relevance"] = relevance_score(p, topic)

    papers.sort(key=lambda x: x["relevance"], reverse=True)
    papers = [p for p in papers if p["relevance"] >= min_relevance]

    if progress_cb:
        progress_cb(f"After relevance filter: {len(papers)} papers (min score: {min_relevance})")

    # Cap to max_papers
    papers = papers[:max_papers]

    # Step 3: Download PDFs
    downloaded = 0
    failed = 0

    # Track existing filenames in the dataset dir for dedup
    existing_files = set(
        f for f in os.listdir(dataset_dir) if not f.startswith(".")
    ) if os.path.isdir(dataset_dir) else set()

    for i, p in enumerate(papers):
        if progress_cb:
            progress_cb(
                f"Downloading {i+1}/{len(papers)}: {p['title'][:60]}... "
                f"(relevance: {p['relevance']:.2f})"
            )

        pdf_path = download_pdf(p["pdf_url"], p["title"], dataset_dir, existing_files)
        if pdf_path:
            text = extract_pdf_text(pdf_path)
            if text and is_quality_pdf(text):
                p["pdf_path"] = pdf_path
                p["text_len"] = len(text)
                existing_files.add(os.path.basename(pdf_path))
                downloaded += 1
            else:
                try:
                    os.remove(pdf_path)
                except OSError:
                    pass
                failed += 1
        else:
            failed += 1

        time.sleep(PDF_DOWNLOAD_DELAY)

    if progress_cb:
        progress_cb(f"Downloaded {downloaded}/{len(papers)} papers successfully")

    stats = {
        "topic": topic,
        "dataset": dataset_name,
        "searched": len(papers),
        "downloaded": downloaded,
        "failed": failed,
        "papers": [
            {
                "title": p["title"],
                "arxiv_id": p["arxiv_id"],
                "relevance": p["relevance"],
                "has_pdf": "pdf_path" in p,
            }
            for p in papers
        ],
    }

    return stats
