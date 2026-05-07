"""Automated research paper discovery, download, and ingestion.

Given a topic/subject, searches arXiv for relevant papers, downloads PDFs,
extracts text, scores for relevance, and ingests into a dataset.
"""

import os
import random
import re
import time
import hashlib
from pathlib import Path
from typing import Optional

import arxiv
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from config import DOCUMENTS_DIR, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP

PDF_DOWNLOAD_DELAY = float(os.environ.get("RAG_DISCOVER_PDF_DELAY", "4.0"))
MAX_PDF_MB = 50
MIN_TEXT_LEN = 500

# arXiv asks for at least 3 seconds between requests. Keep the app more
# conservative because discovery can issue multiple searches plus PDF downloads.
ARXIV_INTER_QUERY_DELAY = float(os.environ.get("RAG_DISCOVER_ARXIV_DELAY", "7.0"))
ARXIV_429_BACKOFF = (60.0, 120.0, 240.0, 480.0)
ARXIV_OTHER_BACKOFF = (5.0, 15.0, 30.0, 60.0)
DEFAULT_MAX_PER_QUERY = 10
MAX_DISCOVERY_QUERIES = int(os.environ.get("RAG_DISCOVER_MAX_QUERIES", "12"))
DISCOVERY_STOPWORDS = {
    "about", "after", "again", "against", "also", "among", "analysis", "based", "been",
    "before", "between", "clinical", "could", "disease", "effect", "effects", "from",
    "have", "into", "method", "methods", "model", "models", "paper", "patients", "recent",
    "research", "results", "review", "study", "studies", "than", "that", "their", "there",
    "these", "this", "through", "treatment", "using", "were", "where", "which", "while",
    "with", "would", "and", "for", "the", "of", "in", "on", "to", "or", "by", "as",
}


def title_to_filename(title: str, ext: str = ".pdf", existing: set[str] = None) -> str:
    """Convert a paper title to a short, readable filename.

    Uses the first 7 words, lowercased, joined by hyphens.
    Appends -1, -2, etc. if the name collides with existing filenames.
    """
    if existing is None:
        existing = set()

    # Clean title: strip non-alphanumeric except spaces
    clean = re.sub(r"[^a-zA-Z0-9\s]", "", title)
    words = clean.split()[:7]
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

def normalize_query_text(text: str) -> str:
    """Normalize pasted topics for arXiv search while preserving meaning."""
    text = text.replace("\u2011", "-").replace("\u2010", "-").replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def add_unique(items: list[str], value: str) -> None:
    value = normalize_query_text(value)
    if value and value.lower() not in {item.lower() for item in items}:
        items.append(value)


def normalized_query_plan(topic: str, queries: list[str] | None, num_queries: int) -> list[str]:
    """Clean and cap an externally supplied query plan."""
    planned: list[str] = []
    add_unique(planned, topic)
    for query in queries or []:
        query = re.sub(r"^(?:[-*]|\d+[.)])\s*", "", str(query)).strip().strip("\"'")
        if 3 <= len(query) <= 140:
            add_unique(planned, query)
        if len(planned) >= num_queries:
            break
    return planned[:num_queries]


def extract_adaptive_queries(topic: str, papers: list[dict], limit: int = 4) -> list[str]:
    """Create follow-up queries from terms appearing in early arXiv results."""
    text = " ".join(
        f"{paper.get('title', '')} {paper.get('abstract', '')}"
        for paper in papers[:5]
    )
    words = [
        word.lower()
        for word in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", text)
        if word.lower() not in DISCOVERY_STOPWORDS and not word.isdigit()
    ]
    if not words:
        return []

    candidates: dict[str, int] = {}
    for size in (3, 2):
        for i in range(0, max(0, len(words) - size + 1)):
            phrase_words = words[i : i + size]
            if len(set(phrase_words)) < size:
                continue
            phrase = " ".join(phrase_words)
            candidates[phrase] = candidates.get(phrase, 0) + size

    topic_terms = [
        word.lower()
        for word in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", topic)
        if word.lower() not in DISCOVERY_STOPWORDS
    ]
    followups: list[str] = []
    for phrase, _score in sorted(candidates.items(), key=lambda item: (-item[1], item[0])):
        if any(term in phrase for term in topic_terms):
            add_unique(followups, phrase)
        elif topic_terms:
            add_unique(followups, f"{topic_terms[0]} {phrase}")
        else:
            add_unique(followups, phrase)
        if len(followups) >= limit:
            break
    return followups


def topic_signals(topic: str) -> dict[str, bool]:
    text = normalize_query_text(topic).lower()
    return {
        "car": bool(re.search(r"\bcar\b|chimeric antigen receptor", text)),
        "t_cell": bool(re.search(r"\bt[- ]?cells?\b|\btreg\b|regulatory t", text)),
        "ra": bool(re.search(r"\bra\b|rheumatoid arthritis|autoimmune arthritis", text)),
        "autoimmune": bool(re.search(r"autoimmune|autoimmunity|rheumatoid|arthritis|\bra\b", text)),
        "cell_therapy": bool(re.search(r"cell|engineered|therapy|treatment|approach", text)),
    }


def expanded_topic_phrases(topic: str) -> list[str]:
    """Return semantically related phrases for biomedical paper discovery.

    The first item is always the user's exact topic. Later items broaden common
    acronyms and shift wording toward terminology found in titles/abstracts.
    """
    base = normalize_query_text(topic)
    signals = topic_signals(base)
    phrases: list[str] = []
    add_unique(phrases, base)

    if signals["car"]:
        car_terms = ["CAR T-cell", "CAR T cells", "chimeric antigen receptor T cells", "CAR-engineered cells"]
        if signals["t_cell"]:
            car_terms.extend(["CAR T cell therapy", "CAR Treg", "CAR regulatory T cells"])
        else:
            car_terms.extend(["chimeric antigen receptor therapy", "CAR immune cells", "CAR NK cells"])
    else:
        car_terms = []

    if signals["ra"]:
        disease_terms = ["rheumatoid arthritis", "RA", "autoimmune arthritis"]
    elif signals["autoimmune"]:
        disease_terms = ["autoimmune disease", "autoimmunity", "inflammatory arthritis"]
    else:
        disease_terms = []

    if car_terms and disease_terms:
        add_unique(phrases, "CAR T-cell research")
        add_unique(phrases, "CAR T-cell research autoimmune disease")
        add_unique(phrases, "CAR RA treatments")
        add_unique(phrases, "CAR and T-cells in autoimmune disease")
        add_unique(phrases, "CAR T-cell rheumatoid arthritis")
        add_unique(phrases, "chimeric antigen receptor T cells rheumatoid arthritis")
        add_unique(phrases, "CAR-engineered cells autoimmune disease")
        add_unique(phrases, "CAR Treg autoimmune arthritis")
        add_unique(phrases, "CAR regulatory T cells rheumatoid arthritis")
        add_unique(phrases, "CAR NK cells rheumatoid arthritis")
        add_unique(phrases, "engineered T cells rheumatoid arthritis")
        add_unique(phrases, "adoptive cell therapy rheumatoid arthritis")
        for car_term in car_terms:
            for disease_term in disease_terms:
                add_unique(phrases, f"{car_term} {disease_term}")
    elif car_terms:
        for car_term in car_terms:
            add_unique(phrases, car_term)
        add_unique(phrases, "CAR T-cell research")
        add_unique(phrases, "chimeric antigen receptor cell therapy")
    elif disease_terms:
        for disease_term in disease_terms:
            add_unique(phrases, disease_term)
        add_unique(phrases, f"{base} therapy")
        add_unique(phrases, f"{base} treatment")

    add_unique(phrases, base.replace("approaches", "therapy"))
    add_unique(phrases, base.replace("approaches", "treatment"))
    return phrases


def generate_queries(topic: str, num_queries: int = 8) -> list[str]:
    """Generate exact and related arXiv search queries for a topic."""
    base = normalize_query_text(topic)
    phrases = expanded_topic_phrases(base)
    queries: list[str] = []

    for phrase in phrases:
        add_unique(queries, phrase)

    # Add a few literature-intent variants after the semantic expansions. Keep
    # these later so exact and synonym searches happen first.
    for suffix in ("review", "clinical trial", "preclinical", "recent advances", "study", "survey"):
        add_unique(queries, f"{base} {suffix}")

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
    max_attempts = len(ARXIV_429_BACKOFF) + 1
    for attempt in range(max_attempts):
        try:
            return _arxiv_search_raw(client, query, max_results)
        except Exception as e:
            last_exc = e
            if attempt >= len(ARXIV_429_BACKOFF):
                break
            msg = str(e)
            is_429 = "429" in msg or "Too Many Requests" in msg
            base_wait = ARXIV_429_BACKOFF[attempt] if is_429 else ARXIV_OTHER_BACKOFF[attempt]
            wait = base_wait * random.uniform(0.9, 1.2)
            if progress_cb:
                why = "rate-limited (429)" if is_429 else f"error: {type(e).__name__}"
                progress_cb(f"  arXiv {why}; cooling down for {wait:.0f}s before retry {attempt+2}/{max_attempts}.")
            sleep_with_progress(wait, progress_cb, prefix="  arXiv cooldown")
    raise last_exc


def sleep_with_progress(seconds: float, progress_cb=None, prefix: str = "Waiting") -> None:
    """Sleep in chunks so long arXiv backoffs show visible progress."""
    remaining = max(0.0, seconds)
    while remaining > 0:
        step = min(15.0, remaining)
        time.sleep(step)
        remaining -= step
        if progress_cb and remaining > 0:
            progress_cb(f"{prefix}: {remaining:.0f}s remaining")


def search_arxiv_papers(topic: str, max_per_query: int = DEFAULT_MAX_PER_QUERY, num_queries: int = 8,
                        progress_cb=None, query_plan: list[str] | None = None,
                        adaptive: bool = True) -> list[dict]:
    """Search arXiv for papers related to the topic."""
    num_queries = max(1, min(int(num_queries), MAX_DISCOVERY_QUERIES))
    max_per_query = max(1, min(int(max_per_query), DEFAULT_MAX_PER_QUERY))
    queries = normalized_query_plan(topic, query_plan, num_queries) if query_plan else generate_queries(topic, num_queries)
    # delay_seconds: minimum gap the client itself enforces between page fetches.
    # page_size: 100 = arXiv's max per page, so max_results<=100 is a single fetch.
    client = arxiv.Client(delay_seconds=ARXIV_INTER_QUERY_DELAY, num_retries=1)
    papers = []
    seen_ids = set()

    if progress_cb:
        progress_cb(
            f"arXiv pacing: {len(queries)} queries, up to {max_per_query} results/query, "
            f"{ARXIV_INTER_QUERY_DELAY:.0f}s between queries."
        )
        progress_cb("Expanded query plan: " + " | ".join(queries))

    i = 0
    while i < len(queries):
        q = queries[i]
        if progress_cb:
            progress_cb(f"Searching arXiv: query {i+1}/{len(queries)} — '{q}'")
        try:
            results = _arxiv_search_with_retry(client, q, max_per_query, progress_cb=progress_cb)
            new_papers: list[dict] = []
            for r in results:
                aid = r.entry_id.split("/")[-1]
                if aid in seen_ids:
                    continue
                seen_ids.add(aid)

                if not r.summary or len(r.summary) < 100:
                    continue

                paper = {
                    "arxiv_id": aid,
                    "title": r.title,
                    "abstract": r.summary,
                    "authors": [a.name for a in r.authors[:5]],
                    "published": r.published.strftime("%Y-%m-%d") if r.published else "",
                    "pdf_url": f"https://arxiv.org/pdf/{aid}.pdf",
                }
                papers.append(paper)
                new_papers.append(paper)
            if adaptive and i == 0 and len(queries) < num_queries and new_papers:
                adaptive_queries = extract_adaptive_queries(topic, new_papers, limit=num_queries - len(queries))
                for query in adaptive_queries:
                    add_unique(queries, query)
                if progress_cb and adaptive_queries:
                    progress_cb("Adaptive follow-up queries: " + " | ".join(adaptive_queries))
        except Exception as e:
            if progress_cb:
                progress_cb(f"Query failed after retries: {e}")

        # Polite spacing between queries regardless of success/failure.
        if i < len(queries) - 1:
            sleep_with_progress(ARXIV_INTER_QUERY_DELAY, progress_cb=None)
        i += 1

    return papers


# ── Relevance scoring ────────────────────────────────────────────────────────

def relevance_score(paper: dict, topic: str, query_plan: list[str] | None = None) -> float:
    """Score how relevant a paper is to the topic (0-1)."""
    text = (paper.get("title", "") + " " + paper.get("abstract", "")).lower()
    if not text.strip():
        return 0.0

    phrases = normalized_query_plan(topic, query_plan, MAX_DISCOVERY_QUERIES) if query_plan else expanded_topic_phrases(topic)
    query_text = " ".join(phrases).lower()
    topic_words = [w.lower() for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9-]{1,}", query_text) if len(w) > 2]
    word_count = max(len(text.split()), 1)

    hits = 0
    for word in topic_words:
        hits += text.count(word) * 2

    for phrase in phrases[:8]:
        if phrase.lower() in text:
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
    max_per_query: int = DEFAULT_MAX_PER_QUERY,
    min_relevance: float = 0.05,
    query_plan: list[str] | None = None,
    progress_cb=None,
) -> dict:
    """
    Full pipeline: search → score → download → extract.

    Returns stats dict with counts of papers found, downloaded, etc.
    """
    dataset_dir = os.path.join(DOCUMENTS_DIR, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    max_papers = max(1, min(int(max_papers), 50))
    num_queries = max(1, min(int(num_queries), MAX_DISCOVERY_QUERIES))
    max_per_query = max(1, min(int(max_per_query), DEFAULT_MAX_PER_QUERY))

    # Step 1: Search
    if progress_cb:
        progress_cb(
            f"Searching arXiv for '{topic}' with conservative pacing "
            f"({num_queries} queries, max {max_papers} downloads)..."
        )
    papers = search_arxiv_papers(
        topic,
        max_per_query=max_per_query,
        num_queries=num_queries,
        progress_cb=progress_cb,
        query_plan=query_plan,
        adaptive=True,
    )
    if progress_cb:
        progress_cb(f"Found {len(papers)} unique papers")

    # Step 2: Score and filter
    for p in papers:
        p["relevance"] = relevance_score(p, topic, query_plan=query_plan)

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
        "queries": normalized_query_plan(topic, query_plan, num_queries) if query_plan else generate_queries(topic, num_queries),
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
