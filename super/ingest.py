"""Document ingestion pipeline with multi-dataset support.

Usage:
    python ingest.py CERN                           # Ingest all files in ./documents/CERN/
    python ingest.py CERN --reset                   # Clear and re-ingest CERN dataset
    python ingest.py CERN path/to/file.pdf          # Add a file to CERN dataset
"""

import os
import sys
import hashlib
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import (
    EMBEDDING_MODEL, VECTORSTORE_DIR, DOCUMENTS_DIR,
    CHUNK_SIZE, CHUNK_OVERLAP,
)


def load_file(path: str) -> str:
    """Extract text from a file based on extension."""
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(path)
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)

    elif ext == ".docx":
        from docx import Document
        doc = Document(path)
        return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())

    elif ext in (".txt", ".md", ".csv", ".json", ".log"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    else:
        print(f"  Skipping unsupported file type: {ext}")
        return ""


def chunk_text(text: str, source: str) -> list[dict]:
    """Split text into overlapping chunks with metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
    )
    chunks = splitter.split_text(text)
    return [
        {
            "id": hashlib.sha256(f"{source}::{i}::{c[:64]}".encode()).hexdigest()[:16],
            "text": c,
            "metadata": {"source": os.path.basename(source), "chunk_index": i},
        }
        for i, c in enumerate(chunks)
        if c.strip()
    ]


def ingest_dataset(dataset_name: str, paths: list[str], reset: bool = False):
    """Ingest documents into a dataset-specific ChromaDB collection."""
    total = 0
    for _ in ingest_dataset_streaming(dataset_name, paths, reset=reset):
        pass
    # Get final count
    collection_name = f"rag_{dataset_name.lower().replace(' ', '_')}"
    client = chromadb.PersistentClient(path=VECTORSTORE_DIR)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    return collection.count()


def ingest_dataset_streaming(dataset_name: str, paths: list[str], reset: bool = False):
    """Ingest documents, yielding progress strings for each step."""
    import time
    collection_name = f"rag_{dataset_name.lower().replace(' ', '_')}"

    yield f"Loading embedding model..."
    embedder = SentenceTransformer(EMBEDDING_MODEL, device="cpu")

    client = chromadb.PersistentClient(path=VECTORSTORE_DIR)

    if reset:
        try:
            client.delete_collection(collection_name)
            yield "Cleared existing collection."
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    total_chunks = 0
    for file_idx, path in enumerate(paths):
        filename = os.path.basename(path)
        yield f"[{file_idx+1}/{len(paths)}] Reading: {filename}"
        text = load_file(path)
        if not text:
            yield f"  ⟶ skipped (no text extracted)"
            continue

        chunks = chunk_text(text, path)
        total_chunks += len(chunks)
        yield f"  ⟶ {len(chunks)} chunks, embedding..."

        batch_size = 128
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c["text"] for c in batch]
            ids = [c["id"] for c in batch]
            metadatas = [c["metadata"] for c in batch]

            t0 = time.time()
            embeddings = embedder.encode(texts).tolist()
            embed_time = time.time() - t0

            collection.upsert(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
                embeddings=embeddings,
            )
            batch_end = min(i + batch_size, len(chunks))
            yield f"  ⟶ embedded batch {i+1}-{batch_end}/{len(chunks)} ({embed_time:.1f}s)"

    total = collection.count()
    yield f"Dataset '{dataset_name}': {total} chunks indexed ({total_chunks} new from {len(paths)} files)"


def main():
    if len(sys.argv) < 2:
        # List available datasets
        print("Available datasets:")
        for name in sorted(os.listdir(DOCUMENTS_DIR)):
            full = os.path.join(DOCUMENTS_DIR, name)
            if os.path.isdir(full) and not name.startswith("."):
                n_files = len([f for f in os.listdir(full) if not f.startswith(".")])
                print(f"  {name}/ ({n_files} files)")
        print(f"\nUsage: python ingest.py <dataset_name> [--reset] [extra_files...]")
        sys.exit(0)

    dataset_name = sys.argv[1]
    reset = "--reset" in sys.argv
    extra_paths = [a for a in sys.argv[2:] if a != "--reset"]

    # Gather files from the dataset folder
    doc_dir = os.path.join(DOCUMENTS_DIR, dataset_name)
    os.makedirs(doc_dir, exist_ok=True)

    paths = [
        os.path.join(doc_dir, f)
        for f in sorted(os.listdir(doc_dir))
        if not f.startswith(".")
    ]
    paths.extend(extra_paths)

    if not paths:
        print(f"No documents found in {doc_dir}/. Place files there or pass paths as arguments.")
        sys.exit(1)

    ingest_dataset(dataset_name, paths, reset=reset)


if __name__ == "__main__":
    main()
