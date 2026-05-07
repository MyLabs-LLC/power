"""Shared configuration for RAG pipeline."""

import os
from pathlib import Path


def load_env_file(path: str | os.PathLike | None = None) -> None:
    """Load repo-local .env values without overriding existing environment."""
    env_path = Path(path) if path else Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        os.environ.setdefault(key, value)

    # Support both common Hugging Face env var spellings.
    if "HUGGINGFACE_HUB_TOKEN" in os.environ:
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", os.environ["HUGGINGFACE_HUB_TOKEN"])


load_env_file()

# LLM server (llama.cpp llama-server)
LLM_BASE_URL = os.getenv("OPENAI_BASE_URL", os.getenv("LLM_BASE_URL", "http://localhost:8001/v1"))
LLM_MODEL = os.getenv("OPENAI_MODEL", os.getenv("LLM_MODEL", "nemotron-3-nano"))
LLM_API_KEY = os.getenv("OPENAI_API_KEY", os.getenv("LLM_API_KEY", "sk-no-key-required"))

# Server context window — matches --ctx-size in start_server.sh. Used for the
# context-usage display only; does not cap the actual request.
LLM_CTX_SIZE = 1048576

# Embedding model
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
EMBEDDING_QUERY_PROMPT_NAME = os.getenv("EMBEDDING_QUERY_PROMPT_NAME", "query")

# Reranker (cross-encoder, runs on CPU)
RERANKER_MODEL = "BAAI/bge-reranker-base"

# ChromaDB
VECTORSTORE_DIR = "./vectorstore"

# Documents root — each dataset is a subfolder
DOCUMENTS_DIR = "./documents"

# Chunking
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 128

# Retrieval
RETRIEVAL_K = 20   # initial retrieval (overfetch)
TOP_K = 8          # after reranking

# Generation
MAX_TOKENS = 4096
TEMPERATURE = 0.6
TOP_P = 0.95

SYSTEM_PROMPT_RAG_ONLY = """You are an expert analyst with access to a curated document collection.
You have access to retrieved excerpts from published research papers and documents.

Instructions:
- Answer ONLY based on the retrieved context. Do not use prior knowledge to fill gaps.
- If the context is insufficient, say so explicitly.
- Cite facts inline using the exact citation labels from the retrieved context, for example [source: filename.pdf, page N, chunk N]. Do not put citations only in a separate sources section.
- Give precise numbers, dates, and specifics when available in the context.
- Distinguish between stated facts and inferences.
- If multiple documents discuss the same topic, synthesize them and note any differences."""

SYSTEM_PROMPT_RAG_PLUS_MODEL = """You are an expert analyst with access to a curated document collection.
You have access to retrieved excerpts from published research papers and documents.

Instructions:
- Use the retrieved context as your primary source of truth.
- You MAY supplement with your own knowledge to provide deeper analysis, explanations, or background context.
- Clearly distinguish between facts from the documents and your own knowledge (e.g., "According to [source]..." vs "More broadly...").
- Cite facts from the documents inline using the exact citation labels from the retrieved context, for example [source: filename.pdf, page N, chunk N]. Do not put citations only in a separate sources section.
- Give precise numbers, dates, and specifics when available in the context.
- If multiple documents discuss the same topic, synthesize them and note any differences."""

# Default (backwards compatibility)
SYSTEM_PROMPT = SYSTEM_PROMPT_RAG_ONLY
