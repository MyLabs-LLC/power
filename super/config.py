"""Shared configuration for RAG pipeline."""

# LLM server (llama.cpp llama-server)
LLM_BASE_URL = "http://localhost:8001/v1"
LLM_MODEL = "nemotron-3-nano"
LLM_API_KEY = "sk-no-key-required"

# Server context window — matches --ctx-size in start_server.sh. Used for the
# context-usage display only; does not cap the actual request.
LLM_CTX_SIZE = 1048576

# Embedding model (runs locally on CPU)
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

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
- Cite the specific document and chunk when referencing a fact (e.g., [source: filename, chunk N]).
- Give precise numbers, dates, and specifics when available in the context.
- Distinguish between stated facts and inferences.
- If multiple documents discuss the same topic, synthesize them and note any differences."""

SYSTEM_PROMPT_RAG_PLUS_MODEL = """You are an expert analyst with access to a curated document collection.
You have access to retrieved excerpts from published research papers and documents.

Instructions:
- Use the retrieved context as your primary source of truth.
- You MAY supplement with your own knowledge to provide deeper analysis, explanations, or background context.
- Clearly distinguish between facts from the documents and your own knowledge (e.g., "According to [source]..." vs "More broadly...").
- Cite the specific document and chunk when referencing a fact from the context (e.g., [source: filename, chunk N]).
- Give precise numbers, dates, and specifics when available in the context.
- If multiple documents discuss the same topic, synthesize them and note any differences."""

# Default (backwards compatibility)
SYSTEM_PROMPT = SYSTEM_PROMPT_RAG_ONLY
