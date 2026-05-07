"""RAG retrieval and generation logic with cross-encoder reranking and multi-dataset support."""

import time
from openai import OpenAI
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from config import (
    LLM_BASE_URL, LLM_MODEL, LLM_API_KEY,
    EMBEDDING_MODEL, EMBEDDING_QUERY_PROMPT_NAME, RERANKER_MODEL,
    VECTORSTORE_DIR,
    RETRIEVAL_K, TOP_K, MAX_TOKENS, TEMPERATURE, TOP_P,
    SYSTEM_PROMPT_RAG_ONLY, SYSTEM_PROMPT_RAG_PLUS_MODEL,
)


class RAGEngine:
    def __init__(self):
        self.llm = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
        print("Loading embedding model...")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
        print("Loading reranker model...")
        self.reranker = CrossEncoder(RERANKER_MODEL, device="cpu")
        self.db = chromadb.PersistentClient(path=VECTORSTORE_DIR)
        self._last_retrieval_k = RETRIEVAL_K

        # Current active collection
        self.collection = None
        self._dataset_name = None

    def encode_query(self, query: str):
        """Encode retrieval queries with a model-specific prompt when available."""
        if EMBEDDING_QUERY_PROMPT_NAME and EMBEDDING_QUERY_PROMPT_NAME in getattr(self.embedder, "prompts", {}):
            return self.embedder.encode(query, prompt_name=EMBEDDING_QUERY_PROMPT_NAME)
        return self.embedder.encode(query)

    def switch_dataset(self, dataset_name: str):
        """Switch to a different dataset's ChromaDB collection."""
        collection_name = f"rag_{dataset_name.lower().replace(' ', '_')}"
        self.collection = self.db.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._dataset_name = dataset_name
        return self.collection.count()

    def list_collections(self) -> list[dict]:
        """List all dataset collections with their chunk counts."""
        collections = self.db.list_collections()
        result = []
        for col in collections:
            name = col.name
            display_name = name[4:] if name.startswith("rag_") else name
            result.append({"name": display_name, "collection": name, "chunks": col.count()})
        return result

    def delete_dataset(self, dataset_name: str):
        """Delete a dataset's collection from ChromaDB."""
        collection_name = f"rag_{dataset_name.lower().replace(' ', '_')}"
        try:
            self.db.delete_collection(collection_name)
        except Exception:
            pass

    def all_chunks_by_source(self, dataset_name: str, page_size: int = 1000) -> dict[str, list[dict]]:
        """Return all indexed chunks for a dataset grouped by source document."""
        self.switch_dataset(dataset_name)
        if self.collection is None or self.collection.count() == 0:
            return {}

        grouped: dict[str, list[dict]] = {}
        total = self.collection.count()
        for offset in range(0, total, page_size):
            result = self.collection.get(
                limit=page_size,
                offset=offset,
                include=["documents", "metadatas"],
            )
            documents = result.get("documents") or []
            metadatas = result.get("metadatas") or []
            for text, meta in zip(documents, metadatas):
                source = meta.get("source", "unknown")
                grouped.setdefault(source, []).append(
                    {
                        "text": text,
                        "source": source,
                        "chunk_index": meta.get("chunk_index", -1),
                        "page": meta.get("page"),
                    }
                )

        for chunks in grouped.values():
            chunks.sort(key=lambda c: (c.get("chunk_index", -1), c.get("page") or 0))
        return dict(sorted(grouped.items()))

    def retrieve(self, query: str, top_k: int = TOP_K) -> tuple[list[dict], dict]:
        """Retrieve chunks: overfetch with embeddings, then rerank with cross-encoder."""
        if self.collection is None or self.collection.count() == 0:
            return [], {"embed": 0, "search": 0, "rerank": 0}

        timings = {}
        self._last_retrieval_k = RETRIEVAL_K

        t0 = time.time()
        query_embedding = self.encode_query(query).tolist()
        timings["embed"] = time.time() - t0

        n_results = min(RETRIEVAL_K, self.collection.count())

        t0 = time.time()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        timings["search"] = time.time() - t0

        candidates = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            candidates.append({
                "text": doc,
                "source": meta.get("source", "unknown"),
                "chunk_index": meta.get("chunk_index", -1),
                "page": meta.get("page"),
                "distance": dist,
            })

        if not candidates:
            return [], timings

        t0 = time.time()
        pairs = [[query, c["text"]] for c in candidates]
        scores = self.reranker.predict(pairs)
        timings["rerank"] = time.time() - t0

        for c, score in zip(candidates, scores):
            c["rerank_score"] = float(score)

        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates[:top_k], timings

    def build_context(self, chunks: list[dict]) -> str:
        """Format retrieved chunks into a context block."""
        parts = []
        for i, c in enumerate(chunks):
            page = f", page {c['page']}" if c.get("page") else ""
            citation = f"[source: {c['source']}{page}, chunk {c['chunk_index']}]"
            parts.append(
                f"{citation} [relevance: {c['rerank_score']:.2f}]\n{c['text']}"
            )
        return "\n\n---\n\n".join(parts)

    def generate_stream(self, query: str, chunks: list[dict], history: list[dict] = None, mode: str = "RAG Only"):
        """Stream tokens from the LLM given pre-retrieved chunks.

        Yields (delta_text, usage_or_none) tuples. The final yield includes
        usage stats from the server (prompt_tokens, completion_tokens, etc.).
        """
        context = self.build_context(chunks)

        augmented_prompt = (
            f"### Retrieved Context\n\n{context}\n\n"
            f"### User Question\n\n{query}"
        )

        system_prompt = SYSTEM_PROMPT_RAG_PLUS_MODEL if mode == "RAG + Model" else SYSTEM_PROMPT_RAG_ONLY
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": augmented_prompt})

        response = self.llm.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            stream=True,
            stream_options={"include_usage": True},
        )

        usage = None
        for chunk in response:
            if chunk.usage:
                usage = {
                    "prompt_tokens": chunk.usage.prompt_tokens,
                    "completion_tokens": chunk.usage.completion_tokens,
                    "total_tokens": chunk.usage.total_tokens,
                }
            if chunk.choices:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta, usage
                    usage = None
        # Yield final usage if it arrived after last content delta
        if usage:
            yield "", usage

    def generate_stream_direct(self, query: str, history: list[dict] = None):
        """Stream tokens directly from the LLM without RAG context.

        Yields (delta_text, usage_or_none) tuples.
        """
        messages = [{"role": "system", "content": "You are an expert analyst. Answer the user's question using your own knowledge. Be precise and thorough."}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": query})

        response = self.llm.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            stream=True,
            stream_options={"include_usage": True},
        )

        usage = None
        for chunk in response:
            if chunk.usage:
                usage = {
                    "prompt_tokens": chunk.usage.prompt_tokens,
                    "completion_tokens": chunk.usage.completion_tokens,
                    "total_tokens": chunk.usage.total_tokens,
                }
            if chunk.choices:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta, usage
                    usage = None
        if usage:
            yield "", usage

    def generate_full(self, query: str, history: list[dict] = None, mode: str = "RAG Only") -> tuple[str, list[dict]]:
        """Non-streaming version of generate."""
        chunks, _ = self.retrieve(query)
        context = self.build_context(chunks)

        augmented_prompt = (
            f"### Retrieved Context\n\n{context}\n\n"
            f"### User Question\n\n{query}"
        )

        system_prompt = SYSTEM_PROMPT_RAG_PLUS_MODEL if mode == "RAG + Model" else SYSTEM_PROMPT_RAG_ONLY
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": augmented_prompt})

        response = self.llm.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )

        answer = response.choices[0].message.content
        return answer, chunks
