"""Document ingestion pipeline with multi-dataset support.

Usage:
    python ingest.py CERN                           # Ingest all files in ./documents/CERN/
    python ingest.py CERN --reset                   # Clear and re-ingest CERN dataset
    python ingest.py CERN path/to/file.pdf          # Add a file to CERN dataset
    python ingest.py --all --reset                  # Re-index every dataset under ./documents/
"""

import os
import shutil
import sys
import hashlib
import contextlib
import gc
import io
import logging
import socket
import subprocess
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from config import (
    EMBEDDING_MODEL, VECTORSTORE_DIR, DOCUMENTS_DIR,
    CHUNK_SIZE, CHUNK_OVERLAP,
)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REQUIRED_CONDA_ENV = "rag"
DEFAULT_CUDA_BATCH_SIZE = 1024
DEFAULT_CPU_BATCH_SIZE = 128
BYTES_PER_GIB = 1024 ** 3
APP_DIR = Path(__file__).resolve().parent
START_SERVER_SCRIPT = APP_DIR / "start_server.sh"
STOP_SERVER_SCRIPT = APP_DIR / "stop_server.sh"

warnings.filterwarnings(
    "ignore",
    message=r"resource_tracker: There appear to be .* leaked semaphore objects.*",
)
warnings.filterwarnings("ignore", category=UserWarning, module=r"multiprocessing\.resource_tracker")


USAGE = """Usage:
  python ingest.py                         List available datasets
  python ingest.py <dataset_name> [--reset] [extra_files...]
  python ingest.py --all --reset
  python ingest.py -all --reset
  python ingest.py --all --reset --keep-server

Examples:
  python ingest.py CERN
  python ingest.py CERN --reset
  python ingest.py --all --reset

Speed controls:
  RAG_INGEST_DEVICE=cuda          Auto by default when CUDA is available
  RAG_INGEST_DEVICE=cpu           Force CPU embeddings
  RAG_INGEST_ALL_GPUS=0           Disable multi-GPU embedding workers
  RAG_INGEST_DEVICES=cuda:0,cuda:1 Override worker GPU list
  RAG_INGEST_BATCH_SIZE=1024      Override auto VRAM-based embedding batch size
  RAG_INGEST_VRAM_HEADROOM_GB=2   Free VRAM to leave unused per GPU
  RAG_INGEST_MAX_BATCH_SIZE=3072  Cap adaptive ramp without disabling auto-detect
  RAG_INGEST_SUMMARY=0            Skip cached dataset summary creation after CLI ingest
  RAG_INGEST_MANAGE_SERVER=0      Do not stop/restart MyLabs Studio around CLI ingest
"""


def ensure_ingest_environment() -> None:
    """Run ingestion commands inside the project conda environment."""
    if os.environ.get("CONDA_DEFAULT_ENV") == REQUIRED_CONDA_ENV:
        return

    conda = shutil.which("conda")
    if not conda:
        print(
            "Ingestion requires the 'rag' conda environment.\n"
            "Run:\n"
            "  conda activate rag\n"
            f"  python {' '.join(sys.argv)}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Switching to conda env '{REQUIRED_CONDA_ENV}' for ingestion...", file=sys.stderr)
    os.execvp(conda, [conda, "run", "-n", REQUIRED_CONDA_ENV, "python", *sys.argv])


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in ("0", "false", "no", "off")


def port_is_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.25)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def project_stack_is_running() -> bool:
    return port_is_open(8001) or port_is_open(7860)


def run_script_streaming(command: list[str]):
    """Run a project script and yield combined stdout/stderr lines."""
    process = subprocess.Popen(
        command,
        cwd=APP_DIR,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip()
        if line:
            yield line
    code = process.wait()
    if code != 0:
        raise RuntimeError(f"{' '.join(command)} exited with code {code}")


def stop_project_stack_for_ingest():
    """Stop MyLabs Studio for CLI ingestion and report whether it should restart."""
    if not env_flag("RAG_INGEST_MANAGE_SERVER", default=True):
        yield "Server lifecycle management disabled by RAG_INGEST_MANAGE_SERVER=0."
        return False
    if not project_stack_is_running():
        yield "MyLabs Studio is not running; no server shutdown needed."
        return False
    if not STOP_SERVER_SCRIPT.exists():
        yield "stop_server.sh not found; leaving server state unchanged."
        return False

    yield "Stopping MyLabs Studio to free GPU memory for ingestion..."
    for update in run_script_streaming([str(STOP_SERVER_SCRIPT)]):
        yield update
    yield "GPU memory released for ingestion."
    return True


def restart_project_stack_after_ingest():
    """Restart MyLabs Studio after CLI ingestion."""
    if not START_SERVER_SCRIPT.exists():
        yield "start_server.sh not found; MyLabs Studio was not restarted."
        return
    yield "Restarting MyLabs Studio after ingestion..."
    for update in run_script_streaming([str(START_SERVER_SCRIPT)]):
        yield update


def load_file(path: str) -> str:
    """Extract text from a file based on extension."""
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        return "\n\n".join(text for _, _, _, text in iter_pdf_page_texts(path))

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


def choose_embedding_device() -> str:
    """Pick the fastest available embedding device unless overridden."""
    requested = os.environ.get("RAG_INGEST_DEVICE") or os.environ.get("INGEST_DEVICE")
    if requested:
        return requested

    try:
        import torch
        if torch.cuda.is_available():
            best_idx = 0
            best_free = -1
            for idx in range(torch.cuda.device_count()):
                try:
                    free_mem, _ = torch.cuda.mem_get_info(idx)
                except Exception:
                    free_mem = 0
                if free_mem > best_free:
                    best_idx = idx
                    best_free = free_mem
            return f"cuda:{best_idx}"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def cuda_devices() -> list[str]:
    requested_devices = os.environ.get("RAG_INGEST_DEVICES")
    if requested_devices:
        return [d.strip() for d in requested_devices.split(",") if d.strip()]
    try:
        import torch
        if torch.cuda.is_available():
            return [f"cuda:{idx}" for idx in range(torch.cuda.device_count())]
    except Exception:
        pass
    return []


def cuda_device_index(device: str) -> int:
    if device == "cuda":
        return 0
    if device.startswith("cuda:"):
        try:
            return int(device.split(":", 1)[1])
        except ValueError:
            return 0
    return 0


def gpu_memory_info(devices: list[str] | None = None) -> list[dict]:
    """Return current free/total VRAM for selected CUDA devices."""
    try:
        import torch
        if not torch.cuda.is_available():
            return []
        selected = devices or [f"cuda:{idx}" for idx in range(torch.cuda.device_count())]
        info = []
        for device in selected:
            idx = cuda_device_index(device)
            with torch.cuda.device(idx):
                free_bytes, total_bytes = torch.cuda.mem_get_info(idx)
            info.append(
                {
                    "device": f"cuda:{idx}",
                    "name": torch.cuda.get_device_name(idx),
                    "free_bytes": free_bytes,
                    "total_bytes": total_bytes,
                }
            )
        return info
    except Exception:
        return []


def format_gib(value: int | float) -> str:
    return f"{value / BYTES_PER_GIB:.1f} GiB"


def vram_summary(devices: list[str] | None = None) -> str:
    details = gpu_memory_info(devices)
    if not details:
        return "No CUDA VRAM detected."
    return "; ".join(
        f"{item['device']} {format_gib(item['free_bytes'])} free / {format_gib(item['total_bytes'])} total"
        for item in details
    )


def vram_based_batch_size(devices: list[str]) -> int:
    details = gpu_memory_info(devices)
    if not details:
        return DEFAULT_CUDA_BATCH_SIZE
    try:
        headroom_gb = float(os.environ.get("RAG_INGEST_VRAM_HEADROOM_GB", "2"))
    except ValueError:
        headroom_gb = 2.0

    min_free_gb = min(item["free_bytes"] for item in details) / BYTES_PER_GIB
    usable_gb = max(0.0, min_free_gb - headroom_gb)

    model_name = EMBEDDING_MODEL.lower()
    if "qwen3-embedding-8b" in model_name:
        if usable_gb >= 16:
            return 64
        if usable_gb >= 8:
            return 32
        return 16
    if "qwen3-embedding-4b" in model_name:
        if usable_gb >= 16:
            return 128
        if usable_gb >= 8:
            return 64
        return 32
    if "qwen3-embedding-0.6b" in model_name or "qwen3-embedding" in model_name:
        if usable_gb >= 16:
            return 3072
        if usable_gb >= 8:
            return 2048
        if usable_gb >= 4:
            return 1024
        return 128

    if usable_gb >= 20:
        return 8192
    if usable_gb >= 12:
        return 4096
    if usable_gb >= 8:
        return 3072
    if usable_gb >= 5:
        return 2048
    if usable_gb >= 2.5:
        return 1024
    if usable_gb >= 1.25:
        return 512
    return 256


def embedding_batch_size(device_or_devices: str | list[str]) -> int:
    """Choose embedding batch size from available VRAM, with an env override."""
    override = os.environ.get("RAG_INGEST_BATCH_SIZE") or os.environ.get("INGEST_BATCH_SIZE")
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            pass
    max_override = os.environ.get("RAG_INGEST_MAX_BATCH_SIZE")
    max_batch_size = None
    if max_override:
        try:
            max_batch_size = max(1, int(max_override))
        except ValueError:
            pass
    if isinstance(device_or_devices, list):
        batch_size = vram_based_batch_size(device_or_devices) if device_or_devices else DEFAULT_CPU_BATCH_SIZE
    elif device_or_devices.startswith("cuda"):
        batch_size = vram_based_batch_size([device_or_devices])
    else:
        batch_size = DEFAULT_CUDA_BATCH_SIZE if device_or_devices.startswith("mps") else DEFAULT_CPU_BATCH_SIZE
    return min(batch_size, max_batch_size) if max_batch_size else batch_size


def max_embedding_batch_size(start_batch_size: int) -> int:
    """Upper cap for adaptive GPU batch ramp-up."""
    override = os.environ.get("RAG_INGEST_MAX_BATCH_SIZE")
    if override:
        try:
            return max(start_batch_size, int(override))
        except ValueError:
            pass

    model_name = EMBEDDING_MODEL.lower()
    if "qwen3-embedding-0.6b" in model_name or "qwen3-embedding" in model_name:
        return max(start_batch_size, 4096)
    if "qwen3-embedding-4b" in model_name:
        return max(start_batch_size, 256)
    if "qwen3-embedding-8b" in model_name:
        return max(start_batch_size, 128)
    return max(start_batch_size, start_batch_size * 2)


def is_cuda_oom(error: Exception) -> bool:
    message = str(error).lower()
    return "out of memory" in message or "cuda" in message and "memory" in message


def clear_device_cache(device: str) -> None:
    if not device.startswith("cuda"):
        return
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def clear_all_cuda_caches() -> None:
    try:
        import torch
        if not torch.cuda.is_available():
            return
        for idx in range(torch.cuda.device_count()):
            with torch.cuda.device(idx):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
    except Exception:
        pass


def encode_texts(embedder, texts: list[str], batch_size: int, device: str):
    """Encode text with adaptive GPU batch-size fallback."""
    requested_batch_size = batch_size
    while True:
        try:
            embeddings = embedder.encode(
                texts,
                batch_size=min(requested_batch_size, len(texts)),
                show_progress_bar=False,
            ).tolist()
            return embeddings, requested_batch_size
        except RuntimeError as exc:
            if requested_batch_size <= 16 or not is_cuda_oom(exc):
                raise
            clear_device_cache(device)
            requested_batch_size = max(16, requested_batch_size // 2)


def encode_texts_multi_gpu(embedders: list[tuple[str, object]], texts: list[str], batch_size: int):
    """Encode text across local GPUs without multiprocessing semaphore leaks."""
    requested_batch_size = batch_size
    while True:
        try:
            worker_count = max(1, len(embedders))
            shard_size = max(1, (len(texts) + worker_count - 1) // worker_count)
            shards = [
                (idx, embedders[idx][0], embedders[idx][1], texts[idx * shard_size : (idx + 1) * shard_size])
                for idx in range(worker_count)
                if texts[idx * shard_size : (idx + 1) * shard_size]
            ]

            def encode_shard(item):
                idx, device, model, shard = item
                result = model.encode(
                    shard,
                    batch_size=min(requested_batch_size, len(shard)),
                    show_progress_bar=False,
                ).tolist()
                return idx, result

            ordered_results = []
            with ThreadPoolExecutor(max_workers=len(shards)) as executor:
                for idx, result in executor.map(encode_shard, shards):
                    ordered_results.append((idx, result))

            embeddings = []
            for _, result in sorted(ordered_results, key=lambda item: item[0]):
                embeddings.extend(result)
            return embeddings, requested_batch_size
        except RuntimeError as exc:
            if requested_batch_size <= 16 or not is_cuda_oom(exc):
                raise
            clear_all_cuda_caches()
            requested_batch_size = max(16, requested_batch_size // 2)


def load_embedding_model_quietly(model_name: str, device: str = "cpu", local_files_only: bool = False):
    """Load a SentenceTransformer while suppressing noisy HF/transformers startup logs."""
    from sentence_transformers import SentenceTransformer

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    captured = io.StringIO()
    previous_disable_level = logging.root.manager.disable
    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            logging.disable(logging.WARNING)
            try:
                return SentenceTransformer(
                    model_name,
                    device=device,
                    local_files_only=local_files_only,
                    token=token,
                )
            finally:
                logging.disable(previous_disable_level)


def iter_pdf_page_texts(path: str):
    """Yield PDF text per page using PyMuPDF first, with pypdf as a fallback."""
    try:
        import fitz
        doc = fitz.open(path)
        try:
            total_pages = doc.page_count
            for page_idx in range(total_pages):
                page_num = page_idx + 1
                try:
                    text = doc.load_page(page_idx).get_text("text") or ""
                except Exception:
                    text = ""
                yield "PyMuPDF", total_pages, page_num, text
            return
        finally:
            doc.close()
    except Exception:
        pass

    from pypdf import PdfReader
    reader = PdfReader(path)
    total_pages = len(reader.pages)
    for page_num, page in enumerate(reader.pages, start=1):
        yield "pypdf", total_pages, page_num, page.extract_text() or ""


def chunk_text(text: str, source: str, page_number: int | None = None, chunk_offset: int = 0) -> list[dict]:
    """Split text into overlapping chunks with metadata."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
    )
    chunks = splitter.split_text(text)
    output = []
    for i, c in enumerate(chunks):
        if not c.strip():
            continue
        chunk_index = chunk_offset + i
        metadata = {"source": os.path.basename(source), "chunk_index": chunk_index}
        if page_number is not None:
            metadata["page"] = page_number
        output.append(
            {
                "id": hashlib.sha256(f"{source}::{page_number or 0}::{chunk_index}::{c[:64]}".encode()).hexdigest()[:16],
                "text": c,
                "metadata": metadata,
            }
        )
    return output


def chunk_file_streaming(path: str):
    """Extract and chunk a file, yielding progress while preserving PDF page numbers."""
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        chunks = []
        header_sent = False
        total_pages = 0
        for extractor, total_pages, page_num, text in iter_pdf_page_texts(path):
            if not header_sent:
                yield f"  ⟶ PDF has {total_pages} pages; extracting text with {extractor}..."
                header_sent = True
            if page_num == 1 or page_num == total_pages or page_num % 10 == 0:
                yield f"  ⟶ extracting page {page_num}/{total_pages}"
            if not text.strip():
                continue
            page_chunks = chunk_text(text, path, page_number=page_num, chunk_offset=len(chunks))
            chunks.extend(page_chunks)
        if not header_sent:
            yield "  ⟶ PDF has 0 pages; no text extracted"
        return chunks

    text = load_file(path)
    return chunk_text(text, path) if text else []


def chunk_file(path: str) -> list[dict]:
    """Extract and chunk a file, preserving PDF page numbers when available."""
    stream = chunk_file_streaming(path)
    while True:
        try:
            next(stream)
        except StopIteration as done:
            return done.value


def ingest_dataset(dataset_name: str, paths: list[str], reset: bool = False):
    """Ingest documents into a dataset-specific ChromaDB collection."""
    import chromadb

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


def build_cached_summary_cli(dataset_name: str) -> None:
    """Build the cached dataset summary from the CLI after vector indexing."""
    if not env_flag("RAG_INGEST_SUMMARY", default=True):
        print("Skipping cached summary build because RAG_INGEST_SUMMARY=0.", flush=True)
        return
    if not port_is_open(8001):
        print(
            "Skipping cached summary build because the LLM server is not reachable on :8001. "
            "Start MyLabs Studio and use the UI re-index/upload flow, or run ./start_server.sh before rebuilding summaries.",
            flush=True,
        )
        return

    script = """
import sys
from app import build_dataset_summary

dataset = sys.argv[1]

def emit(_event, data):
    message = data.get("message") if isinstance(data, dict) else data
    if message:
        print(message, flush=True)

build_dataset_summary(dataset, emit)
"""
    print("Building cached dataset summary...", flush=True)
    process = subprocess.Popen(
        [sys.executable, "-c", script, dataset_name],
        cwd=APP_DIR,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip()
        if line:
            print(line, flush=True)
    code = process.wait()
    if code != 0:
        print(f"Cached summary build failed for {dataset_name} (exit {code}).", flush=True)


def ingest_dataset_cli(
    dataset_name: str,
    paths: list[str],
    reset: bool = False,
    use_all_gpus: bool = True,
    build_summary: bool = True,
) -> None:
    """Run ingestion and print streaming progress for CLI users."""
    for update in ingest_dataset_streaming(dataset_name, paths, reset=reset, use_all_gpus=use_all_gpus):
        print(update, flush=True)
    if build_summary:
        build_cached_summary_cli(dataset_name)


def print_generator_updates(generator):
    """Print generator updates and return the generator's final value."""
    while True:
        try:
            update = next(generator)
        except StopIteration as done:
            return done.value
        if update:
            print(update, flush=True)


def run_cli_ingest_with_server_lifecycle(callback, restart_after: bool = False) -> None:
    """Free GPU memory for CLI ingestion, then restore the app stack afterward."""
    stopped_running_stack = False
    try:
        stopped_running_stack = bool(print_generator_updates(stop_project_stack_for_ingest()))
        callback()
    finally:
        if restart_after or stopped_running_stack:
            print_generator_updates(restart_project_stack_after_ingest())


def dataset_dirs() -> list[str]:
    """List dataset folders under the documents root."""
    os.makedirs(DOCUMENTS_DIR, exist_ok=True)
    return sorted(
        name
        for name in os.listdir(DOCUMENTS_DIR)
        if os.path.isdir(os.path.join(DOCUMENTS_DIR, name)) and not name.startswith(".")
    )


def document_paths(dataset_name: str) -> list[str]:
    """List ingestible file paths for a dataset folder."""
    doc_dir = os.path.join(DOCUMENTS_DIR, dataset_name)
    os.makedirs(doc_dir, exist_ok=True)
    return [
        os.path.join(doc_dir, f)
        for f in sorted(os.listdir(doc_dir))
        if not f.startswith(".")
    ]


def ingest_dataset_streaming(dataset_name: str, paths: list[str], reset: bool = False, use_all_gpus: bool = False):
    """Ingest documents, yielding progress strings for each step."""
    import time
    import chromadb

    collection_name = f"rag_{dataset_name.lower().replace(' ', '_')}"

    requested_all_gpus = use_all_gpus and env_flag("RAG_INGEST_ALL_GPUS", default=True)
    devices = cuda_devices() if requested_all_gpus else []
    multi_gpu = requested_all_gpus and len(devices) > 1
    device = "cpu" if multi_gpu else choose_embedding_device()
    selected_devices = devices if multi_gpu else ([device] if device.startswith("cuda") else [])
    batch_size = embedding_batch_size(selected_devices if multi_gpu else device)
    max_batch_size = max_embedding_batch_size(batch_size)
    multi_gpu_embedders: list[tuple[str, object]] = []
    if selected_devices:
        yield f"Detected VRAM: {vram_summary(selected_devices)}"
    yield f"Loading embedding model on {'all local GPUs' if multi_gpu else device}..."
    try:
        embedder = load_embedding_model_quietly(EMBEDDING_MODEL, device=device, local_files_only=True)
        if multi_gpu:
            del embedder
            for target_device in devices:
                yield f"Loading embedding model on {target_device}..."
                multi_gpu_embedders.append(
                    (target_device, load_embedding_model_quietly(EMBEDDING_MODEL, device=target_device, local_files_only=True))
                )
            yield f"Embedding model loaded from local cache on {len(devices)} GPUs."
        else:
            yield f"Embedding model loaded from local cache on {device}."
    except Exception as exc:
        if device != "cpu" and is_cuda_oom(exc):
            yield f"GPU memory pressure while loading embeddings on {device}; falling back to CPU."
            device = "cpu"
            batch_size = embedding_batch_size(device)
            embedder = load_embedding_model_quietly(EMBEDDING_MODEL, device=device, local_files_only=True)
            yield "Embedding model loaded from local cache on cpu."
            yield f"Embedding batch size: {batch_size}"
        else:
            token_status = "using HF_TOKEN" if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") else "without HF_TOKEN"
            yield f"Embedding model not found in local cache; downloading from Hugging Face ({token_status})..."
            embedder = load_embedding_model_quietly(EMBEDDING_MODEL, device=device, local_files_only=False)
            if multi_gpu:
                del embedder
                for target_device in devices:
                    yield f"Loading embedding model on {target_device}..."
                    multi_gpu_embedders.append(
                        (target_device, load_embedding_model_quietly(EMBEDDING_MODEL, device=target_device, local_files_only=False))
                    )
                yield f"Embedding model downloaded from Hugging Face ({token_status}) and loaded on {len(devices)} GPUs."
            else:
                yield f"Embedding model downloaded from Hugging Face ({token_status}) and loaded on {device}."
            yield f"Embedding batch size: {batch_size}"
    else:
        if max_batch_size > batch_size:
            yield f"Embedding batch size: {batch_size}, adaptive max: {max_batch_size}"
        else:
            yield f"Embedding batch size: {batch_size}"

    try:
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
            chunks = yield from chunk_file_streaming(path)
            if not chunks:
                yield f"  ⟶ skipped (no text extracted)"
                continue

            total_chunks += len(chunks)
            yield f"  ⟶ {len(chunks)} chunks, embedding..."

            i = 0
            while i < len(chunks):
                batch = chunks[i : i + batch_size]
                batch_end = i + len(batch)
                texts = [c["text"] for c in batch]
                ids = [c["id"] for c in batch]
                metadatas = [c["metadata"] for c in batch]

                t0 = time.time()
                if multi_gpu_embedders:
                    embeddings, adjusted_batch_size = encode_texts_multi_gpu(multi_gpu_embedders, texts, batch_size)
                    if adjusted_batch_size != batch_size:
                        yield f"  ⟶ reduced embedding batch size to {adjusted_batch_size} after GPU memory pressure"
                        batch_size = adjusted_batch_size
                else:
                    embeddings, adjusted_batch_size = encode_texts(embedder, texts, batch_size, device)
                    if adjusted_batch_size != batch_size:
                        yield f"  ⟶ reduced embedding batch size to {adjusted_batch_size} after GPU memory pressure"
                        batch_size = adjusted_batch_size
                embed_time = time.time() - t0

                collection.upsert(
                    ids=ids,
                    documents=texts,
                    metadatas=metadatas,
                    embeddings=embeddings,
                )
                yield f"  ⟶ embedded batch {i+1}-{batch_end}/{len(chunks)} ({embed_time:.1f}s)"
                i = batch_end
                if selected_devices and batch_size < max_batch_size:
                    next_batch_size = min(max_batch_size, batch_size * 2)
                    yield f"  ⟶ increasing embedding batch size to {next_batch_size}"
                    batch_size = next_batch_size

        total = collection.count()
        yield f"Dataset '{dataset_name}': {total} chunks indexed ({total_chunks} new from {len(paths)} files)"
    finally:
        if multi_gpu_embedders:
            multi_gpu_embedders.clear()
        try:
            del embedder
        except Exception:
            pass
        gc.collect()
        clear_all_cuda_caches()


def main():
    if any(arg in ("-h", "--help") for arg in sys.argv[1:]):
        print(USAGE)
        sys.exit(0)

    if len(sys.argv) < 2:
        # List available datasets
        print("Available datasets:")
        for name in dataset_dirs():
            n_files = len(document_paths(name))
            print(f"  {name}/ ({n_files} files)")
        print(f"\n{USAGE}")
        sys.exit(0)

    reset = any(arg in ("--reset", "-reset") for arg in sys.argv)
    if "--keep-server" in sys.argv:
        os.environ["RAG_INGEST_MANAGE_SERVER"] = "0"

    if sys.argv[1] in ("--all", "-all", "all"):
        ensure_ingest_environment()
        names = dataset_dirs()
        if not names:
            print(f"No datasets found under {DOCUMENTS_DIR}/.")
            sys.exit(1)

        summary_queue: list[str] = []

        def ingest_all():
            for dataset_name in names:
                paths = document_paths(dataset_name)
                if not paths:
                    print(f"\nSkipping {dataset_name}: no documents found.")
                    continue
                print(f"\n=== Re-indexing {dataset_name} ({len(paths)} files) ===")
                ingest_dataset_cli(dataset_name, paths, reset=reset, build_summary=False)
                if env_flag("RAG_INGEST_SUMMARY", default=True):
                    summary_queue.append(dataset_name)

        run_cli_ingest_with_server_lifecycle(
            ingest_all,
            restart_after=env_flag("RAG_INGEST_MANAGE_SERVER", default=True),
        )
        for dataset_name in summary_queue:
            print(f"\n=== Building cached summary for {dataset_name} ===")
            build_cached_summary_cli(dataset_name)
        return

    dataset_name = sys.argv[1]
    ensure_ingest_environment()
    extra_paths = [a for a in sys.argv[2:] if a not in ("--reset", "-reset", "--keep-server")]

    # Gather files from the dataset folder
    doc_dir = os.path.join(DOCUMENTS_DIR, dataset_name)
    paths = document_paths(dataset_name)
    paths.extend(extra_paths)

    if not paths:
        print(f"No documents found in {doc_dir}/. Place files there or pass paths as arguments.")
        sys.exit(1)

    ingest_dataset_cli(dataset_name, paths, reset=reset, use_all_gpus=False)


if __name__ == "__main__":
    main()
