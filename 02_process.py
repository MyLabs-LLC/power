#!/usr/bin/env python3
"""
Step 2: Tokenize downloaded documents and save as training-ready Arrow dataset.

Reads parquet from step 1, tokenizes every document into fixed-length chunks,
and writes an Arrow dataset that step 3 can memory-map directly.

Usage:
    python 02_process.py                                          # auto-detect latest download
    python 02_process.py --input data/pretrain/fineweb_edu_sample-10BT_score5.parquet
    python 02_process.py --seq-len 2048                           # longer context
"""

import argparse
import json
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = ROOT / "data" / "pretrain"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "tokenized"


def parse_args():
    p = argparse.ArgumentParser(description="Tokenize documents into training chunks")
    p.add_argument("--input", type=str, default=None,
                    help="Input parquet from step 1 (auto-detected if omitted)")
    p.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                    help="Output directory for tokenized Arrow files")
    p.add_argument("--seq-len", type=int, default=1024,
                    help="Sequence length for training chunks")
    p.add_argument("--tokenizer", type=str, default="gpt2",
                    help="HuggingFace tokenizer name")
    p.add_argument("--force", action="store_true",
                    help="Overwrite existing output")
    return p.parse_args()


def find_latest_parquet(input_dir: Path) -> Path:
    parquets = sorted(input_dir.glob("*.parquet"), key=lambda p: p.stat().st_mtime)
    if not parquets:
        raise FileNotFoundError(
            f"No parquet files found in {input_dir}. Run 01_download.py first."
        )
    return parquets[-1]


def tokenize_and_chunk(texts, tokenizer, seq_len: int, eos_id: int):
    """Tokenize documents and yield fixed-length (input_ids, labels) chunks."""
    chunk_span = seq_len + 1

    for text in texts:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if not token_ids:
            continue
        token_ids.append(eos_id)

        for start in range(0, len(token_ids) - 1, seq_len):
            chunk = token_ids[start : start + chunk_span]
            if len(chunk) < 2:
                break

            input_ids = chunk[:-1]
            labels = chunk[1:]
            valid = len(input_ids)
            pad_len = seq_len - valid

            if pad_len > 0:
                input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
                labels = labels + [-100] * pad_len

            yield input_ids, labels


def main():
    args = parse_args()

    if args.input:
        input_path = Path(args.input)
        if not input_path.is_absolute():
            input_path = ROOT / input_path
    else:
        input_path = find_latest_parquet(DEFAULT_INPUT_DIR)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = input_path.stem
    out_path = out_dir / f"{stem}_seq{args.seq_len}.arrow"
    stats_path = out_dir / f"{stem}_seq{args.seq_len}_stats.json"

    print("=" * 60)
    print("Step 2: Tokenize & Chunk")
    print("=" * 60)
    print(f"  Input    : {input_path}")
    print(f"  Tokenizer: {args.tokenizer}")
    print(f"  Seq len  : {args.seq_len}")
    print(f"  Output   : {out_path}")
    print("=" * 60)

    if out_path.exists() and not args.force:
        print(f"\n✅ Already exists: {out_path}")
        print(f"   Use --force to re-process.")
        return

    if args.seq_len < 1:
        raise ValueError("--seq-len must be >= 1")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 10_000_000

    print(f"\nReading {input_path.name}...")
    table = pq.read_table(str(input_path), columns=["text"])
    texts = table.column("text").to_pylist()
    print(f"  {len(texts):,} documents loaded")

    t0 = time.time()
    print("Tokenizing and chunking...")

    all_input_ids = []
    all_labels = []

    for input_ids, labels in tqdm(
        tokenize_and_chunk(texts, tokenizer, args.seq_len, tokenizer.eos_token_id),
        desc="Chunks",
        unit="chunk",
    ):
        all_input_ids.append(input_ids)
        all_labels.append(labels)

    n_chunks = len(all_input_ids)
    total_tokens = sum(
        sum(1 for lbl in row if lbl != -100)
        for row in all_labels
    )

    print(f"\nWriting {n_chunks:,} chunks to Arrow...")
    arrow_table = pa.table({
        "input_ids": pa.array(all_input_ids, type=pa.list_(pa.int32())),
        "labels": pa.array(all_labels, type=pa.list_(pa.int32())),
    })

    writer = pa.ipc.new_file(str(out_path), arrow_table.schema)
    writer.write_table(arrow_table)
    writer.close()

    elapsed = time.time() - t0
    size_mb = out_path.stat().st_size / (1024 ** 2)

    stats = {
        "input_parquet": str(input_path),
        "tokenizer": args.tokenizer,
        "seq_len": args.seq_len,
        "num_documents": len(texts),
        "num_chunks": n_chunks,
        "total_training_tokens": total_tokens,
        "arrow_path": str(out_path),
        "size_mb": round(size_mb, 1),
        "process_seconds": round(elapsed, 1),
    }
    stats_path.write_text(json.dumps(stats, indent=2))

    print(f"\n✅ Processed {len(texts):,} docs → {n_chunks:,} chunks")
    print(f"   Training tokens: {total_tokens:,}")
    print(f"   Size: {size_mb:.1f} MB")
    print(f"   Time: {elapsed:.0f}s")
    print(f"\nNext step: python 03_train_1b.py --data {out_path}")


if __name__ == "__main__":
    main()
