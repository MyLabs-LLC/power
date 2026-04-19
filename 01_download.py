#!/usr/bin/env python3
"""
Step 1: Download FineWeb-Edu dataset and filter to level-5 documents.

Saves filtered parquet files to data/pretrain/ for the next pipeline step.

Usage:
    python 01_download.py                         # default: sample-10BT, level 5
    python 01_download.py --config sample-100BT   # larger sample
    python 01_download.py --min-score 4            # include level 4+
    python 01_download.py --max-docs 5000          # cap document count (for testing)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from datasets import load_dataset, Dataset
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent


def load_env():
    env_path = ROOT / ".env"
    if env_path.exists():
        for raw_line in env_path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            v = value.strip().strip('"').strip("'")
            os.environ.setdefault(key.strip(), v)
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def parse_args():
    p = argparse.ArgumentParser(description="Download FineWeb-Edu level-5 data")
    p.add_argument("--config", default="sample-10BT",
                    help="HF dataset config (sample-10BT, sample-100BT, default)")
    p.add_argument("--min-score", type=int, default=5,
                    help="Minimum int_score to keep (default: 5 = highest quality)")
    p.add_argument("--max-docs", type=int, default=None,
                    help="Maximum documents to download (None = all)")
    p.add_argument("--output-dir", default="data/pretrain",
                    help="Output directory for parquet files")
    p.add_argument("--force", action="store_true",
                    help="Overwrite existing files")
    return p.parse_args()


def main():
    args = parse_args()
    hf_token = load_env()

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = f"fineweb_edu_{args.config}_score{args.min_score}"
    if args.max_docs:
        tag += f"_max{args.max_docs}"
    out_path = out_dir / f"{tag}.parquet"
    stats_path = out_dir / f"{tag}_stats.json"

    print("=" * 60)
    print("Step 1: Download FineWeb-Edu")
    print("=" * 60)
    print(f"  Dataset  : HuggingFaceFW/fineweb-edu")
    print(f"  Config   : {args.config}")
    print(f"  Min score: {args.min_score}")
    print(f"  Max docs : {args.max_docs or 'all'}")
    print(f"  Output   : {out_path}")
    print(f"  HF token : {'yes' if hf_token else 'NO (will be slow)'}")
    print("=" * 60)

    if out_path.exists() and not args.force:
        print(f"\n✅ Already exists: {out_path}")
        print(f"   Use --force to re-download.")
        return

    t0 = time.time()

    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name=args.config,
        split="train",
        streaming=True,
        token=hf_token,
    )

    filtered = ds.filter(lambda x: x.get("int_score", 0) >= args.min_score)

    print(f"\nDownloading documents with int_score >= {args.min_score}...")

    docs = []
    total_tokens = 0
    for doc in tqdm(filtered, desc="Downloading", unit="doc"):
        docs.append(doc)
        total_tokens += doc.get("token_count", 0)
        if args.max_docs and len(docs) >= args.max_docs:
            break

    if not docs:
        print("❌ No documents found matching the filter.")
        sys.exit(1)

    dataset = Dataset.from_list(docs)
    dataset.to_parquet(str(out_path))

    elapsed = time.time() - t0
    size_gb = out_path.stat().st_size / (1024 ** 3)

    stats = {
        "dataset": "HuggingFaceFW/fineweb-edu",
        "config": args.config,
        "min_score": args.min_score,
        "num_documents": len(docs),
        "total_tokens": total_tokens,
        "parquet_path": str(out_path),
        "size_gb": round(size_gb, 3),
        "download_seconds": round(elapsed, 1),
    }
    stats_path.write_text(json.dumps(stats, indent=2))

    print(f"\n✅ Downloaded {len(docs):,} documents ({total_tokens:,} tokens)")
    print(f"   Size: {size_gb:.2f} GB")
    print(f"   Time: {elapsed:.0f}s")
    print(f"   Saved to: {out_path}")
    print(f"\nNext step: python 02_process.py --input {out_path}")


if __name__ == "__main__":
    main()
