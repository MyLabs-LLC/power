# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLM pretraining pipeline for a scaled-down Nemotron-3 Super model. Three sequential scripts form the pipeline:

1. **`01_download.py`** — Downloads and filters FineWeb-Edu dataset (level-5 quality docs) → saves parquet to `data/pretrain/`
2. **`02_process.py`** — Tokenizes documents into fixed-length chunks → saves Arrow files to `data/tokenized/`
3. **`03_train_1b.py`** — Trains a 1B-parameter Nemotron model using PyTorch DDP (`torchrun` for multi-GPU) with bitsandbytes 8-bit AdamW
4. **`04_inference.py`** — Evaluates model quality (perplexity on training chunks) then launches interactive text-completion chat

## Running the Pipeline

```bash
# Step 1: Download data (requires HF_TOKEN in .env for speed)
python 01_download.py --max-docs 5000  # small test run

# Step 2: Tokenize (auto-detects latest parquet from step 1)
python 02_process.py

# Step 3: Train (auto-detects latest .arrow from step 2, auto-detects GPUs)
python 03_train_1b.py --max-steps 500 --batch-size 2  # quick test

# Step 4: Evaluate + chat (auto-detects latest checkpoint + data)
python 04_inference.py
python 04_inference.py --skip-eval                     # skip quality eval
```

Each step auto-detects output from the previous step. Use `--force` to re-run.

## Architecture

**`nemotron3.py`** — Core model implementation with two variants:
- `Nemotron3Nano` (30B-A3B): Mamba-Transformer hybrid with standard sparse MoE (128 experts, top-6)
- `Nemotron3Super` (120B-A12B): Adds Latent MoE (512 experts, top-22, 4096→1024 bottleneck), more GQA heads, and multi-token prediction (MTP)

The training config (`Nemotron1BConfig` in `03_train_1b.py`) is a scaled-down Super variant: d_model=1536, 40 experts (top-5), d_latent=640, d_ff=2560, 20 layers (two macro-blocks), vocab 50304, default `max_seq_len`/training `--seq-len` 1024. Match `--seq-len` in `04_inference.py` to the checkpoint. Training supports `--val-split` and `--eval-every` for validation.

**Layer system**: Models are built from macro-block specs — lists of `(repeat, [layer_types])` where types are `"m"` (Mamba-2), `"a"` (GQA attention), `"e"` (MoE/LatentMoE). `HybridLayer` wraps each with pre-norm + residual. Whether MoE uses latent bottleneck is determined by presence of `d_latent` in the config.

## Key Dependencies

- `torch`, `bitsandbytes`, `transformers`, `datasets`, `pyarrow`, `tqdm`
- Training uses AMP (bfloat16 when supported, else float16 + GradScaler), gradient checkpointing optional via `--gradient-checkpointing`
- Checkpoints saved to `checkpoints/` directory

## Data Flow

- `data/pretrain/*.parquet` — raw filtered documents
- `data/tokenized/*.arrow` — tokenized fixed-length chunks (input_ids + labels as int32 lists)
- `checkpoints/step_N/model.pt` and `checkpoints/final/model.pt` — model state dicts
- `logs/` — optional logs from `run_*.py` orchestration scripts
