#!/usr/bin/env python3
"""
Step 3: Train 1B Nemotron model on tokenized data.

Supports single-GPU and multi-GPU via torchrun DDP.

Usage:
    python 03_train_1b.py                                  # single GPU
    torchrun --nproc_per_node=2 03_train_1b.py             # 2-GPU DDP
    python 03_train_1b.py --max-steps 500 --batch-size 2   # quick test run
    python 03_train_1b.py --resume checkpoints/final/model.pt
"""

import warnings
warnings.filterwarnings("ignore", message="functools.partial will be a method descriptor")

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pyarrow.ipc as ipc
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from nemotron3 import Nemotron3Super, NemotronSuperConfig, count_params, summarize_layers


# ─────────────────────────────────────────────────────────────────────────────
#  Model config (1B version)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Nemotron1BConfig(NemotronSuperConfig):
    d_model: int = 1536
    vocab_size: int = 50304
    norm_eps: float = 1e-6
    mamba_d_state: int = 48
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    n_q_heads: int = 12
    n_kv_heads: int = 4
    max_seq_len: int = 1024
    n_experts: int = 40
    n_shared_experts: int = 2
    top_k: int = 5
    d_ff: int = 2560
    d_latent: int = 640
    mtp_heads: int = 1
    macro_blocks: list = field(default_factory=lambda: [
        (3, ["e", "a", "m", "m"]),
        (2, ["e", "a", "m", "m"]),
    ])


# ─────────────────────────────────────────────────────────────────────────────
#  Arrow dataset
# ─────────────────────────────────────────────────────────────────────────────

class ArrowChunkDataset(Dataset):
    """Rows from a PyArrow table (or slice). Pass `table=` to avoid re-reading the file."""

    def __init__(self, arrow_path: Optional[str] = None, *, table=None, start_idx: int = 0, end_idx: Optional[int] = None):
        if table is not None:
            self.table = table
        else:
            if not arrow_path:
                raise ValueError("ArrowChunkDataset requires arrow_path or table=")
            reader = ipc.open_file(arrow_path)
            self.table = reader.read_all()
        n = len(self.table)
        self.start_idx = start_idx
        self.end_idx = end_idx if end_idx is not None else n
        if self.end_idx > n:
            self.end_idx = n
        if self.start_idx < 0 or self.start_idx > self.end_idx:
            raise ValueError(f"Invalid row range start_idx={self.start_idx} end_idx={self.end_idx} (n={n})")
        self.input_ids_col = self.table.column("input_ids")
        self.labels_col = self.table.column("labels")

    def __len__(self):
        return self.end_idx - self.start_idx

    def __getitem__(self, idx):
        real_idx = self.start_idx + idx
        return {
            "input_ids": torch.tensor(self.input_ids_col[real_idx].as_py(), dtype=torch.long),
            "labels": torch.tensor(self.labels_col[real_idx].as_py(), dtype=torch.long),
        }


# ─────────────────────────────────────────────────────────────────────────────
#  DDP helpers
# ─────────────────────────────────────────────────────────────────────────────

def setup_ddp():
    """Initialize DDP if launched via torchrun, otherwise single-GPU."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        return device, rank, world_size
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device, 0, 1


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank):
    return rank == 0


# ─────────────────────────────────────────────────────────────────────────────
#  Training
# ─────────────────────────────────────────────────────────────────────────────

def find_latest_arrow(search_dir: Path) -> Path:
    arrows = sorted(search_dir.glob("*.arrow"), key=lambda p: p.stat().st_mtime)
    if not arrows:
        raise FileNotFoundError(f"No .arrow files in {search_dir}. Run 02_process.py first.")
    return arrows[-1]


def parse_args():
    p = argparse.ArgumentParser(description="Train 1B Nemotron")
    p.add_argument("--data", type=str, default=None)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=2, help="Per-GPU micro batch size")
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-steps", type=int, default=100_000)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--val-split", type=float, default=0.0,
                    help="Fraction of data to hold out for validation (0 = no val)")
    p.add_argument("--eval-every", type=int, default=500,
                    help="Evaluate on validation set every N steps")
    return p.parse_args()


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


@torch.no_grad()
def run_validation(model, val_dataset, device, model_cfg, max_samples=200):
    """Compute validation loss/perplexity on held-out data."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    n_samples = min(max_samples, len(val_dataset))
    indices = torch.randperm(len(val_dataset))[:n_samples].tolist()
    criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")

    for idx in indices:
        batch = val_dataset[idx]
        input_ids = batch["input_ids"].unsqueeze(0).to(device)
        labels = batch["labels"].unsqueeze(0).to(device)
        n_valid = (labels != -100).sum().item()

        with torch.amp.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
            logits, _ = model(input_ids)
            loss = criterion(logits.reshape(-1, model_cfg.vocab_size), labels.reshape(-1))

        total_loss += loss.item()
        total_tokens += n_valid

    model.train()
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    return avg_loss, ppl


def main():
    load_env()
    args = parse_args()
    device, rank, world_size = setup_ddp()

    if not torch.cuda.is_available():
        if rank == 0:
            print("This training script requires CUDA (bitsandbytes 8-bit AdamW and AMP).", file=sys.stderr)
        if dist.is_initialized():
            dist.destroy_process_group()
        sys.exit(1)

    # Resolve data path
    if args.data:
        arrow_path = Path(args.data)
        if not arrow_path.is_absolute():
            arrow_path = ROOT / arrow_path
    else:
        arrow_path = find_latest_arrow(ROOT / "data" / "tokenized")

    if not arrow_path.exists():
        print(f"Data file not found: {arrow_path}")
        sys.exit(1)

    # Build model
    model_cfg = Nemotron1BConfig(max_seq_len=args.seq_len)
    model = Nemotron3Super(model_cfg)

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_absolute():
            resume_path = ROOT / resume_path
        state = torch.load(str(resume_path), map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        if is_main(rank):
            print(f"  Resumed from: {resume_path}")

    if args.gradient_checkpointing:
        model.set_gradient_checkpointing(True)

    model = model.to(device)

    # Wrap in DDP if multi-GPU
    if world_size > 1:
        model = DDP(model, device_ids=[device], find_unused_parameters=True)

    # Dataset & dataloader (single read of Arrow; optional validation split)
    full_reader = ipc.open_file(str(arrow_path))
    full_table = full_reader.read_all()
    total_chunks = len(full_table)

    val_dataset = None
    if args.val_split > 0:
        val_size = max(1, int(total_chunks * args.val_split))
        train_size = total_chunks - val_size
        train_table = full_table.slice(0, train_size)
        val_table = full_table.slice(train_size, val_size)
        dataset = ArrowChunkDataset(table=train_table)
        val_dataset = ArrowChunkDataset(table=val_table)
        if is_main(rank):
            print(f"  Train chunks: {train_size:,}  Val chunks: {val_size:,}")
    else:
        dataset = ArrowChunkDataset(table=full_table)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # Optimizer & scheduler
    import bitsandbytes as bnb
    from transformers import get_cosine_schedule_with_warmup

    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=args.lr, weight_decay=0.1)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    use_bf16 = torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = None if use_bf16 else torch.amp.GradScaler("cuda")

    params = count_params(model.module if hasattr(model, "module") else model)
    eff_batch = args.batch_size * args.grad_accum * world_size
    if is_main(rank):
        print("=" * 60)
        print("Step 3: Train Nemotron")
        print("=" * 60)
        print(f"  Parameters : {params['total'] / 1e9:.2f}B")
        print(f"  Architecture: {summarize_layers(model_cfg)}")
        print(f"  Dataset    : {len(dataset):,} chunks")
        print(f"  Seq len    : {args.seq_len}")
        print(f"  GPUs       : {world_size}")
        print(f"  Batch/GPU  : {args.batch_size}")
        print(f"  Grad accum : {args.grad_accum}")
        print(f"  Eff batch  : {eff_batch} ({eff_batch * args.seq_len:,} tokens/step)")
        print(f"  LR         : {args.lr}")
        print(f"  Max steps  : {args.max_steps}")
        print(f"  Grad ckpt  : {'on' if args.gradient_checkpointing else 'off'}")
        print(f"  AMP dtype  : {amp_dtype}")
        print("=" * 60)
        sys.stdout.flush()

    step = 0
    accum_loss = 0.0
    accum_count = 0
    tokens_seen = 0
    t_start = time.time()
    epoch = 0

    model.train()
    optimizer.zero_grad()

    while step < args.max_steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch += 1

        for batch in dataloader:
            if step >= args.max_steps:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            n_tokens = (labels != -100).sum().item()
            tokens_seen += n_tokens * world_size

            with torch.amp.autocast("cuda", dtype=amp_dtype):
                logits, mtp_logits = model(input_ids)
                loss = criterion(
                    logits.reshape(-1, model_cfg.vocab_size),
                    labels.reshape(-1),
                )
                if len(mtp_logits) > 0 and labels.shape[1] > 1:
                    mtp_labels = labels[:, 1:]
                    mtp_preds = mtp_logits[0][:, :-1, :]
                    mtp_loss = criterion(
                        mtp_preds.reshape(-1, model_cfg.vocab_size),
                        mtp_labels.reshape(-1),
                    )
                    loss = loss + 0.5 * mtp_loss
                loss = loss / args.grad_accum

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            accum_loss += loss.item() * args.grad_accum
            accum_count += 1

            if accum_count >= args.grad_accum:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                step += 1
                avg_loss = accum_loss / accum_count

                if is_main(rank) and step % args.log_every == 0:
                    elapsed = time.time() - t_start
                    tok_per_sec = tokens_seen / elapsed if elapsed > 0 else 0
                    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
                    print(
                        f"Step {step:>6d}/{args.max_steps} | "
                        f"loss {avg_loss:.4f} | ppl {ppl:.1f} | "
                        f"tok {tokens_seen:,} | "
                        f"{tok_per_sec:,.0f} tok/s"
                    )
                    sys.stdout.flush()

                if is_main(rank) and step % args.save_every == 0:
                    unwrapped = model.module if hasattr(model, "module") else model
                    ckpt_dir = ROOT / f"checkpoints/step_{step}"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(unwrapped.state_dict(), ckpt_dir / "model.pt")
                    print(f"  Checkpoint saved: {ckpt_dir / 'model.pt'}")
                    sys.stdout.flush()

                # Validation evaluation
                if val_dataset is not None and step % args.eval_every == 0 and is_main(rank):
                    unwrapped = model.module if hasattr(model, "module") else model
                    val_loss, val_ppl = run_validation(unwrapped, val_dataset, device, model_cfg)
                    print(
                        f"  [VAL] Step {step} | val_loss {val_loss:.4f} | val_ppl {val_ppl:.2f}"
                    )
                    sys.stdout.flush()

                accum_loss = 0.0
                accum_count = 0

                if step >= args.max_steps:
                    break

    # Final save (rank 0 only)
    if is_main(rank):
        final_dir = ROOT / "checkpoints/final"
        final_dir.mkdir(parents=True, exist_ok=True)
        unwrapped = model.module if hasattr(model, "module") else model
        torch.save(unwrapped.state_dict(), final_dir / "model.pt")
        elapsed = time.time() - t_start
        print(f"\nTraining complete — {step} steps in {elapsed:.0f}s")
        print(f"   Tokens processed: {tokens_seen:,}")
        print(f"   Final model: {final_dir / 'model.pt'}")
        sys.stdout.flush()

    cleanup_ddp()


if __name__ == "__main__":
    main()
