#!/usr/bin/env python3
"""
Scale-up training: more data + bigger model to combat overfitting.

Previous run: 7,430 docs, ~6.6M tokens, 0.65B params → PPL 1.03 (overfit)
This run: 500K docs, ~65M+ tokens, ~1.5B params → should generalize

Pipeline:
  1. Download 500K docs from FineWeb-Edu (level 5)
  2. Tokenize at seq_len=1024
  3. Train scaled-up model (d_model=2048, 20 layers, 48 experts)
     with 10% validation split for overfitting detection
  4. Evaluate on validation set

Monitor: tail -f logs/scale_up.log
"""

import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "logs"
LOGS.mkdir(exist_ok=True)
LOG_FILE = LOGS / "scale_up.log"
HISTORY_FILE = LOGS / "scale_up_history.json"


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def run_cmd(cmd, log_path, timeout=14400):
    """Run command, stream to log file, return success + output."""
    with open(log_path, "w") as lf:
        proc = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            cwd=str(ROOT),
        )
        output_lines = []
        for raw in iter(proc.stdout.readline, b""):
            text = raw.decode("utf-8", errors="replace")
            lf.write(text)
            lf.flush()
            output_lines.append(text)
            if any(k in text for k in [
                "Step ", "Training complete", "Perplexity", "Quality grade",
                "Error", "Downloaded", "Processed", "VAL", "chunks", "tokens",
            ]):
                log(f"    {text.rstrip()}")
        proc.wait(timeout=timeout)
    return proc.returncode == 0, "".join(output_lines)


def extract_ppl(output):
    """Extract perplexity from inference output."""
    match = re.search(r"Perplexity\s*:\s*([\d.]+)", output)
    if match:
        return float(match.group(1))
    return float("inf")


def extract_val_ppl(output):
    """Extract the last validation PPL from training output."""
    matches = re.findall(r"\[VAL\].*?val_ppl\s+([\d.]+)", output)
    if matches:
        return float(matches[-1])
    return float("inf")


def save_history(history):
    HISTORY_FILE.write_text(json.dumps(history, indent=2))


def main():
    log("=" * 70)
    log("SCALE-UP TRAINING: MORE DATA + BIGGER MODEL")
    log("=" * 70)
    log("Previous: 7.4K docs, 6.6M tokens, 0.65B params → PPL 1.03 (overfit)")
    log("Target:   500K docs (score>=4, 100BT), ~500M+ tokens, 1.29B params")
    log("=" * 70)

    history = []

    # ─── STEP 1: Download more data ─────────────────────────────────────
    # sample-10BT only has ~7K level-5 docs. Use sample-100BT with min-score 4
    parquet_path = ROOT / "data/pretrain/fineweb_edu_sample-100BT_score4_max500000.parquet"

    if parquet_path.exists():
        log(f"\nStep 1: SKIP — data already exists: {parquet_path}")
    else:
        log(f"\n{'='*70}")
        log("STEP 1: Download 500K documents from FineWeb-Edu (score >= 4, sample-100BT)")
        log(f"{'='*70}")

        ok, output = run_cmd(
            "python 01_download.py --config sample-100BT --min-score 4 --max-docs 500000",
            str(LOGS / "scale_up_download.log"),
            timeout=14400,  # 4 hours for large download
        )
        if not ok:
            log("FATAL: Download failed!")
            sys.exit(1)

        if not parquet_path.exists():
            log(f"FATAL: Expected parquet not found at {parquet_path}")
            sys.exit(1)

        log("Step 1: Download complete!")

    # ─── STEP 2: Tokenize with seq_len=1024 ─────────────────────────────
    arrow_path = ROOT / "data/tokenized/fineweb_edu_sample-100BT_score4_max500000_seq1024.arrow"

    if arrow_path.exists():
        log(f"\nStep 2: SKIP — tokenized data already exists: {arrow_path}")
    else:
        log(f"\n{'='*70}")
        log("STEP 2: Tokenize (seq_len=1024)")
        log(f"{'='*70}")

        ok, output = run_cmd(
            f"python 02_process.py --input {parquet_path} --seq-len 1024",
            str(LOGS / "scale_up_tokenize.log"),
            timeout=3600,
        )
        if not ok:
            log("FATAL: Tokenization failed!")
            sys.exit(1)

        log("Step 2: Tokenization complete!")

    # ─── STEP 3: Train with progressive schedule ────────────────────────
    log(f"\n{'='*70}")
    log("STEP 3: Train ~1.5B model with validation")
    log(f"{'='*70}")
    log("  Architecture: d_model=1792, d_ff=2560, 16 layers, 40 experts (top-5)")
    log("  d_latent=640, n_q_heads=12, n_kv_heads=4, mamba_d_state=48")
    log("  Validation split: 10%")

    # Check if we have a checkpoint to resume from (new arch = start fresh)
    final_ckpt = ROOT / "checkpoints/final/model.pt"

    # Training phases: progressive LR schedule
    # With ~65M tokens, seq=1024, batch=2*2GPUs, grad_accum=8 → 32 seqs/step → 32K tok/step
    # ~2000 steps = 1 epoch. Let's do ~5 epochs = 10000 steps
    # With ~543M tokens, seq=1024, batch=1*2*2=4 seqs/step = 4K tok/step
    # 1 epoch ~= 180K steps. We'll train on ~3-5% of data per phase.
    # At ~12s/step → ~300 steps/hr → 7000 steps in ~23 hours
    train_phases = [
        # (tag, max_steps, lr, warmup, batch_size, grad_accum, resume_flag)
        ("phase1_warmup",  2000, 3e-4,  200, 1, 2, False),    # warmup + initial
        ("phase2_main",    3000, 1.5e-4, 100, 1, 2, True),    # main training
        ("phase3_anneal",  2000, 5e-5,   50, 1, 2, True),     # annealing
    ]

    best_val_ppl = float("inf")

    for tag, max_steps, lr, warmup, bs, ga, resume in train_phases:
        log(f"\n{'='*70}")
        log(f"TRAINING: {tag} — {max_steps} steps, LR={lr}, bs={bs}*{ga}*2={bs*ga*2}")
        log(f"{'='*70}")

        resume_arg = f"--resume {final_ckpt}" if resume and final_ckpt.exists() else ""
        save_every = max(max_steps // 4, 500)
        log_every = max(max_steps // 20, 10)
        eval_every = max(max_steps // 4, 250)

        cmd = (
            f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
            f"torchrun --nproc_per_node=2 03_train_1b.py "
            f"--data {arrow_path} "
            f"--max-steps {max_steps} --batch-size {bs} --grad-accum {ga} "
            f"--lr {lr} --warmup-steps {warmup} "
            f"--log-every {log_every} --save-every {save_every} "
            f"--eval-every {eval_every} "
            f"--gradient-checkpointing --seq-len 1024 "
            f"--val-split 0.1 "
            f"{resume_arg}"
        )

        ok, output = run_cmd(
            cmd,
            str(LOGS / f"scale_up_{tag}_train.log"),
            timeout=14400,  # 4 hours per phase
        )

        if not ok:
            log(f"  {tag} FAILED! Trying with smaller batch...")
            # Retry with halved batch
            cmd = cmd.replace(f"--batch-size {bs}", f"--batch-size {max(1, bs//2)}")
            cmd = cmd.replace(f"--grad-accum {ga}", f"--grad-accum {ga*2}")
            ok, output = run_cmd(
                cmd,
                str(LOGS / f"scale_up_{tag}_train_retry.log"),
                timeout=14400,
            )
            if not ok:
                log(f"  {tag} still failing, stopping training.")
                break

        # Extract validation PPL from training output
        val_ppl = extract_val_ppl(output)

        # Also run formal eval on validation data
        log(f"\n  Evaluating {tag}...")
        eval_cmd = (
            f"python 04_inference.py "
            f"--eval-samples 200 --data {arrow_path} < /dev/null"
        )
        eval_ok, eval_output = run_cmd(
            eval_cmd,
            str(LOGS / f"scale_up_{tag}_eval.log"),
        )
        train_ppl = extract_ppl(eval_output)

        history.append({
            "tag": tag,
            "steps": max_steps,
            "lr": lr,
            "val_ppl": val_ppl,
            "train_ppl": train_ppl,
        })
        save_history(history)

        log(f"  RESULT: {tag} — train_ppl={train_ppl:.2f}, val_ppl={val_ppl:.2f}")

        # Save best model based on val PPL
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            best_dir = ROOT / "checkpoints/best"
            best_dir.mkdir(parents=True, exist_ok=True)
            if final_ckpt.exists():
                shutil.copy2(str(final_ckpt), str(best_dir / "model.pt"))
                (best_dir / "ppl.txt").write_text(f"val_ppl={val_ppl:.2f} train_ppl={train_ppl:.2f}\n")
                log(f"  >>> NEW BEST: val_ppl={val_ppl:.2f}")

        # Early stop if val PPL starts going up significantly (overfitting)
        if len(history) >= 2:
            prev_val = history[-2].get("val_ppl", float("inf"))
            if val_ppl > prev_val * 1.1 and prev_val < float("inf"):
                log(f"  WARNING: Val PPL increased {prev_val:.2f} → {val_ppl:.2f} (overfitting detected)")
                log(f"  Restoring best checkpoint and stopping")
                best_ckpt = ROOT / "checkpoints/best/model.pt"
                if best_ckpt.exists():
                    shutil.copy2(str(best_ckpt), str(final_ckpt))
                break

    # ─── SUMMARY ─────────────────────────────────────────────────────────
    log(f"\n{'='*70}")
    log("SCALE-UP TRAINING COMPLETE")
    log(f"{'='*70}")
    log(f"  Best validation PPL: {best_val_ppl:.2f}")
    log(f"  Training phases completed: {len(history)}")
    total_steps = sum(h.get("steps", 0) for h in history)
    log(f"  Total training steps: {total_steps}")
    log(f"\n  History:")
    for h in history:
        log(f"    {h['tag']}: train_ppl={h['train_ppl']:.2f}, val_ppl={h['val_ppl']:.2f}")
    log(f"\n  Best model: checkpoints/best/model.pt")
    log(f"  Final model: checkpoints/final/model.pt")
    log(f"\n  To evaluate: python 04_inference.py")
    log(f"  To chat:     python 04_inference.py --skip-eval")


if __name__ == "__main__":
    main()
