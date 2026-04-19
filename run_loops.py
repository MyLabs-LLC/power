#!/usr/bin/env python3
"""
Run training loops 2-10 with progressive improvements.
Each loop resumes from previous checkpoint and trains 100 more steps.

Monitor with: tail -f logs/pipeline.log
"""

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "logs"
LOGS.mkdir(exist_ok=True)

PIPELINE_LOG = LOGS / "pipeline.log"


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(PIPELINE_LOG, "a") as f:
        f.write(line + "\n")


def run(argv, log_file, timeout=1800, env=None):
    """Run argv (no shell), tee to log file, return (success, output_tail)."""
    log(f"  CMD: {' '.join(argv)}")
    log(f"  LOG: {log_file}")
    run_env = os.environ if env is None else {**os.environ, **env}
    with open(log_file, "w") as lf:
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(ROOT),
            env=run_env,
            stdin=subprocess.DEVNULL,
        )
        output = []
        try:
            for line in iter(proc.stdout.readline, b""):
                text = line.decode("utf-8", errors="replace")
                lf.write(text)
                lf.flush()
                output.append(text)
                # Print key lines to stdout
                if any(k in text for k in ["Step ", "loss ", "Training complete", "Perplexity", "Quality grade", "Error", "Traceback"]):
                    print(f"  {text.rstrip()}", flush=True)
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            log("  TIMEOUT: subprocess killed")
    tail = "".join(output[-20:]).lower()
    return proc.returncode == 0, tail


def get_latest_checkpoint():
    final = ROOT / "checkpoints/final/model.pt"
    if final.exists():
        return str(final)
    return None


def run_eval(loop_num, data_path):
    log(f"=== EVAL Loop {loop_num} ===")
    argv = [
        sys.executable,
        str(ROOT / "04_inference.py"),
        "--eval-samples", "50",
        "--data", str(ROOT / data_path) if not os.path.isabs(data_path) else data_path,
    ]
    ok, tail = run(argv, str(LOGS / f"loop{loop_num:02d}_eval.log"), env={"CUDA_VISIBLE_DEVICES": "0"})
    return ok


def run_train(loop_num, extra_args=""):
    log(f"=== TRAIN Loop {loop_num} ===")
    ckpt = get_latest_checkpoint()
    argv = [
        sys.executable,
        str(ROOT / "03_train_1b.py"),
        "--max-steps", "100",
        "--log-every", "10",
        "--save-every", "100",
        "--gradient-checkpointing",
        "--seq-len", "256",
    ]
    if ckpt:
        argv.extend(["--resume", ckpt])
    if extra_args.strip():
        argv.extend(extra_args.split())
    ok, tail = run(argv, str(LOGS / f"loop{loop_num:02d}_train.log"), timeout=1800, env={"CUDA_VISIBLE_DEVICES": "0"})
    if "loss nan" in tail or "nan loss" in tail:
        log(f"  WARNING: NaN detected in loop {loop_num}!")
        return False
    return ok


def main():
    data_256 = "data/tokenized/fineweb_edu_sample-10BT_score5_max50000_seq256.arrow"

    log("=" * 60)
    log("STARTING IMPROVEMENT LOOPS 2-10")
    log("=" * 60)

    # Loop 2: Continue training with same LR, more grad accum for stability
    log("\n>>> Loop 2: Continue training, larger effective batch")
    run_train(2, "--batch-size 4 --grad-accum 8 --lr 5e-4 --warmup-steps 10")
    run_eval(2, data_256)

    # Loop 3: Lower LR for fine-tuning, higher weight decay
    log("\n>>> Loop 3: Lower LR for refinement")
    run_train(3, "--batch-size 4 --grad-accum 8 --lr 3e-4 --warmup-steps 5")
    run_eval(3, data_256)

    # Loop 4: Even lower LR, longer effective training
    log("\n>>> Loop 4: Fine-tune with LR 1e-4")
    run_train(4, "--batch-size 4 --grad-accum 8 --lr 1e-4 --warmup-steps 5")
    run_eval(4, data_256)

    # Loop 5: Slightly higher batch for better gradient estimates
    log("\n>>> Loop 5: Larger batch, moderate LR")
    run_train(5, "--batch-size 8 --grad-accum 4 --lr 2e-4 --warmup-steps 5")
    run_eval(5, data_256)

    # Loop 6: Push LR back up slightly with warmup
    log("\n>>> Loop 6: Warm restart with LR 4e-4")
    run_train(6, "--batch-size 4 --grad-accum 8 --lr 4e-4 --warmup-steps 20")
    run_eval(6, data_256)

    # Loop 7: Continue at moderate LR
    log("\n>>> Loop 7: Steady training LR 3e-4")
    run_train(7, "--batch-size 4 --grad-accum 8 --lr 3e-4 --warmup-steps 10")
    run_eval(7, data_256)

    # Loop 8: Lower LR cooldown
    log("\n>>> Loop 8: Cooldown LR 1.5e-4")
    run_train(8, "--batch-size 8 --grad-accum 4 --lr 1.5e-4 --warmup-steps 5")
    run_eval(8, data_256)

    # Loop 9: Final fine-tuning at very low LR
    log("\n>>> Loop 9: Fine-tune LR 5e-5")
    run_train(9, "--batch-size 8 --grad-accum 4 --lr 5e-5 --warmup-steps 0")
    run_eval(9, data_256)

    # Loop 10: Last push at minimal LR
    log("\n>>> Loop 10: Final push LR 2e-5")
    run_train(10, "--batch-size 8 --grad-accum 4 --lr 2e-5 --warmup-steps 0")
    run_eval(10, data_256)

    log("\n" + "=" * 60)
    log("ALL 10 LOOPS COMPLETE")
    log("=" * 60)


if __name__ == "__main__":
    main()
