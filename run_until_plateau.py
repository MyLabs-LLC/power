#!/usr/bin/env python3
"""
Progressive training: ramp steps, then evolve architecture when plateaued.

Phase 1: Ramp steps (200 → 400 → 800 → 1600 → 3200) with LR cycling
Phase 2: When steps hit max, start architecture experiments:
  - Adjust macro_blocks (more attention vs mamba layers)
  - Tune d_model, d_ff, n_experts, top_k
  - Adjust MoE latent bottleneck
  - Retrain with best config

Stops when PPL improvement <1.5% for 3 consecutive rounds.

Monitor: tail -f logs/plateau.log
"""

import json
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
LOG_FILE = LOGS / "plateau.log"

DATA_256 = ROOT / "data/tokenized/fineweb_edu_sample-10BT_score5_max50000_seq256.arrow"


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def run_cmd(cmd, log_path, timeout=7200):
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
            if any(k in text for k in ["Step ", "Training complete", "Perplexity", "Quality grade", "Error"]):
                log(f"    {text.rstrip()}")
        proc.wait(timeout=timeout)
    return proc.returncode == 0, "".join(output_lines)


def train_run(tag, max_steps, lr, resume_from=None, extra_args=""):
    if resume_from == "NONE":
        resume = ""
    else:
        ckpt = resume_from or (ROOT / "checkpoints/final/model.pt")
        resume = f"--resume {ckpt}" if Path(ckpt).exists() else ""
    log_every = max(max_steps // 10, 1)
    warmup = max(max_steps // 20, 5)
    # Only include default batch-size if not overridden in extra_args
    batch_args = "" if "--batch-size" in extra_args else "--batch-size 8 --grad-accum 2"
    cmd = (
        f"torchrun --nproc_per_node=2 03_train_1b.py "
        f"--max-steps {max_steps} {batch_args} "
        f"--log-every {log_every} --save-every {max_steps} "
        f"--gradient-checkpointing --seq-len 256 "
        f"--lr {lr} --warmup-steps {warmup} "
        f"{resume} {extra_args}"
    )
    ok, output = run_cmd(cmd, str(LOGS / f"plateau_{tag}_train.log"))
    if "loss nan" in output.lower():
        log(f"  NaN detected!")
        return False
    return ok


def evaluate(tag):
    cmd = (
        f"python 04_inference.py "
        f"--eval-samples 100 --data {DATA_256} < /dev/null"
    )
    ok, output = run_cmd(cmd, str(LOGS / f"plateau_{tag}_eval.log"))
    match = re.search(r"Perplexity\s*:\s*([\d.]+)", output)
    if match:
        return float(match.group(1))
    return float("inf")


def save_best(ppl):
    best_dir = ROOT / "checkpoints/best"
    best_dir.mkdir(parents=True, exist_ok=True)
    src = ROOT / "checkpoints/final/model.pt"
    if src.exists():
        shutil.copy2(str(src), str(best_dir / "model.pt"))
        # Also save the PPL for reference
        (best_dir / "ppl.txt").write_text(f"{ppl:.2f}\n")
        log(f"  >>> NEW BEST: PPL={ppl:.2f} saved to checkpoints/best/")


def update_model_config(config_changes: dict):
    """Modify Nemotron1BConfig in 03_train_1b.py and 04_inference.py."""
    for filepath in ["03_train_1b.py", "04_inference.py"]:
        path = ROOT / filepath
        content = path.read_text()
        for key, value in config_changes.items():
            # Match "key: type = old_value" pattern in dataclass
            import re as _re
            pattern = rf"({key}\s*:\s*\w+\s*=\s*)([\w\d.e\-\+]+|\[.*?\])"
            if isinstance(value, list):
                replacement = rf"\g<1>{value}"
            else:
                replacement = rf"\g<1>{value}"
            content = _re.sub(pattern, replacement, content, flags=_re.DOTALL)
        path.write_text(content)
    log(f"  Config updated: {config_changes}")


def update_macro_blocks(new_blocks_str: str):
    """Replace macro_blocks in both config files."""
    for filepath in ["03_train_1b.py", "04_inference.py"]:
        path = ROOT / filepath
        content = path.read_text()
        # Find and replace macro_blocks field_default_factory
        import re as _re
        pattern = r"(macro_blocks:\s*list\s*=\s*field\(default_factory=lambda:\s*)\[.*?\](\s*\))"
        replacement = rf"\g<1>{new_blocks_str}\g<2>"
        content = _re.sub(pattern, replacement, content, flags=_re.DOTALL)
        path.write_text(content)
    log(f"  Macro blocks updated: {new_blocks_str}")


# ─────────────────────────────────────────────────────────────────────────────
#  Architecture experiments
# ─────────────────────────────────────────────────────────────────────────────

ARCH_EXPERIMENTS = [
    # ── Layer pattern experiments (same ~0.65B size) ──
    {
        "name": "more_attention_0.65B",
        "desc": "More attention layers for better pattern matching (0.65B)",
        "blocks": """[
        (2, ["e", "a", "m", "a"]),
        (1, ["e", "a", "m", "m"]),
    ]""",
        "config": {},
        "steps": 1600,
        "batch_size": 8,
    },
    {
        "name": "all_attention_0.65B",
        "desc": "Mostly attention, minimal mamba for short seqs (0.65B)",
        "blocks": """[
        (3, ["e", "a", "a", "a"]),
        (1, ["e", "a", "m", "m"]),
    ]""",
        "config": {},
        "steps": 1600,
        "batch_size": 8,
    },
    # ── Scale up: 0.78B (deeper) ──
    {
        "name": "deeper_0.78B",
        "desc": "16 layers, same width (0.78B)",
        "blocks": """[
        (3, ["e", "a", "m", "m"]),
        (1, ["e", "m", "m", "m"]),
    ]""",
        "config": {},
        "steps": 1600,
        "batch_size": 8,
    },
    # ── Scale up: ~1B (wider + deeper) ──
    {
        "name": "wide_deep_1B",
        "desc": "d_model=1792, d_ff=2560, 16 layers (~1B)",
        "blocks": """[
        (3, ["e", "a", "m", "m"]),
        (1, ["e", "m", "m", "m"]),
    ]""",
        "config": {"d_model": 1792, "d_ff": 2560},
        "steps": 1600,
        "batch_size": 4,
    },
    # ── Scale up: ~1.1B (d_model=2048) ──
    {
        "name": "big_1.1B",
        "desc": "d_model=2048, d_ff=2560, 16 layers (~1.1B)",
        "blocks": """[
        (3, ["e", "a", "m", "m"]),
        (1, ["e", "m", "m", "m"]),
    ]""",
        "config": {"d_model": 2048, "d_ff": 2560},
        "steps": 1600,
        "batch_size": 4,
    },
    # ── Scale up: ~1.5B (widest + deepest) ──
    {
        "name": "max_1.5B",
        "desc": "d_model=2048, d_ff=3072, 20 layers, d_latent=640 (~1.5B)",
        "blocks": """[
        (3, ["e", "a", "m", "m"]),
        (2, ["e", "m", "m", "m"]),
    ]""",
        "config": {"d_model": 2048, "d_ff": 3072, "d_latent": 640},
        "steps": 1600,
        "batch_size": 2,
    },
]


def save_original_config():
    """Save original config files before experiments."""
    for f in ["03_train_1b.py", "04_inference.py"]:
        src = ROOT / f
        bak = ROOT / f"{f}.bak"
        if not bak.exists():
            shutil.copy2(str(src), str(bak))


def restore_original_config():
    """Restore config files from backup."""
    for f in ["03_train_1b.py", "04_inference.py"]:
        bak = ROOT / f"{f}.bak"
        dst = ROOT / f
        if bak.exists():
            shutil.copy2(str(bak), str(dst))


def apply_experiment(exp):
    """Apply an architecture experiment's config changes."""
    restore_original_config()
    if exp["blocks"]:
        update_macro_blocks(exp["blocks"])
    if exp["config"]:
        update_model_config(exp["config"])


def main():
    log("=" * 60)
    log("PROGRESSIVE TRAINING: RAMP STEPS → EVOLVE ARCHITECTURE")
    log("=" * 60)

    history = []
    # Check if we have a best PPL from prior runs
    best_ppl_file = ROOT / "checkpoints/best/ppl.txt"
    if best_ppl_file.exists():
        raw = best_ppl_file.read_text().strip()
        m = re.search(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", raw)
        best_ppl = float(m.group(0)) if m else float("inf")
        log(f"  Resuming from best PPL: {best_ppl:.2f}")
    else:
        best_ppl = float("inf")
    stall_count = 0

    # ─── PHASE 1: Ramp up steps with LR cycling ───
    step_schedule = [200, 400, 800, 1600, 3200]
    lr_schedule =   [3e-4, 2.5e-4, 2e-4, 1.5e-4, 1e-4]
    round_num = 0

    log("\n>>> PHASE 1: Ramp up training steps")

    for steps, lr in zip(step_schedule, lr_schedule):
        round_num += 1
        tag = f"p1r{round_num:02d}"

        log(f"\n{'='*60}")
        log(f"PHASE 1 ROUND {round_num}: {steps} steps, LR={lr}")
        log(f"{'='*60}")

        ok = train_run(tag, steps, lr)
        if not ok:
            log("  Training failed, trying lower LR")
            ok = train_run(tag + "b", steps, lr / 2)
            if not ok:
                log("  Still failing, skipping")
                continue

        ppl = evaluate(tag)
        improvement = (best_ppl - ppl) / best_ppl * 100 if best_ppl < float("inf") else 100
        history.append({"phase": 1, "round": round_num, "steps": steps, "lr": lr, "ppl": ppl})

        log(f"  RESULT: PPL={ppl:.2f} (best={best_ppl:.2f}, change={improvement:+.1f}%)")

        if ppl < best_ppl:
            best_ppl = ppl
            save_best(ppl)
            stall_count = 0
        else:
            stall_count += 1

        if stall_count >= 2:
            log(f"  Phase 1 stalled after {round_num} rounds, moving to Phase 2")
            break

    # ─── PHASE 2: Architecture experiments ───
    log(f"\n{'='*60}")
    log(f">>> PHASE 2: Architecture experiments (best PPL so far: {best_ppl:.2f})")
    log(f"{'='*60}")

    save_original_config()
    best_experiment = None

    for exp in ARCH_EXPERIMENTS:
        round_num += 1
        tag = f"p2_{exp['name']}"

        log(f"\n{'='*60}")
        log(f"EXPERIMENT: {exp['name']} — {exp['desc']}")
        log(f"{'='*60}")

        # Apply architecture change
        apply_experiment(exp)

        # Train from scratch with this architecture (can't resume with different arch)
        exp_steps = exp.get("steps", 1600)
        exp_bs = exp.get("batch_size", 8)
        ok = train_run(tag, exp_steps, 3e-4, resume_from="NONE",
                        extra_args=f"--batch-size {exp_bs}")
        if not ok:
            log(f"  {exp['name']} failed, trying with smaller batch")
            ok = train_run(tag + "b", exp_steps, 3e-4, resume_from="NONE",
                           extra_args=f"--batch-size {max(1, exp_bs // 2)}")
            if not ok:
                log(f"  {exp['name']} still failing, skipping")
                restore_original_config()
                continue

        ppl = evaluate(tag)
        improvement = (best_ppl - ppl) / best_ppl * 100 if best_ppl < float("inf") else 100
        history.append({"phase": 2, "round": round_num, "name": exp["name"], "ppl": ppl})

        log(f"  RESULT: PPL={ppl:.2f} (best={best_ppl:.2f}, change={improvement:+.1f}%)")

        if ppl < best_ppl:
            best_ppl = ppl
            best_experiment = exp
            save_best(ppl)
            log(f"  >>> NEW BEST ARCHITECTURE: {exp['name']}")

        restore_original_config()

    # ─── PHASE 3: Apply best arch and do final extended training ───
    if best_experiment:
        log(f"\n{'='*60}")
        log(f">>> PHASE 3: Final training with best architecture: {best_experiment['name']}")
        log(f"{'='*60}")

        apply_experiment(best_experiment)

        # Restore best checkpoint
        best_ckpt = ROOT / "checkpoints/best/model.pt"
        if best_ckpt.exists():
            shutil.copy2(str(best_ckpt), str(ROOT / "checkpoints/final/model.pt"))

        # Extended training with decreasing LR
        for i, (steps, lr) in enumerate([(1600, 2e-4), (3200, 1e-4), (3200, 5e-5)]):
            tag = f"p3_final_{i+1}"
            log(f"\n  Final training {i+1}/3: {steps} steps, LR={lr}")

            ok = train_run(tag, steps, lr)
            if not ok:
                break

            ppl = evaluate(tag)
            history.append({"phase": 3, "round": i+1, "steps": steps, "ppl": ppl})
            log(f"  RESULT: PPL={ppl:.2f}")

            if ppl < best_ppl:
                best_ppl = ppl
                save_best(ppl)
    else:
        log("\n  No architecture improvement found, keeping original")

        # Do final extended training with original arch
        log(f"\n>>> PHASE 3: Final extended training with original architecture")

        # Restore best checkpoint
        best_ckpt = ROOT / "checkpoints/best/model.pt"
        if best_ckpt.exists():
            shutil.copy2(str(best_ckpt), str(ROOT / "checkpoints/final/model.pt"))

        for i, (steps, lr) in enumerate([(3200, 1.5e-4), (3200, 8e-5), (3200, 3e-5)]):
            tag = f"p3_ext_{i+1}"
            log(f"\n  Extended training {i+1}/3: {steps} steps, LR={lr}")

            ok = train_run(tag, steps, lr)
            if not ok:
                break

            ppl = evaluate(tag)
            history.append({"phase": 3, "round": i+1, "steps": steps, "ppl": ppl})
            log(f"  RESULT: PPL={ppl:.2f}")

            if ppl < best_ppl:
                best_ppl = ppl
                save_best(ppl)

    # ─── Summary ───
    log(f"\n{'='*60}")
    log(f"TRAINING COMPLETE")
    log(f"{'='*60}")
    log(f"  Best perplexity: {best_ppl:.2f}")
    log(f"  Total rounds: {len(history)}")
    log(f"  Total training steps: {sum(h.get('steps', 1600) for h in history)}")
    log(f"\nFull history:")
    for h in history:
        name = h.get("name", f"steps={h.get('steps', '?')}")
        log(f"  Phase {h['phase']}, Round {h['round']}: {name} → PPL={h['ppl']:.2f}")

    # Save history
    (LOGS / "plateau_history.json").write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
