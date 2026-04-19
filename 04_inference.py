#!/usr/bin/env python3
"""
Step 4: Evaluate model quality and launch interactive chat.

Loads a trained checkpoint, scores it on held-out training chunks (perplexity),
then opens an interactive text-completion prompt.

Usage:
    python 04_inference.py                                      # auto-detect checkpoint + data
    python 04_inference.py --checkpoint checkpoints/final/model.pt
    python 04_inference.py --eval-samples 200                   # more eval samples
    python 04_inference.py --max-tokens 128 --temperature 0.9   # generation settings
"""

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from nemotron3 import Nemotron3Super, NemotronSuperConfig, count_params, summarize_layers
from dataclasses import dataclass, field


# ─────────────────────────────────────────────────────────────────────────────
#  Model config (must match training)
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
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_latest_checkpoint(search_dir: Path) -> Path:
    """Find the most recent model.pt under checkpoints/."""
    candidates = sorted(search_dir.rglob("model.pt"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(
            f"No model.pt found under {search_dir}. Run 03_train_1b.py first."
        )
    return candidates[-1]


def find_latest_arrow(search_dir: Path) -> Path:
    arrows = sorted(search_dir.glob("*.arrow"), key=lambda p: p.stat().st_mtime)
    if not arrows:
        raise FileNotFoundError(
            f"No .arrow files in {search_dir}. Run 02_process.py first."
        )
    return arrows[-1]


def load_model(checkpoint_path: Path, device: torch.device, max_seq_len: int) -> Nemotron3Super:
    cfg = Nemotron1BConfig(max_seq_len=max_seq_len)
    model = Nemotron3Super(cfg)
    state_dict = torch.load(str(checkpoint_path), map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
#  Quality evaluation (perplexity on training data samples)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_quality(model, arrow_path: Path, device: torch.device,
                     n_samples: int = 100, seq_len: int = 1024) -> dict:
    """Compute perplexity on a random subset of training chunks."""
    import pyarrow.ipc as ipc

    reader = ipc.open_file(str(arrow_path))
    table = reader.read_all()
    input_ids_col = table.column("input_ids")
    labels_col = table.column("labels")
    total_chunks = len(table)

    n_samples = min(n_samples, total_chunks)

    # Sample random indices
    indices = torch.randperm(total_chunks)[:n_samples].tolist()

    total_loss = 0.0
    total_tokens = 0

    print(f"\nEvaluating on {n_samples} random chunks from {arrow_path.name}...")

    with torch.no_grad():
        for i, idx in enumerate(indices):
            ids = torch.tensor(input_ids_col[idx].as_py(), dtype=torch.long).unsqueeze(0).to(device)
            labs = torch.tensor(labels_col[idx].as_py(), dtype=torch.long).unsqueeze(0).to(device)

            logits, _ = model(ids)
            loss = F.cross_entropy(
                logits.reshape(-1, model.cfg.vocab_size),
                labs.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            n_valid = (labs != -100).sum().item()
            total_loss += loss.item()
            total_tokens += n_valid

            if (i + 1) % 25 == 0:
                running_ppl = math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")
                print(f"  [{i+1}/{n_samples}] running perplexity: {running_ppl:.2f}")

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float("inf")

    return {
        "perplexity": perplexity,
        "avg_loss": avg_loss,
        "eval_samples": n_samples,
        "eval_tokens": total_tokens,
    }


def print_quality_report(stats: dict):
    ppl = stats["perplexity"]

    print("\n" + "=" * 60)
    print("  MODEL QUALITY REPORT")
    print("=" * 60)
    print(f"  Perplexity       : {ppl:.2f}")
    print(f"  Avg cross-entropy: {stats['avg_loss']:.4f}")
    print(f"  Eval samples     : {stats['eval_samples']}")
    print(f"  Eval tokens      : {stats['eval_tokens']:,}")
    print()

    # Quality rating based on perplexity ranges for a pretrained LM
    if ppl < 10:
        grade, desc = "A", "Excellent — model has learned the training distribution well"
    elif ppl < 30:
        grade, desc = "B", "Good — model captures most patterns in the data"
    elif ppl < 80:
        grade, desc = "C", "Fair — model has learned basic structure but struggles with detail"
    elif ppl < 200:
        grade, desc = "D", "Poor — model is undertrained, needs more steps or data"
    else:
        grade, desc = "F", "Very poor — model barely learned from training data"

    print(f"  Quality grade: {grade}")
    print(f"  {desc}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
#  Text generation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_tokens: int = 100,
             temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9,
             device: torch.device = torch.device("cpu")) -> str:
    """Autoregressive text generation with top-k/top-p sampling."""
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    max_seq = model.cfg.max_seq_len

    generated = []
    for _ in range(max_tokens):
        # Truncate to max sequence length if needed
        if ids.shape[1] > max_seq:
            ids = ids[:, -max_seq:]

        logits, _ = model(ids)
        next_logits = logits[:, -1, :] / temperature

        # Top-k filtering
        if top_k > 0:
            topk_vals, _ = next_logits.topk(top_k, dim=-1)
            threshold = topk_vals[:, -1].unsqueeze(-1)
            next_logits[next_logits < threshold] = float("-inf")

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_idx = next_logits.sort(descending=True, dim=-1)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove_mask = cum_probs > top_p
            remove_mask[:, 1:] = remove_mask[:, :-1].clone()
            remove_mask[:, 0] = False
            sorted_logits[remove_mask] = float("-inf")
            next_logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

        probs = F.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

        if next_id.item() == tokenizer.eos_token_id:
            break

        generated.append(next_id.item())
        ids = torch.cat([ids, next_id], dim=1)

    return tokenizer.decode(generated)


# ─────────────────────────────────────────────────────────────────────────────
#  Interactive chat loop
# ─────────────────────────────────────────────────────────────────────────────

def chat_loop(model, tokenizer, device, max_tokens: int, temperature: float):
    print("\n" + "=" * 60)
    print("  INTERACTIVE CHAT")
    print("=" * 60)
    print("  Type a prompt and the model will continue the text.")
    print("  Commands:")
    print("    /quit or /exit  — exit the chat")
    print("    /temp <value>   — change temperature")
    print("    /tokens <value> — change max generated tokens")
    print("=" * 60 + "\n")

    while True:
        try:
            prompt = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not prompt:
            continue

        if prompt.lower() in ("/quit", "/exit"):
            print("Goodbye!")
            break

        if prompt.lower().startswith("/temp "):
            try:
                temperature = float(prompt.split()[1])
                print(f"  Temperature set to {temperature}")
            except (IndexError, ValueError):
                print("  Usage: /temp 0.8")
            continue

        if prompt.lower().startswith("/tokens "):
            try:
                max_tokens = int(prompt.split()[1])
                print(f"  Max tokens set to {max_tokens}")
            except (IndexError, ValueError):
                print("  Usage: /tokens 128")
            continue

        output = generate(
            model, tokenizer, prompt,
            max_tokens=max_tokens, temperature=temperature, device=device,
        )
        print(f"\nModel> {output}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate and chat with trained Nemotron")
    p.add_argument("--checkpoint", type=str, default=None,
                    help="Path to model.pt (auto-detected if omitted)")
    p.add_argument("--data", type=str, default=None,
                    help="Path to tokenized .arrow for evaluation (auto-detected if omitted)")
    p.add_argument("--tokenizer", type=str, default="gpt2",
                    help="HuggingFace tokenizer name (must match training)")
    p.add_argument("--seq-len", type=int, default=1024,
                    help="Model max_seq_len / training sequence length (must match checkpoint)")
    p.add_argument("--eval-samples", type=int, default=100,
                    help="Number of chunks to evaluate on")
    p.add_argument("--skip-eval", action="store_true",
                    help="Skip quality evaluation, go straight to chat")
    p.add_argument("--max-tokens", type=int, default=100,
                    help="Max tokens to generate per response")
    p.add_argument("--temperature", type=float, default=0.8,
                    help="Sampling temperature")
    p.add_argument("--device", type=str, default=None,
                    help="Device (auto-detected if omitted)")
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Resolve checkpoint
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.is_absolute():
            ckpt_path = ROOT / ckpt_path
    else:
        ckpt_path = find_latest_checkpoint(ROOT / "checkpoints")

    print("=" * 60)
    print("Step 4: Inference")
    print("=" * 60)
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  Device     : {device}")
    print(f"  Seq len    : {args.seq_len} (must match training)")
    print(f"  Tokenizer  : {args.tokenizer}")

    # Load model
    print("\nLoading model...")
    model = load_model(ckpt_path, device, max_seq_len=args.seq_len)
    params = count_params(model)
    print(f"  Parameters: {params['total'] / 1e9:.2f}B")
    print(f"  Layers: {summarize_layers(model.cfg)}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Quality evaluation
    if not args.skip_eval:
        if args.data:
            arrow_path = Path(args.data)
            if not arrow_path.is_absolute():
                arrow_path = ROOT / arrow_path
        else:
            arrow_path = find_latest_arrow(ROOT / "data" / "tokenized")

        print(f"  Eval data  : {arrow_path}")

        stats = evaluate_quality(model, arrow_path, device, n_samples=args.eval_samples)
        print_quality_report(stats)

    # Interactive chat
    chat_loop(model, tokenizer, device, args.max_tokens, args.temperature)


if __name__ == "__main__":
    main()
