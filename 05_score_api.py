#!/usr/bin/env python3
"""
Score an LLM served via OpenAI-compatible API on the same eval data as 04_inference.py.

Computes quality metrics by feeding eval chunks as prompts and measuring
generation confidence (logprobs). Results are comparable across models.

Usage:
    # Score the 120B model running on llama-server (port 8001)
    python 05_score_api.py

    # Score with custom settings
    python 05_score_api.py --api-base http://localhost:8001/v1 --eval-samples 50

    # Compare against a different server
    python 05_score_api.py --api-base http://localhost:8002/v1 --model my-model
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

import pyarrow.ipc as ipc
from openai import OpenAI
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent


def find_latest_arrow(search_dir: Path) -> Path:
    arrows = sorted(search_dir.glob("*.arrow"), key=lambda p: p.stat().st_mtime)
    if not arrows:
        raise FileNotFoundError(f"No .arrow files in {search_dir}. Run 02_process.py first.")
    return arrows[-1]


def load_eval_chunks(arrow_path: Path, tokenizer, n_samples: int, prompt_tokens: int):
    """Load chunks from arrow, decode to text, return as list of (prompt, continuation) pairs."""
    reader = ipc.open_file(str(arrow_path))
    table = reader.read_all()
    input_ids_col = table.column("input_ids")
    total_chunks = len(table)

    n_samples = min(n_samples, total_chunks)

    import torch
    indices = torch.randperm(total_chunks)[:n_samples].tolist()

    samples = []
    for idx in indices:
        ids = input_ids_col[idx].as_py()
        if not ids:
            continue
        text = tokenizer.decode(ids, skip_special_tokens=True)
        # Re-tokenize to get a clean split point (original uses GPT-2 tokenizer,
        # the API model uses its own tokenizer, so we split by character ratio)
        split_point = int(len(text) * (prompt_tokens / len(ids)))
        prompt_text = text[:split_point]
        continuation_text = text[split_point:]
        if len(prompt_text.strip()) > 50 and len(continuation_text.strip()) > 50:
            samples.append({
                "prompt": prompt_text,
                "continuation": continuation_text,
                "full_text": text,
                "original_tokens": len(ids),
            })

    return samples


def score_sample(client, model, prompt, max_gen_tokens, temperature=0.0):
    """Send prompt to API, generate tokens with logprobs, return metrics."""
    try:
        response = client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=max_gen_tokens,
            temperature=temperature,
            logprobs=1,
        )
    except Exception as e:
        return None, str(e)

    choice = response.choices[0]
    usage = response.usage

    logprobs_data = choice.logprobs
    if not logprobs_data:
        return None, "No logprobs returned"

    # Handle both object and dict formats (varies by API client version)
    content = logprobs_data.content if hasattr(logprobs_data, 'content') else logprobs_data.get("content")
    if not content:
        # llama.cpp may use .token_logprobs instead of .content
        if hasattr(logprobs_data, 'token_logprobs') and logprobs_data.token_logprobs:
            token_logprobs = [lp for lp in logprobs_data.token_logprobs if lp is not None]
            tokens = logprobs_data.tokens or []
        else:
            return None, "No logprobs content returned"
    else:
        token_logprobs = [t["logprob"] if isinstance(t, dict) else t.logprob for t in content]
        tokens = [t["token"] if isinstance(t, dict) else t.token for t in content]

    n_tokens = len(token_logprobs)
    if n_tokens == 0:
        return None, "No tokens generated"

    avg_logprob = sum(token_logprobs) / n_tokens
    # Perplexity = exp(-avg_logprob)
    gen_perplexity = math.exp(-avg_logprob) if -avg_logprob < 20 else float("inf")

    # Token-level confidence: fraction with logprob > -1 (>36% probability)
    high_conf = sum(1 for lp in token_logprobs if lp > -1.0) / n_tokens

    # Timings from llama.cpp (if available)
    timings = {}
    raw = response.model_dump() if hasattr(response, "model_dump") else {}
    if "timings" in raw:
        timings = raw["timings"]

    return {
        "n_tokens": n_tokens,
        "avg_logprob": avg_logprob,
        "gen_perplexity": gen_perplexity,
        "high_confidence_ratio": high_conf,
        "min_logprob": min(token_logprobs),
        "max_logprob": max(token_logprobs),
        "prompt_tokens": usage.prompt_tokens if usage else 0,
        "completion_tokens": usage.completion_tokens if usage else n_tokens,
        "generated_text": choice.text[:200],
        "timings": timings,
    }, None


def print_quality_report(results, model_name, arrow_name, elapsed):
    valid = [r for r in results if r is not None]
    if not valid:
        print("\nERROR: No valid results to report.")
        return

    n = len(valid)
    avg_ppl = sum(r["gen_perplexity"] for r in valid) / n
    avg_logprob = sum(r["avg_logprob"] for r in valid) / n
    avg_conf = sum(r["high_confidence_ratio"] for r in valid) / n
    total_gen_tokens = sum(r["completion_tokens"] for r in valid)
    total_prompt_tokens = sum(r["prompt_tokens"] for r in valid)
    gen_tps = total_gen_tokens / elapsed if elapsed > 0 else 0

    # Perplexity distribution
    ppls = sorted(r["gen_perplexity"] for r in valid)
    median_ppl = ppls[n // 2]
    p10_ppl = ppls[max(0, int(n * 0.1))]
    p90_ppl = ppls[min(n - 1, int(n * 0.9))]

    print("\n" + "=" * 65)
    print("  API MODEL QUALITY REPORT")
    print("=" * 65)
    print(f"  Model          : {model_name}")
    print(f"  Eval data      : {arrow_name}")
    print(f"  Samples scored : {n}")
    print(f"  Prompt tokens  : {total_prompt_tokens:,}")
    print(f"  Gen tokens     : {total_gen_tokens:,}")
    print(f"  Wall time      : {elapsed:.1f}s ({gen_tps:.1f} tok/s)")
    print()
    print(f"  Generation Perplexity")
    print(f"    Mean         : {avg_ppl:.2f}")
    print(f"    Median       : {median_ppl:.2f}")
    print(f"    P10 (best)   : {p10_ppl:.2f}")
    print(f"    P90 (worst)  : {p90_ppl:.2f}")
    print()
    print(f"  Avg logprob    : {avg_logprob:.4f}")
    print(f"  High-conf ratio: {avg_conf:.1%}  (tokens with >36% prob)")
    print()

    # Quality grade (generation perplexity is typically lower than true perplexity)
    if avg_ppl < 3:
        grade, desc = "A+", "Exceptional — model generates highly confident, coherent continuations"
    elif avg_ppl < 5:
        grade, desc = "A", "Excellent — strong language modeling on this data"
    elif avg_ppl < 10:
        grade, desc = "B", "Good — model handles the domain well"
    elif avg_ppl < 20:
        grade, desc = "C", "Fair — model understands the data but lacks precision"
    elif avg_ppl < 50:
        grade, desc = "D", "Poor — model struggles with this domain"
    else:
        grade, desc = "F", "Very poor — model is not suited for this data"

    print(f"  Quality grade  : {grade}")
    print(f"  {desc}")
    print()
    print("  NOTE: Generation perplexity measures model confidence on its own")
    print("  continuations of eval text. Lower = better. Not directly comparable")
    print("  to 04_inference.py's cross-entropy perplexity on held-out tokens,")
    print("  but useful for comparing different API-served models on the same data.")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(
        description="Score an API-served model on eval data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  python 05_score_api.py --list-models
  python 05_score_api.py --eval-samples 50
  python 05_score_api.py --api-base https://api.openai.com/v1 --api-key $OPENAI_API_KEY --model gpt-4o
""",
    )
    parser.add_argument("--api-base", type=str, default="http://localhost:8001/v1",
                        help="OpenAI-compatible API base URL")
    parser.add_argument("--model", type=str, default="nemotron-3-super",
                        help="Model name/alias on the server")
    parser.add_argument("--api-key", type=str, default="sk-no-key-required")
    parser.add_argument("--list-models", action="store_true",
                        help="List available models on the server and exit")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to tokenized .arrow (auto-detected if omitted)")
    parser.add_argument("--tokenizer", type=str, default="gpt2",
                        help="HuggingFace tokenizer for decoding arrow data")
    parser.add_argument("--eval-samples", type=int, default=50,
                        help="Number of chunks to evaluate")
    parser.add_argument("--prompt-tokens", type=int, default=768,
                        help="Approximate prompt length (in original tokenizer tokens)")
    parser.add_argument("--gen-tokens", type=int, default=128,
                        help="Tokens to generate per sample for scoring")
    args = parser.parse_args()

    # --list-models: show all supported models and exit
    if args.list_models:
        print("=" * 65)
        print("  Available Models for Scoring")
        print("=" * 65)

        # ── Local models (llama-server) ──
        print("\n  LOCAL (llama-server)")
        print("  " + "-" * 40)
        client = OpenAI(base_url="http://localhost:8001/v1", api_key="sk-no-key-required")
        try:
            models = client.models.list()
            for m in models.data:
                meta = ""
                if hasattr(m, "meta") and m.meta:
                    params = m.meta.get("n_params")
                    ctx = m.meta.get("n_ctx_train")
                    parts = []
                    if params:
                        parts.append(f"{params/1e9:.1f}B params")
                    if ctx:
                        parts.append(f"{ctx} ctx")
                    meta = f"  ({', '.join(parts)})" if parts else ""
                print(f"    {m.id}{meta}")
                print(f"      python 05_score_api.py --model {m.id}")
        except Exception:
            print("    (server not running on port 8001)")

        # ── OpenAI ──
        print("\n  OPENAI")
        print("  " + "-" * 40)
        openai_models = [
            ("gpt-4o",           "Latest GPT-4o"),
            ("gpt-4o-mini",      "GPT-4o Mini — fast & cheap"),
            ("gpt-4-turbo",      "GPT-4 Turbo 128K"),
            ("gpt-3.5-turbo",    "GPT-3.5 Turbo — baseline"),
            ("o4-mini",          "Reasoning model — compact"),
        ]
        for model_id, desc in openai_models:
            print(f"    {model_id:20s} {desc}")
        print(f"\n    python 05_score_api.py --api-base https://api.openai.com/v1 \\")
        print(f"      --api-key $OPENAI_API_KEY --model gpt-4o")

        # ── Anthropic (via OpenAI-compatible proxy) ──
        print("\n  ANTHROPIC (requires OpenAI-compatible proxy or LiteLLM)")
        print("  " + "-" * 40)
        anthropic_models = [
            ("claude-opus-4-6",    "Claude Opus 4.6 — most capable"),
            ("claude-sonnet-4-6",  "Claude Sonnet 4.6 — balanced"),
            ("claude-haiku-4-5",   "Claude Haiku 4.5 — fast"),
        ]
        for model_id, desc in anthropic_models:
            print(f"    {model_id:20s} {desc}")
        print(f"\n    # Via LiteLLM proxy:")
        print(f"    litellm --model claude-sonnet-4-6 --port 8002")
        print(f"    python 05_score_api.py --api-base http://localhost:8002/v1 \\")
        print(f"      --api-key $ANTHROPIC_API_KEY --model claude-sonnet-4-6")

        # ── Google ──
        print("\n  GOOGLE")
        print("  " + "-" * 40)
        google_models = [
            ("gemini-2.5-pro",   "Gemini 2.5 Pro — latest"),
            ("gemini-2.5-flash", "Gemini 2.5 Flash — fast"),
        ]
        for model_id, desc in google_models:
            print(f"    {model_id:20s} {desc}")
        print(f"\n    python 05_score_api.py --api-base https://generativelanguage.googleapis.com/v1beta/openai/ \\")
        print(f"      --api-key $GOOGLE_API_KEY --model gemini-2.5-pro")

        # ── Other local ──
        print("\n  OTHER LOCAL (any OpenAI-compatible server)")
        print("  " + "-" * 40)
        print(f"    python 05_score_api.py --api-base http://localhost:PORT/v1 --model MODEL_NAME")

        print()
        sys.exit(0)

    # Resolve data
    if args.data:
        arrow_path = Path(args.data)
    else:
        arrow_path = find_latest_arrow(ROOT / "data" / "tokenized")

    print("=" * 65)
    print("  API Model Scoring")
    print("=" * 65)
    print(f"  API base    : {args.api_base}")
    print(f"  Model       : {args.model}")
    print(f"  Eval data   : {arrow_path.name}")
    print(f"  Samples     : {args.eval_samples}")
    print(f"  Prompt size : ~{args.prompt_tokens} tokens")
    print(f"  Gen tokens  : {args.gen_tokens}")
    print()

    # Connect to API
    client = OpenAI(base_url=args.api_base, api_key=args.api_key)

    # Verify server is reachable
    try:
        models = client.models.list()
        print(f"  Server OK — models: {[m.id for m in models.data]}")
    except Exception as e:
        print(f"  ERROR: Cannot connect to {args.api_base}: {e}")
        sys.exit(1)

    # Load tokenizer and eval data
    print(f"\nLoading tokenizer '{args.tokenizer}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    print(f"Loading eval chunks from {arrow_path.name}...")
    samples = load_eval_chunks(arrow_path, tokenizer, args.eval_samples, args.prompt_tokens)
    print(f"  Loaded {len(samples)} valid samples\n")

    if not samples:
        print("ERROR: No eval samples after filtering (try more --eval-samples or lower length thresholds).")
        sys.exit(1)

    # Score each sample
    results = []
    errors = 0
    t_start = time.time()

    for i, sample in enumerate(samples):
        result, err = score_sample(
            client, args.model, sample["prompt"],
            max_gen_tokens=args.gen_tokens,
        )

        if err:
            errors += 1
            if errors <= 3:
                print(f"  [{i+1}/{len(samples)}] ERROR: {err}")
            continue

        results.append(result)

        # Progress update
        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t_start
            avg_ppl = sum(r["gen_perplexity"] for r in results) / len(results)
            print(
                f"  [{i+1}/{len(samples)}] "
                f"running ppl: {avg_ppl:.2f}  "
                f"last ppl: {result['gen_perplexity']:.2f}  "
                f"elapsed: {elapsed:.0f}s"
            )

    elapsed = time.time() - t_start

    if errors:
        print(f"\n  {errors} samples failed")

    print_quality_report(results, args.model, arrow_path.name, elapsed)

    # Save raw results
    out_path = ROOT / "score_api_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "api_base": args.api_base,
            "arrow": str(arrow_path),
            "n_samples": len(results),
            "n_errors": errors,
            "elapsed_s": elapsed,
            "results": results,
        }, f, indent=2, default=str)
    print(f"\nRaw results saved to {out_path}")


if __name__ == "__main__":
    main()
