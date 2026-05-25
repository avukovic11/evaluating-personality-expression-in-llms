"""Async OpenAI generation pipeline for Style D and Style A essays.

For each planned essay (style × trait × level × i for D, or style × paired
profile for A), this script:

  1. Skips it if already present in the output JSONL (idempotent resume).
  2. Calls the chat-completion API asynchronously with bounded concurrency.
  3. Retries RateLimitError / APITimeoutError / APIConnectionError with
     exponential backoff (other errors propagate).
  4. Atomically appends the record to the output JSONL on success; logs
     fatal failures to a separate `<style>_errors.jsonl`.
  5. Aborts gracefully if estimated total cost exceeds `--max-cost`.

The API key is read from `.env` via python-dotenv, or from the
`OPENAI_API_KEY` environment variable if set.

Run from `code/`:

    # ---- DRY RUNS (cheap; inspect outputs before launching full run) ----
    # Style D: 1 essay per (trait × level) = 15 essays
    python -m src.generate --style D --n-per-condition 1

    # Style A: 5 paired essays
    python -m src.generate --style A --n-paired 5

    # ---- FULL RUNS ----
    python -m src.generate --style D                 # 1500 essays
    python -m src.generate --style A                 # 500 essays

    # ---- USEFUL FLAGS ----
    --model gpt-4o-mini       # default
    --concurrency 8           # max in-flight requests
    --max-cost 5.0            # abort if estimated USD spend exceeds this
    --temperature 0.9         # sampling temperature
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from . import config
from .data import load_essays, load_splits
from .prompts import (
    LEVELS,
    build_messages,
    style_a_user_prompt,
    style_d_user_prompt,
)

# OpenAI USD pricing per 1M tokens (rough current snapshot — update if the
# `--model` flag points elsewhere).
PRICING: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o":      {"input": 2.50, "output": 10.00},
    "gpt-5-mini":  {"input": 0.25, "output": 2.00},
}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    price = PRICING.get(model, PRICING["gpt-4o-mini"])
    return (prompt_tokens * price["input"] + completion_tokens * price["output"]) / 1e6


# ---------------------------------------------------------------------------
# Plan construction
# ---------------------------------------------------------------------------

def style_d_plan(n_per_condition: int) -> list[dict[str, Any]]:
    """Build the list of (essay_id, prompt) for the Style D run."""
    plans: list[dict[str, Any]] = []
    for trait in config.TRAIT_COLS:
        for level in LEVELS:
            for i in range(n_per_condition):
                essay_id = f"D_{trait}_{level}_{i:03d}"
                plans.append({
                    "essay_id": essay_id,
                    "prompt_style": "D",
                    "prompted_trait": trait,
                    "prompted_level": level,
                    "intended_profile": None,
                    "user_prompt": style_d_user_prompt(trait, level),
                })
    return plans


def style_a_plan(n_paired: int, seed: int) -> list[dict[str, Any]]:
    """Sample `n_paired` human profiles from the train split."""
    df = load_essays()
    splits = load_splits(df)
    sub = splits["train"].sample(n=n_paired, random_state=seed)
    plans: list[dict[str, Any]] = []
    for _, row in sub.iterrows():
        profile = {t: int(row[t]) for t in config.TRAIT_COLS}
        plans.append({
            "essay_id": str(row["AUTHID"]),
            "prompt_style": "A",
            "prompted_trait": None,
            "prompted_level": None,
            "intended_profile": profile,
            "user_prompt": style_a_user_prompt(profile),
        })
    return plans


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------

def load_done_ids(path: Path) -> set[str]:
    """Read existing `essay_id`s from output JSONL for idempotent resume."""
    if not path.exists():
        return set()
    done: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["essay_id"])
            except Exception:
                pass
    return done


# ---------------------------------------------------------------------------
# Async API call
# ---------------------------------------------------------------------------

async def call_with_retry(
    client, messages: list[dict[str, str]], model: str,
    temperature: float, max_tokens: int, max_retries: int = 5,
):
    """Chat-completion call with exponential backoff on transient errors.

    `insufficient_quota` is not retryable (project has no credits); we
    let it bubble immediately so the run aborts fast instead of waiting
    through 5 attempts × 15 essays before noticing.
    """
    from openai import APIConnectionError, APITimeoutError, RateLimitError

    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            return await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except (RateLimitError, APITimeoutError, APIConnectionError) as e:
            if isinstance(e, RateLimitError):
                body = getattr(e, "body", None) or {}
                if isinstance(body, dict) and body.get("code") == "insufficient_quota":
                    raise  # account/project has no credits — no point retrying
            last_err = e
            if attempt == max_retries - 1:
                break
            wait = 2 ** attempt + random.random()
            await asyncio.sleep(wait)
    raise RuntimeError(f"max retries exceeded ({max_retries}): {last_err}")


async def run_one(
    client, plan: dict[str, Any], model: str, temperature: float,
    max_tokens: int, sem: asyncio.Semaphore, out_path: Path,
    errors_path: Path, file_lock: asyncio.Lock, cost_state: dict[str, Any],
) -> dict[str, Any] | None:
    """Generate one essay, append record + update running cost. Returns record or None."""
    async with sem:
        if cost_state["aborted"]:
            return None
        messages = build_messages(plan["user_prompt"])
        start = time.time()
        try:
            response = await call_with_retry(
                client, messages, model, temperature, max_tokens,
            )
        except Exception as e:
            err = {
                "essay_id": plan["essay_id"],
                "error": str(e),
                "type": type(e).__name__,
            }
            async with file_lock:
                with open(errors_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(err, ensure_ascii=False) + "\n")
            return None
        latency = time.time() - start
        usage = response.usage
        cost = estimate_cost(model, usage.prompt_tokens, usage.completion_tokens)
        cost_state["total"] += cost
        cost_state["n"] += 1
        record = {
            "essay_id":          plan["essay_id"],
            "prompt_style":      plan["prompt_style"],
            "prompted_trait":    plan["prompted_trait"],
            "prompted_level":    plan["prompted_level"],
            "intended_profile":  plan["intended_profile"],
            "model":             model,
            "finish_reason":     response.choices[0].finish_reason,
            "prompt_tokens":     usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "latency_s":         round(latency, 2),
            "generated_text":    response.choices[0].message.content,
        }
        async with file_lock:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        if cost_state["total"] > cost_state["max_cost"] and not cost_state["aborted"]:
            cost_state["aborted"] = True
            print(
                f"\n[cost guard] estimated total ${cost_state['total']:.4f} "
                f"exceeded --max-cost ${cost_state['max_cost']:.2f}; "
                f"stopping (already-launched requests will finish).",
                file=sys.stderr,
            )
        return record


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

async def run_generation(
    plans: list[dict[str, Any]], model: str, concurrency: int,
    temperature: float, max_tokens: int, out_path: Path, errors_path: Path,
    max_cost: float,
) -> None:
    from openai import AsyncOpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        sys.exit(
            "OPENAI_API_KEY not set. Either:\n"
            "  1. Copy .env.example to .env in the repo root and put your key there, or\n"
            "  2. export OPENAI_API_KEY=sk-... in your shell, or\n"
            "  3. (Colab) os.environ['OPENAI_API_KEY'] = getpass(...) in a notebook cell."
        )
    client = AsyncOpenAI(api_key=api_key)

    sem = asyncio.Semaphore(concurrency)
    file_lock = asyncio.Lock()
    cost_state: dict[str, Any] = {
        "total": 0.0, "n": 0, "max_cost": max_cost, "aborted": False,
    }

    tasks = [
        asyncio.create_task(run_one(
            client, plan, model, temperature, max_tokens,
            sem, out_path, errors_path, file_lock, cost_state,
        ))
        for plan in plans
    ]

    done_count = 0
    total = len(tasks)
    for fut in asyncio.as_completed(tasks):
        await fut
        done_count += 1
        if done_count % 10 == 0 or done_count == total or done_count <= 3:
            avg = cost_state["total"] / cost_state["n"] if cost_state["n"] else 0.0
            print(
                f"  [{done_count}/{total}]  "
                f"running ${cost_state['total']:.4f}  "
                f"avg ${avg:.5f}/essay",
                flush=True,
            )

    await client.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--style", choices=["D", "A"], required=True,
        help="D = single-trait isolated; A = full multi-trait paired.",
    )
    parser.add_argument(
        "--model", type=str, default=config.GEN_MODEL,
        help=f"OpenAI model id; default {config.GEN_MODEL}.",
    )
    parser.add_argument(
        "--n-per-condition", type=int, default=100,
        help="Style D: essays per (trait × level). Default 100 (full run = 1500 total).",
    )
    parser.add_argument(
        "--n-paired", type=int, default=500,
        help="Style A: number of paired human profiles to sample. Default 500.",
    )
    parser.add_argument(
        "--temperature", type=float, default=config.GEN_TEMPERATURE,
        help=f"Sampling temperature. Default {config.GEN_TEMPERATURE}.",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=1100,
        help="Max output tokens per call. ~1100 covers a 600–800 word essay.",
    )
    parser.add_argument(
        "--concurrency", type=int, default=8,
        help="Max concurrent in-flight API requests.",
    )
    parser.add_argument(
        "--max-cost", type=float, default=5.0,
        help="Abort if estimated total cost (USD) exceeds this. Default $5.",
    )
    parser.add_argument(
        "--seed", type=int, default=config.SEED,
        help="Seed for Style A profile sampling. Ignored for Style D.",
    )
    args = parser.parse_args()

    load_dotenv()  # OPENAI_API_KEY from .env if present
    config.ensure_dirs()

    if args.style == "D":
        plans = style_d_plan(args.n_per_condition)
        out_path = config.LLM_OUTPUTS_DIR / "style_d_single_trait.jsonl"
        label = (
            f"Style D — {len(plans)} essays "
            f"({args.n_per_condition} per (trait × level), "
            f"5 traits × 3 levels)"
        )
    else:
        plans = style_a_plan(args.n_paired, args.seed)
        out_path = config.LLM_OUTPUTS_DIR / "style_a_paired.jsonl"
        label = f"Style A — {len(plans)} paired essays"
    errors_path = out_path.with_name(out_path.stem + "_errors.jsonl")

    done = load_done_ids(out_path)
    if done:
        before = len(plans)
        plans = [p for p in plans if p["essay_id"] not in done]
        print(f"Resuming: {before - len(plans)} essays already done, "
              f"{len(plans)} remaining.")

    if not plans:
        print(f"Nothing to do; {len(done)} essays already in {out_path}.")
        return

    print(label)
    print(f"model       : {args.model}")
    print(f"temperature : {args.temperature}")
    print(f"concurrency : {args.concurrency}")
    print(f"max-cost    : ${args.max_cost:.2f}")
    print(f"output      : {out_path}")
    print(f"errors      : {errors_path}")
    print()

    asyncio.run(run_generation(
        plans, model=args.model, concurrency=args.concurrency,
        temperature=args.temperature, max_tokens=args.max_tokens,
        out_path=out_path, errors_path=errors_path, max_cost=args.max_cost,
    ))

    final_done = load_done_ids(out_path)
    n_errs = (
        sum(1 for _ in open(errors_path, encoding="utf-8"))
        if errors_path.exists() else 0
    )
    print(f"\nDone: {len(final_done)} records in {out_path}")
    if n_errs:
        print(f"WARNING: {n_errs} errors logged in {errors_path}")


if __name__ == "__main__":
    main()
