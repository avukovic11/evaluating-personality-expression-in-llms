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

Both datasets share the same pipeline. The `--dataset` flag selects:
  - `essays`     : 650–700 word free-writing essays (Pennebaker register).
  - `recruitview`: 70–100 word interview answers tied to a sampled question
                   from the RECRUITVIEW question pool.

Run from `code/`:

    # ---- TRACK 1 (essays) ----
    # Dry run — 1 per (trait × HIGH|LOW) + 1 NEUTRAL = 11 essays
    python -m src.generate --style D --n-per-trait-level 1 --n-neutral 1
    # Full run — default 125 per (trait × HIGH|LOW) + 250 NEUTRAL = 1500 total
    python -m src.generate --style D
    python -m src.generate --style A                 # 500 paired essays

    # ---- TRACK 2 (recruitview) ----
    python -m src.generate --dataset recruitview --style D \
        --n-per-trait-level 1 --n-neutral 1                     # dry run, 11 answers
    python -m src.generate --dataset recruitview --style A \
        --n-synthetic-users 3                                   # dry run, ~18 answers
    python -m src.generate --dataset recruitview --style D      # 1500 short answers
    python -m src.generate --dataset recruitview --style A      # ~600 answers, 100 users
    # Full-coverage Style A — sample all 230 train users:
    python -m src.generate --dataset recruitview --style A --n-synthetic-users 230

    # ---- MULTI-RUN (append to JSONL, more samples per condition) ----
    python -m src.generate --dataset recruitview --style D --run-tag run2 --seed 43
    python -m src.generate --dataset recruitview --style D --run-tag run3 --seed 44

    # ---- USEFUL FLAGS ----
    --model gpt-4o-mini       # default
    --concurrency 8           # max in-flight requests
    --max-cost 5.0            # abort if estimated USD spend exceeds this
    --temperature 0.9         # sampling temperature
    --run-tag <str>           # suffix essay_ids so a re-run appends instead of skipping
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

import numpy as np
from dotenv import load_dotenv

from . import config
from .data import load_essays, load_splits
from .prompts import (
    LEVELS,
    PROMPT_VARIANTS,
    PromptVariant,
    build_messages,
    discretize_z,
    style_a_recruitview_prompt,
    style_a_user_prompt,
    style_d_recruitview_prompt,
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

def _tag_suffix(run_tag: str) -> str:
    return f"_{run_tag}" if run_tag else ""


def style_d_plan(
    n_per_trait_level: int, n_neutral: int,
    run_tag: str = "", variant: PromptVariant = "full",
) -> list[dict[str, Any]]:
    """Build the essays Style D plan.

    - `n_per_trait_level` essays per (trait × HIGH|LOW): 5 × 2 × n total.
    - `n_neutral` essays in a single shared NEUTRAL pool (no per-trait
      duplication; the prompt is identical across NEUTRAL essays so we
      only sample it once globally).
    - `variant` controls descriptor inclusion (see prompts.PromptVariant).
    """
    plans: list[dict[str, Any]] = []
    suffix = _tag_suffix(run_tag)
    for trait in config.TRAIT_COLS:
        for level in ("HIGH", "LOW"):
            for i in range(n_per_trait_level):
                plans.append({
                    "essay_id": f"D_{trait}_{level}_{i:03d}{suffix}",
                    "dataset": "essays",
                    "prompt_style": "D",
                    "prompt_variant": variant,
                    "prompted_trait": trait,
                    "prompted_level": level,
                    "intended_profile": None,
                    "user_prompt": style_d_user_prompt(trait, level, variant),
                })
    if n_neutral > 0:
        # NEUTRAL has no trait language to strip, so the variant is moot —
        # but we still record it so a label-only NEUTRAL pool can be pooled
        # with a label-only HIGH/LOW pool by downstream tooling if desired.
        neutral_prompt = style_d_user_prompt(
            config.TRAIT_COLS[0], "NEUTRAL", variant,
        )
        for i in range(n_neutral):
            plans.append({
                "essay_id": f"D_NEUTRAL_{i:03d}{suffix}",
                "dataset": "essays",
                "prompt_style": "D",
                "prompt_variant": variant,
                "prompted_trait": None,
                "prompted_level": "NEUTRAL",
                "intended_profile": None,
                "user_prompt": neutral_prompt,
            })
    return plans


def style_a_plan(
    n_paired: int, seed: int, run_tag: str = "",
) -> list[dict[str, Any]]:
    """Sample `n_paired` human profiles from the essays train split."""
    df = load_essays()
    splits = load_splits(df)
    sub = splits["train"].sample(n=n_paired, random_state=seed)
    plans: list[dict[str, Any]] = []
    suffix = _tag_suffix(run_tag)
    for _, row in sub.iterrows():
        profile = {t: int(row[t]) for t in config.TRAIT_COLS}
        plans.append({
            "essay_id": f"{row['AUTHID']}{suffix}",
            "dataset": "essays",
            "prompt_style": "A",
            "prompted_trait": None,
            "prompted_level": None,
            "intended_profile": profile,
            "user_prompt": style_a_user_prompt(profile),
        })
    return plans


# ---------------------------------------------------------------------------
# RECRUITVIEW plan builders
# ---------------------------------------------------------------------------

def _load_recruitview_train_df():
    """Load RECRUITVIEW and return the train-user-only DataFrame."""
    from .data_recruitview import load_recruitview, load_recruitview_splits
    df = load_recruitview()
    return load_recruitview_splits(df)["train"]


def style_d_recruitview_plan(
    n_per_trait_level: int, n_neutral: int, seed: int,
    run_tag: str = "", variant: PromptVariant = "full",
) -> list[dict[str, Any]]:
    """Build the RecruitView Style D plan.

    - `n_per_trait_level` answers per (trait × HIGH|LOW), each with a
      fresh question sampled from the train-split question pool.
    - `n_neutral` answers in a single shared NEUTRAL pool (no per-trait
      duplication); each NEUTRAL essay still pairs with a sampled question
      so the prompt has content to answer, but no trait language.
    - `variant` controls descriptor inclusion (see prompts.PromptVariant).
    """
    train_df = _load_recruitview_train_df()
    question_pool = (
        train_df[["question_id", "question"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    rng = np.random.default_rng(seed)
    plans: list[dict[str, Any]] = []
    suffix = _tag_suffix(run_tag)

    for trait in config.RECRUITVIEW_TRAIT_COLS:
        for level in ("HIGH", "LOW"):
            for i in range(n_per_trait_level):
                q_idx = int(rng.integers(0, len(question_pool)))
                qrow = question_pool.iloc[q_idx]
                question = str(qrow["question"])
                plans.append({
                    "essay_id":             f"D_rv_{trait}_{level}_{i:03d}{suffix}",
                    "dataset":              "recruitview",
                    "prompt_style":         "D",
                    "prompt_variant":       variant,
                    "prompted_trait":       trait,
                    "prompted_level":       level,
                    "prompted_question_id": int(qrow["question_id"]),
                    "prompted_question":    question,
                    "user_prompt":          style_d_recruitview_prompt(
                        trait, level, question, variant,
                    ),
                })

    for i in range(n_neutral):
        q_idx = int(rng.integers(0, len(question_pool)))
        qrow = question_pool.iloc[q_idx]
        question = str(qrow["question"])
        plans.append({
            "essay_id":             f"D_rv_NEUTRAL_{i:03d}{suffix}",
            "dataset":              "recruitview",
            "prompt_style":         "D",
            "prompt_variant":       variant,
            "prompted_trait":       None,
            "prompted_level":       "NEUTRAL",
            "prompted_question_id": int(qrow["question_id"]),
            "prompted_question":    question,
            "user_prompt":          style_d_recruitview_prompt(
                config.RECRUITVIEW_TRAIT_COLS[0], "NEUTRAL", question, variant,
            ),
        })
    return plans


def style_a_recruitview_plan(
    n_synthetic_users: int, seed: int,
    run_tag: str = "", variant: PromptVariant = "full",
) -> list[dict[str, Any]]:
    """Sample `n_synthetic_users` users from train; for each, generate one
    answer per their own (question_id, question) row.

    The per-user output count varies (mean ~6 in RECRUITVIEW); 100 users
    yield ~600 essays. Predictions are aggregated per `paired_user_no` at
    eval time — that matches the dataset's user-level annotation unit and
    is the same lens we use to report user-aggregated probe Spearman.
    """
    train_df = _load_recruitview_train_df()
    unique_users = sorted(train_df["user_no"].unique())
    if n_synthetic_users > len(unique_users):
        raise ValueError(
            f"--n-synthetic-users={n_synthetic_users} exceeds the "
            f"{len(unique_users)} unique train users available."
        )
    rng = np.random.default_rng(seed)
    chosen = rng.choice(
        np.asarray(unique_users, dtype=object),
        size=n_synthetic_users, replace=False,
    )
    plans: list[dict[str, Any]] = []
    suffix = _tag_suffix(run_tag)
    for user_no in chosen:
        user_no = str(user_no)
        user_rows = train_df[train_df["user_no"] == user_no]
        # z-scores are user-level (constant across a user's clips), so
        # take from the first row and reuse for every answer this user
        # generates.
        first_row = user_rows.iloc[0]
        intended_z = {
            t: float(first_row[t]) for t in config.RECRUITVIEW_TRAIT_COLS
        }
        intended_levels = {
            t: discretize_z(intended_z[t])
            for t in config.RECRUITVIEW_TRAIT_COLS
        }
        for _, row in user_rows.iterrows():
            question = str(row["question"])
            qid = int(row["question_id"])
            plans.append({
                "essay_id":           f"A_rv_{user_no}_q{qid:03d}{suffix}",
                "dataset":            "recruitview",
                "prompt_style":       "A",
                "prompt_variant":     variant,
                "paired_user_no":     user_no,
                "paired_question_id": qid,
                "paired_question":    question,
                "intended_z":         intended_z,
                "intended_levels":    intended_levels,
                "user_prompt":        style_a_recruitview_prompt(
                    intended_levels, question, variant,
                ),
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
        messages = build_messages(
            plan["user_prompt"],
            dataset=plan.get("dataset", "essays"),
        )
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
        # Forward every plan field except the prompt itself, then append
        # the API-side fields. Lets each (dataset × style) plan builder
        # declare whatever record schema it needs without touching run_one.
        record = {k: v for k, v in plan.items() if k != "user_prompt"}
        record.update({
            "model":             model,
            "finish_reason":     response.choices[0].finish_reason,
            "prompt_tokens":     usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "latency_s":         round(latency, 2),
            "generated_text":    response.choices[0].message.content,
        })
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
        "--dataset", choices=["essays", "recruitview"], default="essays",
        help="Which dataset's prompt register + output path to use.",
    )
    parser.add_argument(
        "--style", choices=["D", "A"], required=True,
        help="D = single-trait isolated; A = full multi-trait paired.",
    )
    parser.add_argument(
        "--model", type=str, default=config.GEN_MODEL,
        help=f"OpenAI model id; default {config.GEN_MODEL}.",
    )
    parser.add_argument(
        "--n-per-trait-level", type=int, default=125,
        help=(
            "Style D: essays per (trait × HIGH|LOW). "
            "Default 125 → 5 traits × 2 levels × 125 = 1250 HIGH+LOW total."
        ),
    )
    parser.add_argument(
        "--n-neutral", type=int, default=250,
        help=(
            "Style D: essays in the shared NEUTRAL pool (not per-trait, since "
            "the NEUTRAL prompt is identical across traits). Default 250 → "
            "5:1 HIGH+LOW:NEUTRAL ratio, total 1500 with the default split."
        ),
    )
    parser.add_argument(
        "--n-paired", type=int, default=500,
        help=(
            "Style A (essays): number of paired profiles to sample. Default 500. "
            "Ignored for --dataset recruitview (uses --n-synthetic-users instead)."
        ),
    )
    parser.add_argument(
        "--n-synthetic-users", type=int, default=100,
        help=(
            "Style A (recruitview): number of unique train users to sample. "
            "Each user contributes ~6 essays (one per their original train "
            "question). Default 100 → ~600 essays. Max 230 (= all train users)."
        ),
    )
    parser.add_argument(
        "--run-tag", type=str, default="",
        help=(
            "Suffix appended to every essay_id so a second invocation with "
            "the same plan size produces fresh records instead of being "
            "skipped by the idempotent-resume check. Pair with --seed to "
            "also vary the underlying question sampling on recruitview."
        ),
    )
    parser.add_argument(
        "--prompt-variant", choices=list(PROMPT_VARIANTS), default="full",
        help=(
            "Style D descriptor handling. 'full' keeps the LIWC-style word "
            "list after the trait label (default, existing behaviour); "
            "'label-only' drops it. Compare the two to disentangle lexical "
            "instruction-following from genuine style steering."
        ),
    )
    parser.add_argument(
        "--temperature", type=float, default=config.GEN_TEMPERATURE,
        help=f"Sampling temperature. Default {config.GEN_TEMPERATURE}.",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=None,
        help=(
            "Max output tokens per call. Default 1100 for essays "
            "(~600–700 words), 150 for recruitview (~30–60 words)."
        ),
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
        help="Seed for plan sampling (Style A profile sample; Style D RV question sample).",
    )
    args = parser.parse_args()

    # override=True so a .env value wins over any stale OPENAI_API_KEY left
    # in the OS environment (default load_dotenv silently keeps the OS one).
    load_dotenv(override=True)
    config.ensure_dirs()

    # Per-dataset default max-tokens (~60–100 words ≈ 130–170 tokens; pad to 250).
    if args.max_tokens is None:
        args.max_tokens = 1100 if args.dataset == "essays" else 250

    # Warn loudly if a non-default variant is used without a run-tag — the
    # idempotent-resume check would otherwise skip everything as duplicates.
    if args.prompt_variant != "full" and not args.run_tag:
        print(
            f"WARN: --prompt-variant={args.prompt_variant} without --run-tag. "
            f"If essays with these ids already exist in the JSONL (from a "
            f"previous 'full' run), they will be skipped. Use --run-tag "
            f"{args.prompt_variant.replace('-', '')} to keep both variants "
            f"side-by-side.",
            file=sys.stderr,
        )

    def _d_summary(n_per_tl: int, n_neut: int) -> str:
        return (
            f"({n_per_tl} per (trait × HIGH|LOW) × 5 × 2 = "
            f"{5 * 2 * n_per_tl}, + {n_neut} NEUTRAL pool"
            + (f", run-tag={args.run_tag!r}" if args.run_tag else "")
            + ")"
        )

    variant_tag = (
        f" variant={args.prompt_variant!r}"
        if args.prompt_variant != "full" else ""
    )
    if args.dataset == "essays":
        out_dir = config.LLM_OUTPUTS_DIR
        if args.style == "D":
            plans = style_d_plan(
                args.n_per_trait_level, args.n_neutral,
                args.run_tag, args.prompt_variant,
            )
            out_path = out_dir / "style_d_single_trait.jsonl"
            label = (
                f"Essays · Style D — {len(plans)} essays "
                + _d_summary(args.n_per_trait_level, args.n_neutral)
                + variant_tag
            )
        else:
            plans = style_a_plan(args.n_paired, args.seed, args.run_tag)
            out_path = out_dir / "style_a_paired.jsonl"
            label = (
                f"Essays · Style A — {len(plans)} paired essays"
                + (f" (run-tag={args.run_tag!r})" if args.run_tag else "")
            )
    else:  # recruitview
        out_dir = config.LLM_OUTPUTS_RV_DIR
        if args.style == "D":
            plans = style_d_recruitview_plan(
                args.n_per_trait_level, args.n_neutral,
                args.seed, args.run_tag, args.prompt_variant,
            )
            out_path = out_dir / "style_d_recruitview.jsonl"
            label = (
                f"RecruitView · Style D — {len(plans)} answers "
                + _d_summary(args.n_per_trait_level, args.n_neutral)
                + variant_tag
            )
        else:
            plans = style_a_recruitview_plan(
                args.n_synthetic_users, args.seed,
                args.run_tag, args.prompt_variant,
            )
            out_path = out_dir / "style_a_recruitview.jsonl"
            label = (
                f"RecruitView · Style A — {len(plans)} answers across "
                f"{args.n_synthetic_users} synthetic users"
                + variant_tag
                + (f" (run-tag={args.run_tag!r})" if args.run_tag else "")
            )
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
    print(f"max-tokens  : {args.max_tokens}")
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
