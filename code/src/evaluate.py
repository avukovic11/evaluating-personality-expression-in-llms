"""Evaluate LLM-generated essays against the trained personality probe.

Two analyses, one per prompt style:

**Style D — single-trait isolated.**
For each trait T:
  - Build sigmoid-prob distributions on Style-D essays where the prompted trait
    was T: LLM-HIGH, LLM-LOW. Plus the LLM-NEUTRAL pool (all NEUTRAL essays
    share an identical prompt; we pool across traits for more samples) and the
    human-test distribution (read from the trained classifier's saved preds).
  - Wasserstein-1 distances + 95% bootstrap CIs:
      W1(HIGH, LOW)        — primary "is the trait steerable?"
      W1(NEUTRAL, HIGH)    — does HIGH shift up relative to default?
      W1(NEUTRAL, LOW)     — does LOW shift down relative to default?
      W1(humans, NEUTRAL)  — is GPT-4o-mini's default near humans?
  - Overlaid KDE per trait → one PNG per trait.
  - Cross-trait contamination 5×5 heatmap: Δ(T → T') = mean(p_T' | HIGH on T) −
    mean(p_T' | LOW on T). Diagonals = direct effect; off-diagonals = bleed.

**Style A — full multi-trait paired.**
For each LLM essay vs the paired human's intended profile:
  per-trait MAE | AUC | accuracy@0.5 + macro + profile-level exact match.

The trained probe is loaded from
`code/datasets/checkpoints/<model_slug>_seed42/`. Use `--model <hf-id>` to swap
between RoBERTa / ModernBERT.

Outputs go to `code/datasets/results/llm-alignment/<style>/<model_slug>/`.

Run from `code/`:
    python -m src.evaluate --style D
    python -m src.evaluate --style A
    python -m src.evaluate --style D --model answerdotai/ModernBERT-base
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import wasserstein_distance
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from . import config
from .classifier import _sigmoid, get_device


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    out: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    if not out:
        raise ValueError(f"No records in {path}.")
    return out


def load_human_test_probs(model_slug: str) -> pd.DataFrame:
    path = config.RESULTS_DIR / model_slug / "test_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Human-test predictions not at {path}. Train the probe first:\n"
            f"  python -m src.classifier --train --model <hf-id>"
        )
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Probe inference on LLM essays (batched)
# ---------------------------------------------------------------------------

def score_essays(
    texts: list[str], model_slug: str, batch_size: int = 16,
) -> np.ndarray:
    """Run the trained probe over a list of essays. Returns (N, 5) sigmoid probs."""
    ckpt = config.CHECKPOINTS_DIR / f"{model_slug}_seed42"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Probe checkpoint not at {ckpt}. Either:\n"
            f"  1. Train it: python -m src.classifier --train --model <hf-id>\n"
            f"  2. Unzip the downloaded checkpoint zip into {ckpt}"
        )
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
    model = AutoModelForSequenceClassification.from_pretrained(str(ckpt))
    device = get_device()
    model.to(device).eval()

    all_probs: list[np.ndarray] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="scoring", leave=False):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", truncation=True,
            max_length=config.MAX_SEQ_LEN, padding=True,
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits.cpu().numpy()
        all_probs.append(_sigmoid(logits))
    return np.concatenate(all_probs, axis=0)


# ---------------------------------------------------------------------------
# Distributional metric
# ---------------------------------------------------------------------------

def bootstrap_w1(
    a: np.ndarray, b: np.ndarray, n_resamples: int = 1000, seed: int = 42,
) -> dict:
    """Wasserstein-1 + 95% bootstrap CI. NaN CIs when n < 2."""
    point = float(wasserstein_distance(a, b)) if len(a) and len(b) else float("nan")
    if len(a) < 2 or len(b) < 2:
        return {
            "w1": point, "ci_lo": float("nan"), "ci_hi": float("nan"),
            "n_a": len(a), "n_b": len(b),
        }
    rng = np.random.default_rng(seed)
    boots = np.empty(n_resamples, dtype=float)
    for k in range(n_resamples):
        a_s = rng.choice(a, size=len(a), replace=True)
        b_s = rng.choice(b, size=len(b), replace=True)
        boots[k] = wasserstein_distance(a_s, b_s)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "w1": point, "ci_lo": float(lo), "ci_hi": float(hi),
        "n_a": len(a), "n_b": len(b),
    }


# ---------------------------------------------------------------------------
# Style D
# ---------------------------------------------------------------------------

def evaluate_style_d(
    records: list[dict], probs: np.ndarray, human_df: pd.DataFrame,
    out_dir: Path, n_resamples: int,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    for i, t in enumerate(config.TRAIT_COLS):
        df[f"prob_{t}"] = probs[:, i]

    metrics: dict = {"per_trait": {}, "contamination": {}, "n_total": len(df)}
    n_neutral_total = int((df["prompted_level"] == "NEUTRAL").sum())

    for trait in config.TRAIT_COLS:
        sub = df[df["prompted_trait"] == trait]
        high = sub.loc[sub["prompted_level"] == "HIGH", f"prob_{trait}"].to_numpy()
        low = sub.loc[sub["prompted_level"] == "LOW", f"prob_{trait}"].to_numpy()
        # Pool NEUTRAL across traits (all NEUTRAL prompts are identical).
        neutral = df.loc[df["prompted_level"] == "NEUTRAL", f"prob_{trait}"].to_numpy()
        humans = human_df[f"prob_{trait}"].to_numpy()

        metrics["per_trait"][trait] = {
            "n_high": int(len(high)),
            "n_low": int(len(low)),
            "n_neutral": int(len(neutral)),
            "n_humans": int(len(humans)),
            "mean_high":    float(high.mean()) if len(high) else float("nan"),
            "mean_low":     float(low.mean()) if len(low) else float("nan"),
            "mean_neutral": float(neutral.mean()) if len(neutral) else float("nan"),
            "mean_humans":  float(humans.mean()),
            "w1_high_low":       bootstrap_w1(high, low, n_resamples),
            "w1_neutral_high":   bootstrap_w1(neutral, high, n_resamples),
            "w1_neutral_low":    bootstrap_w1(neutral, low, n_resamples),
            "w1_humans_neutral": bootstrap_w1(humans, neutral, n_resamples),
        }
        _plot_kde_for_trait(trait, humans, neutral, high, low, out_dir)

    # Cross-trait contamination heatmap
    contam = np.full((len(config.TRAIT_COLS), len(config.TRAIT_COLS)), np.nan)
    for i, t_prompt in enumerate(config.TRAIT_COLS):
        sub = df[df["prompted_trait"] == t_prompt]
        h = sub[sub["prompted_level"] == "HIGH"]
        l = sub[sub["prompted_level"] == "LOW"]
        if len(h) and len(l):
            for j, t_score in enumerate(config.TRAIT_COLS):
                contam[i, j] = h[f"prob_{t_score}"].mean() - l[f"prob_{t_score}"].mean()
    metrics["contamination"] = {
        config.TRAIT_COLS[i]: {
            config.TRAIT_COLS[j]: (None if np.isnan(contam[i, j]) else float(contam[i, j]))
            for j in range(len(config.TRAIT_COLS))
        }
        for i in range(len(config.TRAIT_COLS))
    }
    _plot_contamination(contam, out_dir)

    # Persist
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    # Also save scored essays for fast re-analysis later
    keep_cols = ["essay_id", "prompted_trait", "prompted_level"] + [
        f"prob_{t}" for t in config.TRAIT_COLS
    ]
    df[keep_cols].to_csv(out_dir / "scored_essays.csv", index=False)

    _print_style_d_summary(metrics, n_neutral_total)
    return metrics


def _plot_kde_for_trait(trait, humans, neutral, high, low, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    layers = [
        ("humans (test)", humans, "gray"),
        ("LLM NEUTRAL",   neutral, "C0"),
        ("LLM HIGH",      high,    "C2"),
        ("LLM LOW",       low,     "C3"),
    ]
    for label, arr, color in layers:
        if len(arr) >= 2:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sns.kdeplot(arr, ax=ax, label=f"{label} (n={len(arr)})",
                            color=color, fill=True, alpha=0.25, clip=(0, 1))
        elif len(arr) == 1:
            ax.axvline(arr[0], color=color, linestyle=":",
                       label=f"{label} (n=1, val={arr[0]:.2f})")
    ax.set_xlim(0, 1)
    ax.set_xlabel(f"predicted P({trait}=high)")
    ax.set_ylabel("density")
    ax.set_title(f"{trait} ({config.TRAIT_NAMES[trait]}) — Style D distributions")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"density_{trait}.png", dpi=120)
    plt.close(fig)


def _plot_contamination(contam: np.ndarray, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5))
    vmax = max(np.nanmax(np.abs(contam)), 0.01)
    sns.heatmap(
        contam, annot=True, fmt=".2f", cmap="RdBu_r",
        vmin=-vmax, vmax=vmax, center=0,
        xticklabels=config.TRAIT_COLS, yticklabels=config.TRAIT_COLS,
        cbar_kws={"label": "Δ prob (HIGH − LOW)"},
        ax=ax,
    )
    ax.set_xlabel("Scored trait")
    ax.set_ylabel("Prompted trait")
    ax.set_title("Cross-trait contamination (Style D)")
    fig.tight_layout()
    fig.savefig(out_dir / "contamination.png", dpi=120)
    plt.close(fig)


def _print_style_d_summary(metrics: dict, n_neutral_total: int) -> None:
    print(f"\n=== Style D — {metrics['n_total']} essays "
          f"({n_neutral_total} NEUTRAL pooled) ===")
    print(f"  {'trait':<6} {'n_hi':>4} {'n_lo':>4}  "
          f"{'W1(H,L)':>10} {'W1(N,H)':>10} {'W1(N,L)':>10} {'W1(hum,N)':>10}")
    for trait, m in metrics["per_trait"].items():
        print(
            f"  {trait:<6} {m['n_high']:>4} {m['n_low']:>4}  "
            f"{m['w1_high_low']['w1']:>10.3f} "
            f"{m['w1_neutral_high']['w1']:>10.3f} "
            f"{m['w1_neutral_low']['w1']:>10.3f} "
            f"{m['w1_humans_neutral']['w1']:>10.3f}"
        )


# ---------------------------------------------------------------------------
# Style A
# ---------------------------------------------------------------------------

def evaluate_style_a(records: list[dict], probs: np.ndarray, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    for i, t in enumerate(config.TRAIT_COLS):
        df[f"prob_{t}"] = probs[:, i]
        df[f"intended_{t}"] = df["intended_profile"].apply(lambda p: int(p[t]))
        df[f"pred_{t}"] = (df[f"prob_{t}"] >= 0.5).astype(int)

    metrics: dict = {"per_trait": {}, "macro": {}, "n": int(len(df))}
    for trait in config.TRAIT_COLS:
        intended = df[f"intended_{trait}"].to_numpy()
        pred = df[f"pred_{trait}"].to_numpy()
        prob = df[f"prob_{trait}"].to_numpy()
        try:
            auc = float(roc_auc_score(intended, prob))
        except ValueError:
            auc = float("nan")  # all-same labels
        metrics["per_trait"][trait] = {
            "accuracy": float((intended == pred).mean()),
            "auc": auc,
            "mae": float(np.abs(intended - prob).mean()),
            "n_intended_high": int(intended.sum()),
            "n_intended_low": int(len(intended) - intended.sum()),
        }
    metrics["macro"] = {
        "accuracy": float(np.mean([m["accuracy"] for m in metrics["per_trait"].values()])),
        "auc":      float(np.nanmean([m["auc"] for m in metrics["per_trait"].values()])),
        "mae":      float(np.mean([m["mae"] for m in metrics["per_trait"].values()])),
    }

    # Profile-level exact match
    intended_mat = df[[f"intended_{t}" for t in config.TRAIT_COLS]].to_numpy()
    pred_mat = df[[f"pred_{t}" for t in config.TRAIT_COLS]].to_numpy()
    metrics["profile_exact_match"] = float((intended_mat == pred_mat).all(axis=1).mean())

    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    keep_cols = (
        ["essay_id"]
        + [f"intended_{t}" for t in config.TRAIT_COLS]
        + [f"prob_{t}" for t in config.TRAIT_COLS]
        + [f"pred_{t}" for t in config.TRAIT_COLS]
    )
    df[keep_cols].to_csv(out_dir / "predictions.csv", index=False)

    _print_style_a_summary(metrics)
    return metrics


def _print_style_a_summary(metrics: dict) -> None:
    print(f"\n=== Style A — n={metrics['n']} paired essays ===")
    print(f"  {'trait':<6}  {'acc':>5}  {'auc':>5}  {'mae':>5}  ({'intended H/L'})")
    for trait, m in metrics["per_trait"].items():
        print(
            f"  {trait:<6}  {m['accuracy']:>5.3f}  "
            f"{m['auc']:>5.3f}  {m['mae']:>5.3f}  "
            f"({m['n_intended_high']}/{m['n_intended_low']})"
        )
    macro = metrics["macro"]
    print(f"  {'macro':<6}  {macro['accuracy']:>5.3f}  "
          f"{macro['auc']:>5.3f}  {macro['mae']:>5.3f}")
    print(f"  profile exact match: {metrics['profile_exact_match']:.3f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--style", choices=["D", "A"], required=True)
    parser.add_argument(
        "--model", type=str, default=config.CLASSIFIER_MODEL,
        help=f"Probe HF id; default {config.CLASSIFIER_MODEL}. "
             f"Checkpoint must exist at <CHECKPOINTS_DIR>/<model_slug>_seed42.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Batch size for probe inference.",
    )
    parser.add_argument(
        "--bootstrap", type=int, default=1000,
        help="Bootstrap resamples for Wasserstein CIs. Default 1000.",
    )
    args = parser.parse_args()

    config.ensure_dirs()
    model_slug = args.model.split("/")[-1]

    if args.style == "D":
        jsonl = config.LLM_OUTPUTS_DIR / "style_d_single_trait.jsonl"
        out_dir = config.RESULTS_DIR / "llm-alignment" / "style_d" / model_slug
    else:
        jsonl = config.LLM_OUTPUTS_DIR / "style_a_paired.jsonl"
        out_dir = config.RESULTS_DIR / "llm-alignment" / "style_a" / model_slug

    records = load_jsonl(jsonl)
    print(f"essays  : {len(records)} loaded from {jsonl}")
    print(f"probe   : {args.model}")
    print(f"output  : {out_dir}")

    # Warn on tiny samples
    if args.style == "D":
        n_per = max(1, len(records) // 15)  # 5 traits × 3 levels = 15 conditions
        if n_per < 10:
            print(
                f"WARN: ~{n_per} essay per (trait × level). "
                f"Wasserstein/bootstrap CIs will be wide / NaN. "
                f"For real conclusions use --n-per-condition >= 50."
            )
    else:
        if len(records) < 20:
            print(
                f"WARN: only {len(records)} paired essays. "
                f"Per-trait AUC needs both classes and is unstable below ~20."
            )

    texts = [r["generated_text"] for r in records]
    probs = score_essays(texts, model_slug, batch_size=args.batch_size)

    if args.style == "D":
        human_df = load_human_test_probs(model_slug)
        evaluate_style_d(records, probs, human_df, out_dir, args.bootstrap)
    else:
        evaluate_style_a(records, probs, out_dir)

    print(f"\nDone. Artifacts in {out_dir}")


if __name__ == "__main__":
    main()
