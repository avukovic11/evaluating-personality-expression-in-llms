"""Evaluate LLM-generated essays against the trained personality probe.

Two analyses, one per prompt style, applied independently to each dataset
(`--dataset essays` or `--dataset recruitview`):

**Style B — single-trait isolated.**
For each trait T:
  - Build probe-output distributions on Style-B essays where the prompted trait
    was T: LLM-HIGH, LLM-LOW. Plus the LLM-NEUTRAL pool (all NEUTRAL essays
    share an identical prompt; we pool across traits for more samples) and the
    human-test distribution (read from the trained classifier's saved preds).
    For essays: sigmoid probabilities (0–1). For recruitview: raw z-scores.
  - Wasserstein-1 distances + 95% bootstrap CIs:
      W1(HIGH, LOW)        — primary "is the trait steerable?"
      W1(NEUTRAL, HIGH)    — does HIGH shift up relative to default?
      W1(NEUTRAL, LOW)     — does LOW shift down relative to default?
      W1(humans, NEUTRAL)  — is the model's default near humans?
  - Overlaid KDE per trait → one PNG per trait.
  - Cross-trait contamination 5×5 heatmap: Δ(T → T') = mean(score_T' | HIGH on T)
    − mean(score_T' | LOW on T). Diagonals = direct effect; off-diagonals = bleed.

**Style A — full multi-trait paired.**
  - Essays: per-trait MAE | AUC | accuracy@0.5 + macro + profile-level exact match
    (paired binary profile).
  - RecruitView: per-trait Spearman ρ | Pearson r | MAE on z-scores + macro +
    per-essay 5-vector Spearman ρ between predicted and intended z-scores.

The trained probe is loaded from
`code/datasets/checkpoints/<results_slug>_seed42/`, where `results_slug` is
`<model>` for essays and `<model>_recruitview` for recruitview.

Outputs go to:
  essays      → `code/datasets/results/llm-alignment/<style>/<model_slug>/`
  recruitview → `code/datasets/results/llm-alignment/recruitview/<style>/<model_slug>/`

Run from `code/`:
    python -m src.evaluate --style B
    python -m src.evaluate --style A --model answerdotai/ModernBERT-base
    python -m src.evaluate --dataset recruitview --style B
    python -m src.evaluate --dataset recruitview --style A
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
from scipy.stats import pearsonr, spearmanr, wasserstein_distance
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from . import config
from .classifier import _result_dir_name, _sigmoid, get_device


# ---------------------------------------------------------------------------
# Dataset specs
# ---------------------------------------------------------------------------

def _trait_cols(dataset: str) -> list[str]:
    return (
        config.TRAIT_COLS if dataset == "essays"
        else config.RECRUITVIEW_TRAIT_COLS
    )


def _trait_display(dataset: str, t: str) -> str:
    if dataset == "essays":
        return config.TRAIT_NAMES[t]
    return config.RECRUITVIEW_TRAIT_NAMES[t]


def _human_score_col(dataset: str, trait: str) -> str:
    """Column in the human-test predictions CSV that holds the probe's score.

    Essays      : sigmoid prob, column `prob_<trait>`.
    RecruitView : raw regression output, column `pred_<trait>`.
    """
    return f"prob_{trait}" if dataset == "essays" else f"pred_{trait}"


def _llm_outputs_dir(dataset: str) -> Path:
    return config.LLM_OUTPUTS_DIR if dataset == "essays" else config.LLM_OUTPUTS_RV_DIR


def _llm_align_root(dataset: str) -> Path:
    base = config.RESULTS_DIR / "llm-alignment"
    return base if dataset == "essays" else base / "recruitview"


def _style_jsonl_name(dataset: str, style: str) -> str:
    if dataset == "essays":
        return "style_b_single_trait.jsonl" if style == "B" else "style_a_paired.jsonl"
    return "style_b_recruitview.jsonl" if style == "B" else "style_a_recruitview.jsonl"


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


def load_human_test_probs(model_slug: str, dataset: str = "essays") -> pd.DataFrame:
    """Load the human-test prediction CSV the classifier wrote.

    `model_slug` is the *base* HF model id (e.g. "roberta-base"); the actual
    directory adds `_recruitview` for the recruitview track.
    """
    results_slug = _result_dir_name(model_slug, dataset)
    path = config.RESULTS_DIR / results_slug / "test_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Human-test predictions not at {path}. Train the probe first:\n"
            f"  python -m src.classifier --train --model <hf-id> "
            f"--dataset {dataset}"
        )
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Probe inference on LLM essays (batched)
# ---------------------------------------------------------------------------

def score_essays(
    texts: list[str], model_slug: str, batch_size: int = 16,
    dataset: str = "essays", max_seq_len: int | None = None,
) -> np.ndarray:
    """Run the trained probe over a list of essays. Returns (N, 5) scores.

    Essays      : sigmoid probabilities (0–1).
    RecruitView : raw regression outputs (z-scores, no transform).

    `max_seq_len` should match the probe's training-time max length. Pass
    e.g. 2048 to use a ModernBERT checkpoint trained at `_seq2048`.
    """
    if max_seq_len is None:
        max_seq_len = config.MAX_SEQ_LEN
    results_slug = _result_dir_name(model_slug, dataset)
    ckpt = config.CHECKPOINTS_DIR / f"{results_slug}_seed42"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Probe checkpoint not at {ckpt}. Either:\n"
            f"  1. Train it: python -m src.classifier --train "
            f"--dataset {dataset} --model <hf-id>\n"
            f"  2. Unzip the downloaded checkpoint zip into {ckpt}"
        )
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
    model = AutoModelForSequenceClassification.from_pretrained(str(ckpt))
    device = get_device()
    model.to(device).eval()

    all_out: list[np.ndarray] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="scoring", leave=False):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", truncation=True,
            max_length=max_seq_len, padding=True,
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits.cpu().numpy()
        all_out.append(_sigmoid(logits) if dataset == "essays" else logits)
    return np.concatenate(all_out, axis=0)


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
# Style B
# ---------------------------------------------------------------------------

def evaluate_style_b(
    records: list[dict], probs: np.ndarray, human_df: pd.DataFrame,
    out_dir: Path, n_resamples: int, dataset: str = "essays",
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    trait_cols = _trait_cols(dataset)
    df = pd.DataFrame(records)
    # Canonical internal column: `score_<trait>`. For essays this is the
    # sigmoid prob (0–1); for recruitview it's the raw predicted z-score.
    for i, t in enumerate(trait_cols):
        df[f"score_{t}"] = probs[:, i]

    metrics: dict = {
        "dataset": dataset,
        "per_trait": {},
        "contamination": {},
        "n_total": len(df),
    }
    n_neutral_total = int((df["prompted_level"] == "NEUTRAL").sum())

    for trait in trait_cols:
        sub = df[df["prompted_trait"] == trait]
        high = sub.loc[sub["prompted_level"] == "HIGH", f"score_{trait}"].to_numpy()
        low = sub.loc[sub["prompted_level"] == "LOW", f"score_{trait}"].to_numpy()
        # Pool NEUTRAL across traits (all NEUTRAL prompts are identical).
        neutral = df.loc[df["prompted_level"] == "NEUTRAL", f"score_{trait}"].to_numpy()
        humans = human_df[_human_score_col(dataset, trait)].to_numpy()

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
        _plot_kde_for_trait(trait, humans, neutral, high, low, out_dir, dataset)

    # Cross-trait contamination heatmap (mean(HIGH) − mean(LOW)).
    contam = np.full((len(trait_cols), len(trait_cols)), np.nan)
    for i, t_prompt in enumerate(trait_cols):
        sub = df[df["prompted_trait"] == t_prompt]
        h = sub[sub["prompted_level"] == "HIGH"]
        l = sub[sub["prompted_level"] == "LOW"]
        if len(h) and len(l):
            for j, t_score in enumerate(trait_cols):
                contam[i, j] = (
                    h[f"score_{t_score}"].mean() - l[f"score_{t_score}"].mean()
                )
    metrics["contamination"] = {
        trait_cols[i]: {
            trait_cols[j]: (None if np.isnan(contam[i, j]) else float(contam[i, j]))
            for j in range(len(trait_cols))
        }
        for i in range(len(trait_cols))
    }
    _plot_contamination(contam, trait_cols, out_dir, dataset)

    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    keep_cols = ["essay_id", "prompted_trait", "prompted_level"] + [
        f"score_{t}" for t in trait_cols
    ]
    df[keep_cols].to_csv(out_dir / "scored_essays.csv", index=False)

    _print_style_b_summary(metrics, n_neutral_total, dataset)
    return metrics


def _plot_kde_for_trait(
    trait, humans, neutral, high, low, out_dir: Path, dataset: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    layers = [
        ("humans (test)", humans, "gray"),
        ("LLM NEUTRAL",   neutral, "C0"),
        ("LLM HIGH",      high,    "C2"),
        ("LLM LOW",       low,     "C3"),
    ]
    if dataset == "essays":
        kde_clip = (0, 1)
        xlim = (0, 1)
        xlabel = f"predicted P({trait}=high)"
    else:
        # Empirical range of the probe's predicted z-scores plus a small pad.
        all_vals = np.concatenate([arr for _, arr, _ in layers if len(arr)])
        pad = 0.3
        xlim = (
            float(np.nanmin(all_vals) - pad) if len(all_vals) else -3.0,
            float(np.nanmax(all_vals) + pad) if len(all_vals) else 3.0,
        )
        kde_clip = xlim
        xlabel = f"predicted z-score ({trait})"

    for label, arr, color in layers:
        if len(arr) >= 2:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sns.kdeplot(arr, ax=ax, label=f"{label} (n={len(arr)})",
                            color=color, fill=True, alpha=0.25, clip=kde_clip)
        elif len(arr) == 1:
            ax.axvline(arr[0], color=color, linestyle=":",
                       label=f"{label} (n=1, val={arr[0]:.2f})")
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.set_title(
        f"{trait} ({_trait_display(dataset, trait)}) — Style B distributions"
    )
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"density_{trait}.png", dpi=120)
    plt.close(fig)


def _plot_contamination(
    contam: np.ndarray, trait_cols: list[str], out_dir: Path, dataset: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5))
    vmax = max(np.nanmax(np.abs(contam)), 0.01)
    cbar_label = (
        "Δ prob (HIGH − LOW)" if dataset == "essays"
        else "Δ z-score (HIGH − LOW)"
    )
    sns.heatmap(
        contam, annot=True, fmt=".2f", cmap="RdBu_r",
        vmin=-vmax, vmax=vmax, center=0,
        xticklabels=trait_cols, yticklabels=trait_cols,
        cbar_kws={"label": cbar_label},
        ax=ax,
    )
    ax.set_xlabel("Scored trait")
    ax.set_ylabel("Prompted trait")
    ax.set_title("Cross-trait contamination (Style B)")
    fig.tight_layout()
    fig.savefig(out_dir / "contamination.png", dpi=120)
    plt.close(fig)


def _print_style_b_summary(
    metrics: dict, n_neutral_total: int, dataset: str = "essays",
) -> None:
    item_label = "essays" if dataset == "essays" else "answers"
    width = 6 if dataset == "essays" else 18
    print(f"\n=== Style B ({dataset}) — {metrics['n_total']} {item_label} "
          f"({n_neutral_total} NEUTRAL pooled) ===")
    print(f"  {'trait':<{width}} {'n_hi':>4} {'n_lo':>4}  "
          f"{'W1(H,L)':>10} {'W1(N,H)':>10} {'W1(N,L)':>10} {'W1(hum,N)':>10}")
    for trait, m in metrics["per_trait"].items():
        print(
            f"  {trait:<{width}} {m['n_high']:>4} {m['n_low']:>4}  "
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
    print(f"\n=== Style A (essays) — n={metrics['n']} paired essays ===")
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


def evaluate_style_a_recruitview(
    records: list[dict], preds: np.ndarray, out_dir: Path,
) -> dict:
    """Per-trait Spearman/Pearson/MAE on z-scores at two granularities.

    `preds` is (N, 5) raw predicted z-scores from the regression probe.
    Records carry `intended_z`, `intended_levels`, and `paired_user_no`.

    Reports two parallel metric sets:
      - per-essay: Spearman over N (intended_z, pred_z) pairs per trait.
        Includes within-user noise from individual questions.
      - per-user: average pred_z per `paired_user_no`, compare to that
        user's (constant) intended_z. Matches how we evaluate the probe
        on humans (user-level Spearman). The headline number for the paper.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    trait_cols = config.RECRUITVIEW_TRAIT_COLS
    df = pd.DataFrame(records)
    for i, t in enumerate(trait_cols):
        df[f"pred_z_{t}"] = preds[:, i]
        df[f"intended_z_{t}"] = df["intended_z"].apply(lambda d, t=t: float(d[t]))
        df[f"intended_level_{t}"] = df["intended_levels"].apply(
            lambda d, t=t: d[t]
        )

    metrics: dict = {
        "dataset": "recruitview",
        "per_essay": {"per_trait": {}, "macro": {}, "n": int(len(df))},
        "per_user":  {"per_trait": {}, "macro": {}},
    }

    def _stats(yt, yp):
        if np.std(yt) > 0 and np.std(yp) > 0:
            rho = float(spearmanr(yt, yp)[0])
            r = float(pearsonr(yt, yp)[0])
            rho = 0.0 if np.isnan(rho) else rho
            r = 0.0 if np.isnan(r) else r
        else:
            rho = r = 0.0
        mae = float(np.abs(yt - yp).mean())
        return rho, r, mae

    # --- Per-essay ---
    rhos, rs, maes = [], [], []
    for trait in trait_cols:
        yt = df[f"intended_z_{trait}"].to_numpy()
        yp = df[f"pred_z_{trait}"].to_numpy()
        rho, r, mae = _stats(yt, yp)
        rhos.append(rho); rs.append(r); maes.append(mae)
        metrics["per_essay"]["per_trait"][trait] = {
            "spearman": rho, "pearson": r, "mae": mae,
            "mean_intended_z": float(yt.mean()),
            "mean_pred_z": float(yp.mean()),
        }
    metrics["per_essay"]["macro"] = {
        "spearman": float(np.mean(rhos)),
        "pearson":  float(np.mean(rs)),
        "mae":      float(np.mean(maes)),
    }

    # --- Per-user (group by paired_user_no) ---
    if "paired_user_no" in df.columns:
        agg = df.groupby("paired_user_no").agg(
            **{f"_t_{t}": (f"intended_z_{t}", "first") for t in trait_cols},
            **{f"_p_{t}": (f"pred_z_{t}",     "mean")  for t in trait_cols},
        )
        n_users = int(len(agg))
        u_rhos, u_rs, u_maes = [], [], []
        for trait in trait_cols:
            a = agg[f"_t_{trait}"].to_numpy()
            b = agg[f"_p_{trait}"].to_numpy()
            rho, r, mae = _stats(a, b)
            u_rhos.append(rho); u_rs.append(r); u_maes.append(mae)
            metrics["per_user"]["per_trait"][trait] = {
                "spearman": rho, "pearson": r, "mae": mae,
            }
        metrics["per_user"]["macro"] = {
            "spearman": float(np.mean(u_rhos)),
            "pearson":  float(np.mean(u_rs)),
            "mae":      float(np.mean(u_maes)),
        }
        metrics["per_user"]["n_users"] = n_users

    # Per-essay 5-vector profile correlation (kept for compat / drill-down).
    intended_mat = df[[f"intended_z_{t}" for t in trait_cols]].to_numpy()
    pred_mat = df[[f"pred_z_{t}" for t in trait_cols]].to_numpy()
    per_essay_rho = np.empty(len(df), dtype=float)
    for k in range(len(df)):
        a, b = intended_mat[k], pred_mat[k]
        if np.std(a) == 0 or np.std(b) == 0:
            per_essay_rho[k] = np.nan
        else:
            rr = spearmanr(a, b)[0]
            per_essay_rho[k] = np.nan if np.isnan(rr) else float(rr)
    valid = ~np.isnan(per_essay_rho)
    metrics["per_essay_profile_rho"] = {
        "mean":   float(per_essay_rho[valid].mean()) if valid.any() else float("nan"),
        "median": float(np.median(per_essay_rho[valid])) if valid.any() else float("nan"),
        "n_valid": int(valid.sum()),
    }

    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    keep_cols = (
        ["essay_id", "paired_user_no", "paired_question_id"]
        + [f"intended_z_{t}" for t in trait_cols]
        + [f"intended_level_{t}" for t in trait_cols]
        + [f"pred_z_{t}" for t in trait_cols]
    )
    df["per_essay_profile_rho"] = per_essay_rho
    keep_cols.append("per_essay_profile_rho")
    df[keep_cols].to_csv(out_dir / "predictions.csv", index=False)

    _print_style_a_recruitview_summary(metrics)
    return metrics


def _print_style_a_recruitview_summary(metrics: dict) -> None:
    pe = metrics["per_essay"]
    pu = metrics.get("per_user", {})
    print(
        f"\n=== Style A (recruitview) — {pe['n']} answers across "
        f"{pu.get('n_users', '?')} synthetic users ==="
    )
    print(
        f"  {'trait':<18}  "
        f"{'rho/clip':>9} {'r/clip':>8} {'mae/clip':>9}  |  "
        f"{'rho/user':>9} {'r/user':>8} {'mae/user':>9}"
    )
    for trait in pe["per_trait"]:
        e = pe["per_trait"][trait]
        u = pu.get("per_trait", {}).get(trait, {})
        line = (
            f"  {trait:<18}  "
            f"{e['spearman']:>+9.3f} {e['pearson']:>+8.3f} {e['mae']:>9.3f}  |  "
        )
        if u:
            line += (
                f"{u['spearman']:>+9.3f} {u['pearson']:>+8.3f} {u['mae']:>9.3f}"
            )
        else:
            line += "(no user grouping)"
        print(line)
    em, um = pe["macro"], pu.get("macro", {})
    line = (
        f"  {'macro':<18}  "
        f"{em['spearman']:>+9.3f} {em['pearson']:>+8.3f} {em['mae']:>9.3f}  |  "
    )
    if um:
        line += f"{um['spearman']:>+9.3f} {um['pearson']:>+8.3f} {um['mae']:>9.3f}"
    print(line)
    per = metrics["per_essay_profile_rho"]
    print(
        f"  per-essay profile Spearman ρ: mean={per['mean']:+.3f}  "
        f"median={per['median']:+.3f}  (n_valid={per['n_valid']})"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dataset", choices=["essays", "recruitview"], default="essays",
        help="Which dataset's probe + LLM essays to evaluate.",
    )
    parser.add_argument("--style", choices=["B", "A"], required=True)
    parser.add_argument(
        "--model", type=str, default=config.CLASSIFIER_MODEL,
        help=(
            f"Probe HF id; default {config.CLASSIFIER_MODEL}. "
            "The checkpoint dir is <CHECKPOINTS_DIR>/<model>_seed42 for "
            "essays and <CHECKPOINTS_DIR>/<model>_recruitview_seed42 for "
            "recruitview (matches classifier.py)."
        ),
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Batch size for probe inference.",
    )
    parser.add_argument(
        "--bootstrap", type=int, default=1000,
        help="Bootstrap resamples for Wasserstein CIs. Default 1000.",
    )
    parser.add_argument(
        "--max-seq-len", type=int, default=config.MAX_SEQ_LEN,
        help=(
            f"Tokenize LLM essays at this length. Default {config.MAX_SEQ_LEN}; "
            f"pass 2048 to use a ModernBERT checkpoint trained at _seq2048. "
            f"The model_slug auto-suffixes with _seqN when N differs from "
            f"the default, so the right checkpoint is located."
        ),
    )
    args = parser.parse_args()

    config.ensure_dirs()
    model_slug = args.model.split("/")[-1]
    if args.max_seq_len != config.MAX_SEQ_LEN:
        model_slug = f"{model_slug}_seq{args.max_seq_len}"

    style_subdir = "style_b" if args.style == "B" else "style_a"
    jsonl = _llm_outputs_dir(args.dataset) / _style_jsonl_name(args.dataset, args.style)
    out_dir = _llm_align_root(args.dataset) / style_subdir / model_slug

    records = load_jsonl(jsonl)
    print(f"dataset : {args.dataset}")
    print(f"essays  : {len(records)} loaded from {jsonl}")
    print(f"probe   : {args.model}")
    print(f"output  : {out_dir}")

    # Warn on tiny samples
    if args.style == "B":
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
                f"Per-trait metrics are unstable below ~20 samples."
            )

    texts = [r["generated_text"] for r in records]
    probs = score_essays(
        texts, model_slug, batch_size=args.batch_size, dataset=args.dataset,
        max_seq_len=args.max_seq_len,
    )

    if args.style == "B":
        human_df = load_human_test_probs(model_slug, dataset=args.dataset)
        evaluate_style_b(
            records, probs, human_df, out_dir, args.bootstrap,
            dataset=args.dataset,
        )
    else:
        if args.dataset == "essays":
            evaluate_style_a(records, probs, out_dir)
        else:
            evaluate_style_a_recruitview(records, probs, out_dir)

    print(f"\nDone. Artifacts in {out_dir}")


if __name__ == "__main__":
    main()
