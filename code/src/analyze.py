"""Linguistic analysis (RQ3): LIWC stats, Style A error dumps, optional SHAP.

Three analyses on the LLM-generated essays vs human essays. Default run is
fast (LIWC + error dumps); SHAP is opt-in via `--shap` because it's slow.

1. **LIWC-style feature comparison** (default): for each trait T, compute
   per-essay LIWC features (reusing `liwc_features` from src.baselines) on three
   conditions:
     - humans-test (n=248)
     - Style-D HIGH-on-T
     - Style-D LOW-on-T
   Report mean ± std per condition, Mann–Whitney U p-values, and Cliff's δ
   effect size for three pairwise comparisons: HIGH vs LOW, humans vs HIGH,
   humans vs LOW.

2. **Style A error dumps** (default): for each trait, dump up to 10 misaligned
   essays (intended ≠ predicted at threshold 0.5) with full text + both
   profiles, for qualitative inspection. Requires evaluate.py to have been run
   for Style A first (uses its predictions.csv).

3. **SHAP token attribution** (`--shap`; slow): for each trait T, sample N
   essays per condition (humans / D-HIGH / D-LOW), compute token-level SHAP
   over the probe's prediction for T, and aggregate top-K tokens by mean
   |SHAP value|. Output side-by-side per-trait CSV.

Outputs go to `code/datasets/results/llm-alignment/analysis/<model_slug>/`:
    liwc_per_trait.csv               # long-form: trait × condition × feature
    liwc_stats_per_trait.csv         # MW p + Cliff's δ for each pairwise comp
    errors_per_trait/<trait>.txt     # qualitative misalignment dumps
    shap_<trait>.csv                 # only with --shap

Run from `code/`:
    python -m src.analyze                            # LIWC + errors (fast)
    python -m src.analyze --shap                     # also run SHAP (slow)
    python -m src.analyze --shap --n-shap 10         # SHAP on a tiny sample
    python -m src.analyze --skip-errors              # LIWC only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from tqdm import tqdm

from . import config
from .baselines import _ensure_nltk, liwc_features
from .data import load_essays, load_splits


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
            if line:
                out.append(json.loads(line))
    return out


# ---------------------------------------------------------------------------
# LIWC comparison
# ---------------------------------------------------------------------------

def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Non-parametric effect size for two independent samples.

    Returns δ ∈ [-1, 1]. Interpretation (Romano et al., 2006):
      |δ| < 0.147   negligible
      |δ| < 0.33    small
      |δ| < 0.474   medium
      |δ| ≥ 0.474   large
    """
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return float("nan")
    diffs = x[:, None] - y[None, :]
    above = int((diffs > 0).sum())
    below = int((diffs < 0).sum())
    return (above - below) / (nx * ny)


def _extract_features(texts: list[str], desc: str) -> pd.DataFrame:
    rows = [liwc_features(t) for t in tqdm(texts, desc=desc, leave=False)]
    return pd.DataFrame(rows)


def run_liwc_comparison(
    style_d_records: list[dict], human_texts: list[str], out_dir: Path,
) -> None:
    print("\n=== LIWC feature comparison ===")
    _ensure_nltk()

    df_d = pd.DataFrame(style_d_records)
    print(f"Humans (test): {len(human_texts)} essays")
    feat_humans = _extract_features(human_texts, "LIWC humans")

    summary_rows: list[dict] = []
    stats_rows: list[dict] = []

    for trait in config.TRAIT_COLS:
        sub = df_d[df_d["prompted_trait"] == trait]
        high_texts = sub.loc[sub["prompted_level"] == "HIGH", "generated_text"].tolist()
        low_texts = sub.loc[sub["prompted_level"] == "LOW", "generated_text"].tolist()
        if not high_texts or not low_texts:
            print(f"  {trait}: skipped (HIGH={len(high_texts)}, LOW={len(low_texts)})")
            continue

        feat_high = _extract_features(high_texts, f"LIWC {trait}-HIGH")
        feat_low = _extract_features(low_texts, f"LIWC {trait}-LOW")

        for feature in feat_humans.columns:
            h = feat_humans[feature].to_numpy()
            hi = feat_high[feature].to_numpy()
            lo = feat_low[feature].to_numpy()

            for cond_name, arr in [("humans", h), ("high", hi), ("low", lo)]:
                summary_rows.append({
                    "trait": trait, "feature": feature, "condition": cond_name,
                    "n": int(len(arr)),
                    "mean": float(arr.mean()) if len(arr) else float("nan"),
                    "std": float(arr.std()) if len(arr) else float("nan"),
                })

            for a_name, a, b_name, b in [
                ("high", hi, "low", lo),
                ("humans", h, "high", hi),
                ("humans", h, "low", lo),
            ]:
                try:
                    p = float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
                except ValueError:
                    p = float("nan")
                stats_rows.append({
                    "trait": trait, "feature": feature,
                    "comparison": f"{a_name}_vs_{b_name}",
                    "n_a": int(len(a)), "n_b": int(len(b)),
                    "mann_whitney_p": p,
                    "cliffs_delta": cliffs_delta(a, b),
                })

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(out_dir / "liwc_per_trait.csv", index=False)
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(out_dir / "liwc_stats_per_trait.csv", index=False)

    print("\n  Top |cliffs_delta| (HIGH vs LOW), per trait:")
    for trait in config.TRAIT_COLS:
        sub = stats_df[
            (stats_df["trait"] == trait)
            & (stats_df["comparison"] == "high_vs_low")
        ].copy()
        if sub.empty:
            continue
        sub["abs_delta"] = sub["cliffs_delta"].abs()
        print(f"\n    {trait} ({config.TRAIT_NAMES[trait]}):")
        for _, row in sub.nlargest(5, "abs_delta").iterrows():
            print(
                f"      {row['feature']:<25} d={row['cliffs_delta']:+.2f}  "
                f"p={row['mann_whitney_p']:.3f}"
            )


# ---------------------------------------------------------------------------
# Style A error dumps
# ---------------------------------------------------------------------------

def run_error_dumps(
    style_a_records: list[dict], predictions_path: Path, out_dir: Path,
    per_trait_limit: int = 10,
) -> None:
    print("\n=== Style A error dumps ===")
    if not predictions_path.exists():
        print(f"  SKIP — predictions not at {predictions_path}.")
        print("  Run `python -m src.evaluate --style A` first.")
        return

    pred_df = pd.read_csv(predictions_path)
    text_by_id = {r["essay_id"]: r["generated_text"] for r in style_a_records}

    err_dir = out_dir / "errors_per_trait"
    err_dir.mkdir(parents=True, exist_ok=True)

    for trait in config.TRAIT_COLS:
        intended_col, pred_col, prob_col = (
            f"intended_{trait}", f"pred_{trait}", f"prob_{trait}",
        )
        if intended_col not in pred_df.columns:
            continue
        mis = pred_df[pred_df[intended_col] != pred_df[pred_col]]
        print(f"  {trait}: {len(mis)} misaligned essays (of {len(pred_df)})")
        if mis.empty:
            continue
        sample = mis.head(per_trait_limit)
        path = err_dir / f"{trait}.txt"
        with open(path, "w", encoding="utf-8") as f:
            for _, row in sample.iterrows():
                aid = row["essay_id"]
                intended = {t: int(row[f"intended_{t}"]) for t in config.TRAIT_COLS}
                predicted = {t: int(row[f"pred_{t}"]) for t in config.TRAIT_COLS}
                probs = {t: float(row[f"prob_{t}"]) for t in config.TRAIT_COLS}
                text = text_by_id.get(aid, "<<text not found>>")
                f.write("=" * 70 + "\n")
                f.write(f"essay_id          : {aid}\n")
                f.write(
                    f"misaligned on     : {trait}  "
                    f"(intended={int(row[intended_col])}, "
                    f"predicted={int(row[pred_col])}, "
                    f"prob={float(row[prob_col]):.3f})\n"
                )
                f.write(f"intended profile  : {intended}\n")
                f.write(f"predicted profile : {predicted}\n")
                f.write(
                    "all probs         : { "
                    + ", ".join(f"{t}={probs[t]:.2f}" for t in config.TRAIT_COLS)
                    + " }\n\n"
                )
                f.write(text + "\n\n")


# ---------------------------------------------------------------------------
# SHAP token attribution (optional)
# ---------------------------------------------------------------------------

def run_shap(
    style_d_records: list[dict], human_texts: list[str], model_slug: str,
    n_shap: int, top_k: int, out_dir: Path,
) -> None:
    print(f"\n=== SHAP token attribution (n_shap={n_shap} per condition) ===")
    print("    Slow: roughly 20–60 s per essay on CPU, 3–8 s on GPU.")

    import shap
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from .classifier import _sigmoid, get_device

    ckpt = config.CHECKPOINTS_DIR / f"{model_slug}_seed42"
    if not ckpt.exists():
        print(f"  SKIP — checkpoint not at {ckpt}.")
        return
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
    model = AutoModelForSequenceClassification.from_pretrained(str(ckpt))
    device = get_device()
    model.to(device).eval()

    df_d = pd.DataFrame(style_d_records)
    masker = shap.maskers.Text(tokenizer)

    rng = np.random.default_rng(config.SEED)
    human_sample = list(rng.choice(
        np.asarray(human_texts, dtype=object),
        size=min(n_shap, len(human_texts)),
        replace=False,
    ))

    for trait_idx, trait in enumerate(config.TRAIT_COLS):
        sub = df_d[df_d["prompted_trait"] == trait]
        high_texts = sub.loc[sub["prompted_level"] == "HIGH", "generated_text"].tolist()[:n_shap]
        low_texts = sub.loc[sub["prompted_level"] == "LOW", "generated_text"].tolist()[:n_shap]
        if not high_texts or not low_texts:
            print(f"  {trait}: skipped (HIGH={len(high_texts)}, LOW={len(low_texts)})")
            continue

        def predict_trait(texts):
            inputs = tokenizer(
                list(texts), return_tensors="pt", truncation=True,
                max_length=config.MAX_SEQ_LEN, padding=True,
            ).to(device)
            with torch.no_grad():
                logits = model(**inputs).logits.cpu().numpy()
            return _sigmoid(logits)[:, trait_idx]

        explainer = shap.Explainer(predict_trait, masker)

        rows: list[dict] = []
        for cond_name, cond_texts in [
            ("humans", human_sample),
            ("high", high_texts),
            ("low", low_texts),
        ]:
            print(f"  {trait} / {cond_name}: SHAP on {len(cond_texts)} essays")
            try:
                sv = explainer(cond_texts)
            except Exception as e:
                print(f"    skipped: {e}")
                continue
            for token, mean_abs in _aggregate_shap(sv, top_k=top_k):
                rows.append({
                    "trait": trait, "condition": cond_name,
                    "token": token, "mean_abs_shap": mean_abs,
                })

        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_dir / f"shap_{trait}.csv", index=False)


def _aggregate_shap(sv, top_k: int) -> list[tuple[str, float]]:
    """Aggregate token-level |SHAP| across documents; return top-K tokens."""
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for i in range(len(sv)):
        tokens = sv.data[i]
        values = sv.values[i]
        for tok, val in zip(tokens, values):
            # Strip subword markers (Ġ for RoBERTa, ▁ for SentencePiece).
            tok = tok.replace("Ġ", "").replace("▁", "").strip().lower()
            if not tok or len(tok) < 2 or not any(c.isalpha() for c in tok):
                continue
            sums[tok] = sums.get(tok, 0.0) + abs(float(val))
            counts[tok] = counts.get(tok, 0) + 1
    means = [(t, sums[t] / counts[t]) for t in sums]
    means.sort(key=lambda x: -x[1])
    return means[:top_k]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model", type=str, default=config.CLASSIFIER_MODEL,
        help=f"Probe model slug (also names the output dir). Default {config.CLASSIFIER_MODEL}.",
    )
    parser.add_argument("--skip-liwc", action="store_true",
                       help="Skip the LIWC comparison.")
    parser.add_argument("--skip-errors", action="store_true",
                       help="Skip the Style A error dumps.")
    parser.add_argument("--shap", action="store_true",
                       help="Run SHAP token attribution (slow).")
    parser.add_argument("--n-shap", type=int, default=30,
                       help="Essays per condition for SHAP. Default 30.")
    parser.add_argument("--top-k", type=int, default=20,
                       help="Top-K tokens to keep per (trait, condition).")
    args = parser.parse_args()

    config.ensure_dirs()
    model_slug = args.model.split("/")[-1]

    style_d_path = config.LLM_OUTPUTS_DIR / "style_d_single_trait.jsonl"
    style_d_records = load_jsonl(style_d_path)
    print(f"Style D: {len(style_d_records)} essays from {style_d_path}")

    style_a_path = config.LLM_OUTPUTS_DIR / "style_a_paired.jsonl"
    style_a_records = load_jsonl(style_a_path) if style_a_path.exists() else []
    if style_a_records:
        print(f"Style A: {len(style_a_records)} essays from {style_a_path}")
    else:
        print(f"Style A: none ({style_a_path} not found; error dumps will skip)")

    df = load_essays()
    splits = load_splits(df)
    human_texts = splits["test"]["TEXT"].tolist()

    out_dir = config.RESULTS_DIR / "llm-alignment" / "analysis" / model_slug
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"output : {out_dir}")

    if not args.skip_liwc:
        run_liwc_comparison(style_d_records, human_texts, out_dir)

    if not args.skip_errors:
        a_pred_path = (
            config.RESULTS_DIR / "llm-alignment" / "style_a" / model_slug
            / "predictions.csv"
        )
        if style_a_records:
            run_error_dumps(style_a_records, a_pred_path, out_dir)
        else:
            print("\n=== Style A error dumps: SKIP — no Style A JSONL ===")

    if args.shap:
        run_shap(
            style_d_records, human_texts, model_slug,
            args.n_shap, args.top_k, out_dir,
        )

    print(f"\nDone. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
