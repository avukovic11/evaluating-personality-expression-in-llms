"""Linguistic analysis (RQ3): LIWC stats, TF-IDF vocab, Style A error dumps, optional SHAP.

Five analyses on the LLM-generated essays vs human texts, applied to either
dataset via `--dataset {essays, recruitview}`. Default run is fast (LIWC +
TF-IDF + keyword frequency + error dumps); SHAP is opt-in via `--shap` because it's slow.

1. **LIWC-style feature comparison** (default): for each trait T, compute
   per-essay LIWC features (reusing `liwc_features` from src.baselines) on
   three conditions:
     - humans-test
     - Style-B HIGH-on-T
     - Style-B LOW-on-T
   Report mean ± std per condition, Mann–Whitney U p-values, and Cliff's δ
   effect size for three pairwise comparisons: HIGH vs LOW, humans vs HIGH,
   humans vs LOW.

2. **TF-IDF vocabulary comparison** (default): for each trait T, fit a shared
   TfidfVectorizer (1-2 grams, min_df=3, max_features=5000) over the three
   conditions' combined corpus, then report the top-20 tokens by mean TF-IDF
   weight within each condition.

3. **Keyword frequency analysis** (default): for each trait T, count
   occurrences of TRAIT_KEYWORDS[T]['high'] and ['low'] word lists per essay
   using word-boundary regex, normalised to rate per 1000 tokens. Aggregates
   across three conditions; reports Mann–Whitney U + Cliff's δ for three
   pairwise comparisons per keyword pole (humans-vs-HIGH, humans-vs-LOW,
   HIGH-vs-LOW).

4. **Style A error dumps** (default): qualitative inspection of high-error
   essays, requires evaluate.py to have run Style A first.
     - Essays      → essays where intended ≠ predicted at threshold 0.5.
     - RecruitView → top |predicted_z − intended_z| residuals per trait.

5. **SHAP token attribution** (`--shap`; slow): for each trait T, sample N
   essays per condition (humans / B-HIGH / B-LOW), compute token-level SHAP
   over the probe's prediction for T, and aggregate top-K tokens by mean
   |SHAP value|. Output side-by-side per-trait CSV. Probe output is sigmoid
   prob for essays, raw z-score for recruitview.

Outputs go to:
    Essays      → code/datasets/results/llm-alignment/analysis/<model_slug>/
    RecruitView → code/datasets/results/llm-alignment/recruitview/analysis/<model_slug>/

    liwc_per_trait.csv               # long-form: trait × condition × feature
    liwc_stats_per_trait.csv         # MW p + Cliff's δ for each pairwise comp
    tfidf_per_trait.csv              # top-20 tokens per (trait, condition)
    keyword_freq_per_trait.csv       # mean rate per 1k tokens per (trait, condition, pole)
    keyword_stats_per_trait.csv      # MW p + Cliff's δ for keyword comparisons
    errors_per_trait/<trait>.txt     # qualitative misalignment dumps
    shap_<trait>.csv                 # only with --shap

Run from `code/`:
    python -m src.analyze                                      # essays, all fast analyses
    python -m src.analyze --shap                               # essays + SHAP (slow)
    python -m src.analyze --dataset recruitview                # recruitview, fast
    python -m src.analyze --dataset recruitview --shap         # recruitview + SHAP
    python -m src.analyze --skip-errors                        # LIWC + TF-IDF + keywords
    python -m src.analyze --skip-liwc --skip-errors            # TF-IDF + keywords only
    python -m src.analyze --skip-keyword-freq --skip-errors    # LIWC + TF-IDF only
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from . import config
from .baselines import _ensure_nltk, liwc_features
from .data import load_essays, load_splits


# ---------------------------------------------------------------------------
# Dataset specs (kept tiny — analyze.py only needs trait list + paths)
# ---------------------------------------------------------------------------

def _trait_cols(dataset: str) -> list[str]:
    return (
        config.TRAIT_COLS if dataset == "essays"
        else config.RECRUITVIEW_TRAIT_COLS
    )


def _trait_display(dataset: str, t: str) -> str:
    return (
        config.TRAIT_NAMES[t] if dataset == "essays"
        else config.RECRUITVIEW_TRAIT_NAMES[t]
    )


def _llm_outputs_dir(dataset: str) -> Path:
    return config.LLM_OUTPUTS_DIR if dataset == "essays" else config.LLM_OUTPUTS_RV_DIR


def _llm_align_root(dataset: str) -> Path:
    base = config.RESULTS_DIR / "llm-alignment"
    return base if dataset == "essays" else base / "recruitview"


def _style_jsonl_name(dataset: str, style: str) -> str:
    if dataset == "essays":
        return "style_b_single_trait.jsonl" if style == "B" else "style_a_paired.jsonl"
    return "style_b_recruitview.jsonl" if style == "B" else "style_a_recruitview.jsonl"


def _human_texts(dataset: str) -> list[str]:
    """Return the held-out human test texts for the chosen dataset."""
    if dataset == "essays":
        df = load_essays()
        splits = load_splits(df)
        return splits["test"]["TEXT"].tolist()
    # recruitview: load the full HF dump + user-level test split
    from .data_recruitview import load_recruitview, load_recruitview_splits
    df = load_recruitview()
    return load_recruitview_splits(df)["test"]["transcript"].tolist()


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
    style_b_records: list[dict], human_texts: list[str], out_dir: Path,
    dataset: str = "essays",
) -> None:
    print(f"\n=== LIWC feature comparison ({dataset}) ===")
    _ensure_nltk()

    trait_cols = _trait_cols(dataset)
    df_b = pd.DataFrame(style_b_records)
    print(f"Humans (test): {len(human_texts)} texts")
    feat_humans = _extract_features(human_texts, "LIWC humans")

    summary_rows: list[dict] = []
    stats_rows: list[dict] = []

    for trait in trait_cols:
        sub = df_b[df_b["prompted_trait"] == trait]
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
    for trait in trait_cols:
        sub = stats_df[
            (stats_df["trait"] == trait)
            & (stats_df["comparison"] == "high_vs_low")
        ].copy()
        if sub.empty:
            continue
        sub["abs_delta"] = sub["cliffs_delta"].abs()
        print(f"\n    {trait} ({_trait_display(dataset, trait)}):")
        for _, row in sub.nlargest(5, "abs_delta").iterrows():
            print(
                f"      {row['feature']:<25} d={row['cliffs_delta']:+.2f}  "
                f"p={row['mann_whitney_p']:.3f}"
            )


# ---------------------------------------------------------------------------
# TF-IDF vocabulary comparison
# ---------------------------------------------------------------------------

def run_tfidf_comparison(
    style_b_records: list[dict], human_texts: list[str], out_dir: Path,
    top_k: int = 20, dataset: str = "essays",
) -> None:
    print("\n=== TF-IDF vocabulary comparison ===")

    trait_cols = _trait_cols(dataset)
    df_d = pd.DataFrame(style_b_records)
    rows: list[dict] = []

    for trait in trait_cols:
        sub = df_d[df_d["prompted_trait"] == trait]
        high_texts = sub.loc[sub["prompted_level"] == "HIGH", "generated_text"].tolist()
        low_texts = sub.loc[sub["prompted_level"] == "LOW", "generated_text"].tolist()
        if not high_texts or not low_texts:
            print(f"  {trait}: skipped (HIGH={len(high_texts)}, LOW={len(low_texts)})")
            continue

        combined = human_texts + high_texts + low_texts
        vec = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=3,
            max_features=5000,
            sublinear_tf=True,
        )
        mat = vec.fit_transform(combined)
        vocab = vec.get_feature_names_out()

        n_h = len(human_texts)
        n_hi = len(high_texts)
        slices = {
            "humans": mat[:n_h],
            "high":   mat[n_h : n_h + n_hi],
            "low":    mat[n_h + n_hi :],
        }

        means = {c: np.asarray(m.mean(axis=0)).ravel() for c, m in slices.items()}
        disc = {
            "humans": means["humans"] - 0.5 * (means["high"] + means["low"]),
            "high":   means["high"]   - means["low"],
            "low":    means["low"]    - means["high"],
        }
        for cond_name in ("humans", "high", "low"):
            top_idx = np.argsort(disc[cond_name])[::-1][:top_k]
            for rank, idx in enumerate(top_idx, start=1):
                rows.append({
                    "trait":                trait,
                    "condition":            cond_name,
                    "token":                vocab[idx],
                    "mean_tfidf":           float(means[cond_name][idx]),
                    "discriminating_score": float(disc[cond_name][idx]),
                    "rank":                 rank,
                })

    out_dir.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(
        rows,
        columns=["trait", "condition", "token", "mean_tfidf", "discriminating_score", "rank"],
    )
    df_out.to_csv(out_dir / "tfidf_per_trait.csv", index=False)
    n_traits = df_out["trait"].nunique()
    print(f"  Saved tfidf_per_trait.csv  ({len(df_out)} rows, {n_traits} traits)")

    print("\n  Top-5 tokens per condition (HIGH vs LOW sample), by trait:")
    for trait in trait_cols:
        sub = df_out[df_out["trait"] == trait]
        if sub.empty:
            continue
        print(f"\n    {trait} ({_trait_display(dataset, trait)}):")
        for cond in ("humans", "high", "low"):
            tokens = sub[sub["condition"] == cond].head(5)["token"].tolist()
            print(f"      {cond:<8}: {', '.join(tokens) if tokens else '—'}")


# ---------------------------------------------------------------------------
# Keyword frequency analysis
# ---------------------------------------------------------------------------

def _compile_kw_pattern(keywords: list[str]) -> re.Pattern:
    alts = "|".join(re.escape(kw) for kw in keywords)
    return re.compile(r"\b(?:" + alts + r")\b", re.IGNORECASE)


def _keyword_rate(text: str, pattern: re.Pattern) -> float:
    """Keyword occurrences per 1000 whitespace tokens."""
    n_tokens = max(len(text.split()), 1)
    return 1000.0 * len(pattern.findall(text)) / n_tokens


# Maps RecruitView full-lowercase trait names to TRAIT_KEYWORDS keys (essays-style).
_RV_KW_KEY: dict[str, str] = {
    "openness":          "cOPN",
    "conscientiousness": "cCON",
    "extraversion":      "cEXT",
    "agreeableness":     "cAGR",
    "neuroticism":       "cNEU",
}


def run_keyword_frequency(
    style_b_records: list[dict], human_texts: list[str], out_dir: Path,
    dataset: str = "essays",
) -> None:
    print("\n=== Keyword frequency analysis ===")

    trait_cols = _trait_cols(dataset)
    df_d = pd.DataFrame(style_b_records)
    freq_rows: list[dict] = []
    stats_rows: list[dict] = []

    for trait in trait_cols:
        kw_key = _RV_KW_KEY.get(trait, trait) if dataset == "recruitview" else trait
        kw = config.TRAIT_KEYWORDS[kw_key]
        pat_high = _compile_kw_pattern(kw["high"])
        pat_low  = _compile_kw_pattern(kw["low"])

        sub = df_d[df_d["prompted_trait"] == trait]
        high_texts = sub.loc[sub["prompted_level"] == "HIGH", "generated_text"].tolist()
        low_texts  = sub.loc[sub["prompted_level"] == "LOW",  "generated_text"].tolist()
        if not high_texts or not low_texts:
            print(f"  {trait}: skipped (HIGH={len(high_texts)}, LOW={len(low_texts)})")
            continue

        groups: dict[str, list[str]] = {
            "humans": human_texts,
            "high":   high_texts,
            "low":    low_texts,
        }
        patterns: dict[str, re.Pattern] = {"high": pat_high, "low": pat_low}

        # Per-condition, per-pole rate arrays
        rates: dict[str, dict[str, np.ndarray]] = {}
        for cond_name, texts in groups.items():
            rates[cond_name] = {}
            for pole_name, pat in patterns.items():
                arr = np.array([_keyword_rate(t, pat) for t in texts])
                rates[cond_name][pole_name] = arr
                freq_rows.append({
                    "trait":          trait,
                    "condition":      cond_name,
                    "pole":           pole_name,
                    "n_essays":       int(len(arr)),
                    "mean_rate_per_1k": float(arr.mean()),
                    "std_rate_per_1k":  float(arr.std()),
                })

        # Three comparisons per pole: humans-vs-HIGH, humans-vs-LOW, HIGH-vs-LOW
        for pole_name, pat in patterns.items():
            h  = rates["humans"][pole_name]
            hi = rates["high"][pole_name]
            lo = rates["low"][pole_name]
            for a_name, a, b_name, b in [
                ("humans", h,  "high", hi),
                ("humans", h,  "low",  lo),
                ("high",   hi, "low",  lo),
            ]:
                try:
                    p = float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
                except ValueError:
                    p = float("nan")
                stats_rows.append({
                    "trait":          trait,
                    "pole":           pole_name,
                    "comparison":     f"{a_name}_vs_{b_name}",
                    "n_a":            int(len(a)),
                    "n_b":            int(len(b)),
                    "mann_whitney_p": p,
                    "cliffs_delta":   cliffs_delta(a, b),
                })

    out_dir.mkdir(parents=True, exist_ok=True)
    df_freq = pd.DataFrame(
        freq_rows,
        columns=["trait", "condition", "pole", "n_essays",
                 "mean_rate_per_1k", "std_rate_per_1k"],
    )
    df_freq.to_csv(out_dir / "keyword_freq_per_trait.csv", index=False)
    df_stats = pd.DataFrame(
        stats_rows,
        columns=["trait", "pole", "comparison", "n_a", "n_b",
                 "mann_whitney_p", "cliffs_delta"],
    )
    df_stats.to_csv(out_dir / "keyword_stats_per_trait.csv", index=False)
    print(
        f"  Saved keyword_freq_per_trait.csv  ({len(df_freq)} rows)  "
        f"keyword_stats_per_trait.csv  ({len(df_stats)} rows)"
    )

    print("\n  Mean HIGH-keyword rate per 1k tokens (humans / HIGH / LOW):")
    for trait in trait_cols:
        sub = df_freq[(df_freq["trait"] == trait) & (df_freq["pole"] == "high")]
        if sub.empty:
            continue
        vals = {r["condition"]: r["mean_rate_per_1k"] for _, r in sub.iterrows()}
        print(
            f"    {trait}: humans={vals.get('humans', float('nan')):.2f}  "
            f"HIGH={vals.get('high', float('nan')):.2f}  "
            f"LOW={vals.get('low', float('nan')):.2f}"
        )


# ---------------------------------------------------------------------------
# Style A error dumps
# ---------------------------------------------------------------------------

def run_error_dumps(
    style_a_records: list[dict], predictions_path: Path, out_dir: Path,
    per_trait_limit: int = 10,
) -> None:
    print("\n=== Style A error dumps (essays) ===")
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


def run_error_dumps_recruitview(
    style_a_records: list[dict], predictions_path: Path, out_dir: Path,
    per_trait_limit: int = 10,
) -> None:
    """For each trait, dump top-|residual| Style-A answers (predicted z vs intended z)."""
    print("\n=== Style A error dumps (recruitview, top |residual| per trait) ===")
    if not predictions_path.exists():
        print(f"  SKIP — predictions not at {predictions_path}.")
        print("  Run `python -m src.evaluate --dataset recruitview --style A` first.")
        return

    pred_df = pd.read_csv(predictions_path)
    text_by_id = {r["essay_id"]: r["generated_text"] for r in style_a_records}
    question_by_id = {
        r["essay_id"]: r.get("paired_question", "")
        for r in style_a_records
    }

    err_dir = out_dir / "errors_per_trait"
    err_dir.mkdir(parents=True, exist_ok=True)

    trait_cols = config.RECRUITVIEW_TRAIT_COLS
    for trait in trait_cols:
        intended_col, pred_col = f"intended_z_{trait}", f"pred_z_{trait}"
        if intended_col not in pred_df.columns:
            continue
        residuals = (pred_df[pred_col] - pred_df[intended_col]).abs()
        order = residuals.sort_values(ascending=False).index
        sample = pred_df.loc[order[:per_trait_limit]]
        print(
            f"  {trait}: dumping top-{len(sample)} of {len(pred_df)} "
            f"(max |residual| = {residuals.max():.3f})"
        )
        path = err_dir / f"{trait}.txt"
        with open(path, "w", encoding="utf-8") as f:
            for _, row in sample.iterrows():
                aid = row["essay_id"]
                intended_z = {
                    t: float(row[f"intended_z_{t}"]) for t in trait_cols
                }
                pred_z = {
                    t: float(row[f"pred_z_{t}"]) for t in trait_cols
                }
                intended_lvl = {
                    t: str(row[f"intended_level_{t}"]) for t in trait_cols
                }
                text = text_by_id.get(aid, "<<text not found>>")
                question = question_by_id.get(aid, "")
                f.write("=" * 70 + "\n")
                f.write(f"essay_id              : {aid}\n")
                f.write(
                    f"high-residual trait   : {trait}  "
                    f"(intended_z={row[intended_col]:+.2f}, "
                    f"pred_z={row[pred_col]:+.2f}, "
                    f"residual={row[pred_col] - row[intended_col]:+.2f})\n"
                )
                f.write(
                    "intended z-scores     : { "
                    + ", ".join(f"{t}={intended_z[t]:+.2f}" for t in trait_cols)
                    + " }\n"
                )
                f.write(
                    "intended levels       : { "
                    + ", ".join(f"{t}={intended_lvl[t]}" for t in trait_cols)
                    + " }\n"
                )
                f.write(
                    "predicted z-scores    : { "
                    + ", ".join(f"{t}={pred_z[t]:+.2f}" for t in trait_cols)
                    + " }\n"
                )
                if question:
                    f.write(f"interview question    : {question}\n")
                f.write("\n" + text + "\n\n")


# ---------------------------------------------------------------------------
# SHAP token attribution (optional)
# ---------------------------------------------------------------------------

def run_shap(
    style_b_records: list[dict], human_texts: list[str], model_slug: str,
    n_shap: int, top_k: int, out_dir: Path, dataset: str = "essays",
) -> None:
    print(f"\n=== SHAP token attribution ({dataset}, n_shap={n_shap} per condition) ===")
    print("    Slow: roughly 20–60 s per essay on CPU, 3–8 s on GPU.")

    import shap
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from .classifier import _result_dir_name, _sigmoid, get_device

    results_slug = _result_dir_name(model_slug, dataset)
    ckpt = config.CHECKPOINTS_DIR / f"{results_slug}_seed42"
    if not ckpt.exists():
        print(f"  SKIP — checkpoint not at {ckpt}.")
        return
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
    model = AutoModelForSequenceClassification.from_pretrained(str(ckpt))
    device = get_device()
    model.to(device).eval()

    trait_cols = _trait_cols(dataset)
    df_b = pd.DataFrame(style_b_records)
    masker = shap.maskers.Text(tokenizer)

    rng = np.random.default_rng(config.SEED)
    human_sample = list(rng.choice(
        np.asarray(human_texts, dtype=object),
        size=min(n_shap, len(human_texts)),
        replace=False,
    ))

    apply_sigmoid = (dataset == "essays")

    for trait_idx, trait in enumerate(trait_cols):
        sub = df_b[df_b["prompted_trait"] == trait]
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
            scores = _sigmoid(logits) if apply_sigmoid else logits
            return scores[:, trait_idx]

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
            for token, count, mean_abs, mean_signed in _aggregate_shap(sv, top_k=top_k):
                rows.append({
                    "trait": trait, "condition": cond_name,
                    "token": token, "count": count,
                    "mean_abs_shap": mean_abs, "mean_signed_shap": mean_signed,
                })

        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            rows,
            columns=["trait", "condition", "token", "count",
                     "mean_abs_shap", "mean_signed_shap"],
        ).to_csv(out_dir / f"shap_{trait}.csv", index=False)


def _aggregate_shap(
    sv, top_k: int, min_count: int = 3,
) -> list[tuple[str, int, float, float]]:
    """Aggregate token-level SHAP across documents; return top-K by mean |SHAP|.

    Returns (token, doc_count, mean_abs_shap, mean_signed_shap).
    Means are averaged over all occurrences across all documents.
    doc_count is the number of documents containing the token; tokens appearing
    in fewer than min_count documents are excluded before ranking so that
    single-document quirks don't dominate the top-K list.
    mean_signed_shap > 0 means the token pushes the probe toward HIGH-on-T;
    < 0 means it pushes toward LOW-on-T.
    """
    abs_sums:    dict[str, float] = {}
    signed_sums: dict[str, float] = {}
    occ_counts:  dict[str, int]   = {}  # total token occurrences (means denominator)
    doc_counts:  dict[str, int]   = {}  # documents containing the token (min_count filter)

    for i in range(len(sv)):
        tokens = sv.data[i]
        values = sv.values[i]
        seen_in_doc: set[str] = set()
        for tok, val in zip(tokens, values):
            # Strip subword markers (Ġ for RoBERTa, ▁ for SentencePiece).
            tok = tok.replace("Ġ", "").replace("▁", "").strip().lower()
            if not tok or len(tok) < 2 or not any(c.isalpha() for c in tok):
                continue
            fval = float(val)
            abs_sums[tok]    = abs_sums.get(tok, 0.0)    + abs(fval)
            signed_sums[tok] = signed_sums.get(tok, 0.0) + fval
            occ_counts[tok]  = occ_counts.get(tok, 0)    + 1
            seen_in_doc.add(tok)
        for tok in seen_in_doc:
            doc_counts[tok] = doc_counts.get(tok, 0) + 1

    results = [
        (
            tok,
            doc_counts[tok],
            abs_sums[tok] / occ_counts[tok],
            signed_sums[tok] / occ_counts[tok],
        )
        for tok in abs_sums
        if doc_counts.get(tok, 0) >= min_count
    ]
    results.sort(key=lambda x: -x[2])
    return results[:top_k]


# ---------------------------------------------------------------------------
# SHAP token attribution — humans-only variant
# ---------------------------------------------------------------------------

def run_shap_humans_only(
    df_test: pd.DataFrame,
    model_slug: str,
    n_shap: int,
    top_k: int,
    out_dir: Path,
) -> None:
    """SHAP attributions for human_high vs human_low per trait.

    Mirrors run_shap() but splits the human test set by ground-truth label
    instead of using LLM-generated essays. Saves shap_<trait>.csv to out_dir
    with the same schema: trait, condition, token, count, mean_abs_shap,
    mean_signed_shap.
    """
    print(f"\n=== SHAP token attribution — humans only (n_shap={n_shap} per condition) ===")
    print("    Slow: ~20–60 s per essay on CPU, 3–8 s on GPU.")

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

    masker = shap.maskers.Text(tokenizer)
    rng = np.random.default_rng(config.SEED)

    for trait_idx, trait in enumerate(config.TRAIT_COLS):
        hi_pool = df_test.loc[df_test[trait] == 1, "TEXT"].tolist()
        lo_pool = df_test.loc[df_test[trait] == 0, "TEXT"].tolist()
        hi_sample = list(rng.choice(
            np.asarray(hi_pool, dtype=object), size=min(n_shap, len(hi_pool)), replace=False,
        ))
        lo_sample = list(rng.choice(
            np.asarray(lo_pool, dtype=object), size=min(n_shap, len(lo_pool)), replace=False,
        ))

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
            ("human_high", hi_sample),
            ("human_low",  lo_sample),
        ]:
            print(f"  {trait} / {cond_name}: SHAP on {len(cond_texts)} essays")
            try:
                sv = explainer(cond_texts)
            except Exception as e:
                print(f"    skipped: {e}")
                continue
            for token, count, mean_abs, mean_signed in _aggregate_shap(sv, top_k=top_k):
                rows.append({
                    "trait": trait, "condition": cond_name,
                    "token": token, "count": count,
                    "mean_abs_shap": mean_abs, "mean_signed_shap": mean_signed,
                })

        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            rows,
            columns=["trait", "condition", "token", "count",
                     "mean_abs_shap", "mean_signed_shap"],
        ).to_csv(out_dir / f"shap_{trait}.csv", index=False)


# ---------------------------------------------------------------------------
# Humans-only analysis (ground-truth label split, no LLM data needed)
# ---------------------------------------------------------------------------

def plot_humans_only(sub_dir: Path, n_per_side: int = 8) -> None:
    """Generate LIWC-effects heatmap and TF-IDF diverging bar chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import seaborn as sns

    print("\n  Generating plots...")

    # ---- LIWC effects heatmap ----
    stats_path = sub_dir / "liwc_stats_per_trait.csv"
    if stats_path.exists():
        df_s = pd.read_csv(stats_path)
        col_order = [t for t in config.TRAIT_COLS if t in df_s["trait"].unique()]
        pivot_d = df_s.pivot(index="feature", columns="trait", values="cliffs_delta")[col_order]
        pivot_p = df_s.pivot(index="feature", columns="trait", values="mann_whitney_p")[col_order]

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(
            pivot_d, ax=ax, cmap="RdBu_r", center=0, vmin=-0.3, vmax=0.3,
            linewidths=0.5, annot=False,
            cbar_kws={"label": "Cliff's δ  (+ = human HIGH)"},
        )
        for i, feat in enumerate(pivot_d.index):
            for j, trait in enumerate(col_order):
                p = pivot_p.loc[feat, trait]
                if pd.notna(p) and p < 0.05:
                    ax.text(j + 0.5, i + 0.5, "*", ha="center", va="center",
                            fontsize=13, color="black", fontweight="bold")
        ax.set_xticklabels(
            [config.TRAIT_NAMES[t] for t in col_order], rotation=25, ha="right",
        )
        ax.set_ylabel("")
        ax.set_title("LIWC feature effect sizes: human HIGH vs human LOW  (* p < .05)")
        plt.tight_layout()
        out = sub_dir / "liwc_effects.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    Saved {out.name}")

    # ---- TF-IDF diverging bar chart ----
    tfidf_path = sub_dir / "tfidf_per_trait.csv"
    if tfidf_path.exists():
        df_t = pd.read_csv(tfidf_path)
        n_traits = len(config.TRAIT_COLS)
        fig, axes = plt.subplots(1, n_traits, figsize=(4 * n_traits, 7))

        for ax, trait in zip(axes, config.TRAIT_COLS):
            sub = df_t[df_t["trait"] == trait]
            hi = sub[sub["condition"] == "human_high"].head(n_per_side)
            lo = sub[sub["condition"] == "human_low"].head(n_per_side)

            tokens = hi["token"].tolist() + lo["token"].tolist()
            scores = (
                hi["discriminating_score"].tolist()
                + [-s for s in lo["discriminating_score"].tolist()]
            )
            if not tokens:
                continue
            pairs = sorted(zip(scores, tokens))
            scores_s, tokens_s = zip(*pairs)

            colors = ["#4575b4" if s < 0 else "#d73027" for s in scores_s]
            y_pos = range(len(tokens_s))
            ax.barh(y_pos, scores_s, color=colors, height=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(tokens_s, fontsize=7.5)
            ax.axvline(0, color="black", linewidth=0.6)
            ax.set_title(config.TRAIT_NAMES[trait], fontsize=10, fontweight="bold")
            ax.set_xlabel("Discriminating score", fontsize=8)
            ax.tick_params(axis="x", labelsize=7)

        hi_patch = mpatches.Patch(color="#d73027", label="human HIGH")
        lo_patch = mpatches.Patch(color="#4575b4", label="human LOW")
        fig.legend(handles=[hi_patch, lo_patch], loc="lower center",
                   ncol=2, bbox_to_anchor=(0.5, -0.02), fontsize=9)
        fig.suptitle(
            "TF-IDF discriminating tokens: human HIGH vs human LOW  "
            "(ranked by discriminating score)",
            fontsize=11, fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0.04, 1, 1])
        out = sub_dir / "tfidf_diverging.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    Saved {out.name}")


def run_humans_only_analysis(
    df_test: pd.DataFrame,
    out_dir: Path,
    model_slug: str = "roberta-base",
    skip_liwc: bool = False,
    skip_tfidf: bool = False,
    skip_keyword_freq: bool = False,
    run_shap: bool = False,
    n_shap: int = 30,
    top_k: int = 20,
    plot: bool = False,
) -> None:
    """Compare human test essays split by ground-truth trait label.

    For each trait T, groups test essays into human_high (y=1) and human_low
    (y=0) and runs LIWC, TF-IDF, and keyword-frequency two-way comparisons.
    No LLM data or model checkpoint required.
    Outputs go to `out_dir/humans_only/`.
    """
    print("\n=== Humans-only analysis (ground-truth label split) ===")
    sub_dir = out_dir / "humans_only"
    sub_dir.mkdir(parents=True, exist_ok=True)

    def _split(trait: str) -> tuple[list[str], list[str]]:
        hi = df_test.loc[df_test[trait] == 1, "TEXT"].tolist()
        lo = df_test.loc[df_test[trait] == 0, "TEXT"].tolist()
        return hi, lo

    # ---- LIWC ----
    if not skip_liwc:
        print("\n  LIWC (human_high vs human_low):")
        _ensure_nltk()
        sum_rows: list[dict] = []
        stat_rows: list[dict] = []
        for trait in config.TRAIT_COLS:
            hi_texts, lo_texts = _split(trait)
            if not hi_texts or not lo_texts:
                continue
            feat_hi = _extract_features(hi_texts, f"LIWC {trait}-high")
            feat_lo = _extract_features(lo_texts, f"LIWC {trait}-low")
            for feat in feat_hi.columns:
                h = feat_hi[feat].to_numpy()
                lo = feat_lo[feat].to_numpy()
                for cond, arr in [("human_high", h), ("human_low", lo)]:
                    sum_rows.append({
                        "trait": trait, "feature": feat, "condition": cond,
                        "n": int(len(arr)),
                        "mean": float(arr.mean()) if len(arr) else float("nan"),
                        "std":  float(arr.std())  if len(arr) else float("nan"),
                    })
                try:
                    p = float(mannwhitneyu(h, lo, alternative="two-sided").pvalue)
                except ValueError:
                    p = float("nan")
                stat_rows.append({
                    "trait": trait, "feature": feat,
                    "comparison": "human_high_vs_human_low",
                    "n_a": int(len(h)), "n_b": int(len(lo)),
                    "mann_whitney_p": p,
                    "cliffs_delta": cliffs_delta(h, lo),
                })
        pd.DataFrame(sum_rows).to_csv(sub_dir / "liwc_per_trait.csv", index=False)
        pd.DataFrame(stat_rows).to_csv(sub_dir / "liwc_stats_per_trait.csv", index=False)
        print(f"    Saved liwc_per_trait.csv + liwc_stats_per_trait.csv")
        print("\n    Top |cliffs_delta| (human_high vs human_low), per trait:")
        stats_df = pd.DataFrame(stat_rows)
        for trait in config.TRAIT_COLS:
            sub = stats_df[stats_df["trait"] == trait].copy()
            if sub.empty:
                continue
            sub["abs_delta"] = sub["cliffs_delta"].abs()
            print(f"\n      {trait} ({config.TRAIT_NAMES[trait]}):")
            for _, row in sub.nlargest(5, "abs_delta").iterrows():
                print(
                    f"        {row['feature']:<25} d={row['cliffs_delta']:+.3f}  "
                    f"p={row['mann_whitney_p']:.3f}"
                )

    # ---- TF-IDF ----
    if not skip_tfidf:
        print("\n  TF-IDF (human_high vs human_low):")
        rows: list[dict] = []
        for trait in config.TRAIT_COLS:
            hi_texts, lo_texts = _split(trait)
            if not hi_texts or not lo_texts:
                continue
            combined = hi_texts + lo_texts
            vec = TfidfVectorizer(
                ngram_range=(1, 2), min_df=3, max_features=5000, sublinear_tf=True,
            )
            mat = vec.fit_transform(combined)
            vocab = vec.get_feature_names_out()
            n_hi = len(hi_texts)
            m_hi = np.asarray(mat[:n_hi].mean(axis=0)).ravel()
            m_lo = np.asarray(mat[n_hi:].mean(axis=0)).ravel()
            for cond_name, mean_arr, disc_arr in [
                ("human_high", m_hi, m_hi - m_lo),
                ("human_low",  m_lo, m_lo - m_hi),
            ]:
                top_idx = np.argsort(disc_arr)[::-1][:20]
                for rank, idx in enumerate(top_idx, start=1):
                    rows.append({
                        "trait":                trait,
                        "condition":            cond_name,
                        "token":                vocab[idx],
                        "mean_tfidf":           float(mean_arr[idx]),
                        "discriminating_score": float(disc_arr[idx]),
                        "rank":                 rank,
                    })
        df_tfidf = pd.DataFrame(
            rows,
            columns=["trait", "condition", "token", "mean_tfidf",
                     "discriminating_score", "rank"],
        )
        df_tfidf.to_csv(sub_dir / "tfidf_per_trait.csv", index=False)
        print(f"    Saved tfidf_per_trait.csv ({len(df_tfidf)} rows)")
        print("\n    Top-5 tokens per condition, by trait:")
        for trait in config.TRAIT_COLS:
            sub = df_tfidf[df_tfidf["trait"] == trait]
            if sub.empty:
                continue
            print(f"\n      {trait} ({config.TRAIT_NAMES[trait]}):")
            for cond in ("human_high", "human_low"):
                tokens = sub[sub["condition"] == cond].head(5)["token"].tolist()
                print(f"        {cond}: {', '.join(tokens) if tokens else '—'}")

    # ---- Keyword frequency ----
    if not skip_keyword_freq:
        print("\n  Keyword frequency (human_high vs human_low):")
        freq_rows: list[dict] = []
        stat_rows2: list[dict] = []
        for trait in config.TRAIT_COLS:
            hi_texts, lo_texts = _split(trait)
            if not hi_texts or not lo_texts:
                continue
            kw = config.TRAIT_KEYWORDS[trait]
            for pole_name in ("high", "low"):
                pat = _compile_kw_pattern(kw[pole_name])
                hi_arr = np.array([_keyword_rate(t, pat) for t in hi_texts])
                lo_arr = np.array([_keyword_rate(t, pat) for t in lo_texts])
                for cond, arr in [("human_high", hi_arr), ("human_low", lo_arr)]:
                    freq_rows.append({
                        "trait":            trait,
                        "condition":        cond,
                        "pole":             pole_name,
                        "n_essays":         int(len(arr)),
                        "mean_rate_per_1k": float(arr.mean()),
                        "std_rate_per_1k":  float(arr.std()),
                    })
                try:
                    p = float(mannwhitneyu(hi_arr, lo_arr, alternative="two-sided").pvalue)
                except ValueError:
                    p = float("nan")
                stat_rows2.append({
                    "trait":          trait,
                    "pole":           pole_name,
                    "comparison":     "human_high_vs_human_low",
                    "n_a":            int(len(hi_arr)),
                    "n_b":            int(len(lo_arr)),
                    "mann_whitney_p": p,
                    "cliffs_delta":   cliffs_delta(hi_arr, lo_arr),
                })
        pd.DataFrame(
            freq_rows,
            columns=["trait", "condition", "pole", "n_essays",
                     "mean_rate_per_1k", "std_rate_per_1k"],
        ).to_csv(sub_dir / "keyword_freq_per_trait.csv", index=False)
        pd.DataFrame(
            stat_rows2,
            columns=["trait", "pole", "comparison", "n_a", "n_b",
                     "mann_whitney_p", "cliffs_delta"],
        ).to_csv(sub_dir / "keyword_stats_per_trait.csv", index=False)
        print("    Saved keyword_freq_per_trait.csv + keyword_stats_per_trait.csv")
        print("\n    Mean HIGH-keyword rate per 1k tokens (human_high / human_low):")
        df_freq = pd.DataFrame(freq_rows)
        for trait in config.TRAIT_COLS:
            sub = df_freq[(df_freq["trait"] == trait) & (df_freq["pole"] == "high")]
            if sub.empty:
                continue
            vals = {r["condition"]: r["mean_rate_per_1k"] for _, r in sub.iterrows()}
            print(
                f"      {trait}: HIGH={vals.get('human_high', float('nan')):.2f}  "
                f"LOW={vals.get('human_low', float('nan')):.2f}"
            )

    if run_shap:
        run_shap_humans_only(df_test, model_slug, n_shap, top_k, sub_dir)

    if plot:
        plot_humans_only(sub_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dataset", choices=["essays", "recruitview"], default="essays",
        help="Which dataset's LLM essays and probe to analyze.",
    )
    parser.add_argument(
        "--model", type=str, default=config.CLASSIFIER_MODEL,
        help=f"Probe model slug (also names the output dir). Default {config.CLASSIFIER_MODEL}.",
    )
    parser.add_argument("--skip-liwc", action="store_true",
                       help="Skip the LIWC comparison.")
    parser.add_argument("--skip-tfidf", action="store_true",
                       help="Skip the TF-IDF vocabulary comparison.")
    parser.add_argument("--skip-keyword-freq", action="store_true",
                       help="Skip the keyword frequency analysis.")
    parser.add_argument("--humans-only", action="store_true",
                       help="Run LIWC/TF-IDF/keyword analyses on human test essays "
                            "split by ground-truth label. No LLM data needed. "
                            "Outputs go to <out_dir>/humans_only/.")
    parser.add_argument("--plot", action="store_true",
                       help="Generate PNG plots after analysis (use with --humans-only).")
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
    out_dir = config.RESULTS_DIR / "llm-alignment" / "analysis" / model_slug
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.humans_only:
        print(f"output : {out_dir / 'humans_only'}")
        df = load_essays()
        splits = load_splits(df)
        run_humans_only_analysis(
            splits["test"], out_dir,
            model_slug=model_slug,
            skip_liwc=args.skip_liwc,
            skip_tfidf=args.skip_tfidf,
            skip_keyword_freq=args.skip_keyword_freq,
            run_shap=args.shap,
            n_shap=args.n_shap,
            top_k=args.top_k,
            plot=args.plot,
        )
        print(f"\nDone. Outputs in {out_dir / 'humans_only'}")
        return

    llm_dir = _llm_outputs_dir(args.dataset)
    style_b_path = llm_dir / _style_jsonl_name(args.dataset, "B")
    style_b_records = load_jsonl(style_b_path)
    print(f"dataset : {args.dataset}")
    print(f"Style B : {len(style_b_records)} essays from {style_b_path}")

    style_a_path = llm_dir / _style_jsonl_name(args.dataset, "A")
    style_a_records = load_jsonl(style_a_path) if style_a_path.exists() else []
    if style_a_records:
        print(f"Style A : {len(style_a_records)} essays from {style_a_path}")
    else:
        print(f"Style A : none ({style_a_path} not found; error dumps will skip)")

    human_texts = _human_texts(args.dataset)

    out_dir = _llm_align_root(args.dataset) / "analysis" / model_slug
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"output  : {out_dir}")

    if not args.skip_liwc:
        run_liwc_comparison(
            style_b_records, human_texts, out_dir, dataset=args.dataset,
        )

    if not args.skip_tfidf:
        run_tfidf_comparison(style_b_records, human_texts, out_dir, dataset=args.dataset)

    if not args.skip_keyword_freq:
        run_keyword_frequency(style_b_records, human_texts, out_dir, dataset=args.dataset)

    if not args.skip_errors:
        a_pred_path = (
            _llm_align_root(args.dataset) / "style_a" / model_slug
            / "predictions.csv"
        )
        if style_a_records:
            if args.dataset == "essays":
                run_error_dumps(style_a_records, a_pred_path, out_dir)
            else:
                run_error_dumps_recruitview(
                    style_a_records, a_pred_path, out_dir,
                )
        else:
            print("\n=== Style A error dumps: SKIP — no Style A JSONL ===")

    if args.shap:
        run_shap(
            style_b_records, human_texts, model_slug,
            args.n_shap, args.top_k, out_dir, dataset=args.dataset,
        )

    print(f"\nDone. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
