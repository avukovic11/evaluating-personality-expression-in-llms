"""Lower-bound baselines on Track 1 (Pennebaker Essays) and Track 2 (RECRUITVIEW).

Track 1 — Pennebaker Essays:
  Multi-label binary classification over 5 Big Five traits. DummyClassifier(most_frequent),
  TF-IDF + LogisticRegression, LIWC-style + LogisticRegression.

Track 2 — RECRUITVIEW:
  Multi-target regression over 5 OCEAN z-scored traits. Mean predictor,
  TF-IDF + Ridge, LIWC-style + Ridge.

Both tracks share the LIWC feature extractor, the TF-IDF vectorizer, and the
per-trait + macro reporting/saving scaffold. The metric set differs by task:
  - classification → accuracy, F1, ROC-AUC
  - regression     → Spearman ρ (primary), Pearson r, MAE, R²

Hyperparameters are tuned on val (never test). Final test metrics are computed
after refitting on train+val.

Per-baseline outputs go to `code/datasets/results/<name>[_recruitview]/`:
    metrics.json           — per-trait + macro
    test_predictions.csv   — id (+ user_no for RV), true/pred per trait, prob if classification
    feature_names.txt      — LIWC only

Run from `code/`:
    python -m src.baselines                            # essays default; all 3
    python -m src.baselines --model tfidf
    python -m src.baselines --dataset recruitview      # all 3 regression baselines
    python -m src.baselines --dataset recruitview --model liwc
"""

from __future__ import annotations

import argparse
import json
import re

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from . import config


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def _safe_corr(fn, a: np.ndarray, b: np.ndarray) -> float:
    """Run a scipy correlation; return 0.0 on NaN or all-constant inputs."""
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    try:
        r = float(fn(a, b)[0])
    except Exception:
        return 0.0
    return 0.0 if np.isnan(r) else r


def metrics_per_trait(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    task: str = "classification",
    trait_cols: list[str] | None = None,
) -> dict:
    """Return per-trait metrics + macro averages.

    For backward compatibility, defaults are essays (binary classification).
    For RECRUITVIEW pass `task="regression"` and `trait_cols=config.RECRUITVIEW_TRAIT_COLS`.
    """
    if trait_cols is None:
        trait_cols = config.TRAIT_COLS

    out: dict = {}
    if task == "classification":
        for i, trait in enumerate(trait_cols):
            d = {
                "accuracy": float(accuracy_score(y_true[:, i], y_pred[:, i])),
                "f1": float(f1_score(y_true[:, i], y_pred[:, i], zero_division=0)),
            }
            if y_prob is not None:
                try:
                    d["roc_auc"] = float(roc_auc_score(y_true[:, i], y_prob[:, i]))
                except ValueError:
                    d["roc_auc"] = float("nan")
            out[trait] = d
        macro_keys = ["accuracy", "f1"]
        if y_prob is not None:
            macro_keys.append("roc_auc")
    else:  # regression
        for i, trait in enumerate(trait_cols):
            yt, yp = y_true[:, i], y_pred[:, i]
            out[trait] = {
                "spearman": _safe_corr(spearmanr, yt, yp),
                "pearson":  _safe_corr(pearsonr, yt, yp),
                "mae":      float(mean_absolute_error(yt, yp)),
                "r2":       float(r2_score(yt, yp)),
            }
        macro_keys = ["spearman", "pearson", "mae", "r2"]

    out["macro"] = {
        k: float(np.nanmean([out[t][k] for t in trait_cols]))
        for k in macro_keys
    }
    return out


def print_metrics(name: str, m: dict) -> None:
    """Print per-trait + macro metrics. Auto-detects task from keys."""
    sample = m[next(t for t in m if t != "macro")]
    is_classification = "accuracy" in sample
    print(f"  {name}")
    if is_classification:
        has_auc = "roc_auc" in sample
        header = f"  {'trait':<8} {'acc':>6} {'f1':>6}" + (f" {'auc':>6}" if has_auc else "")
        print(header)
        for trait, d in m.items():
            if trait == "macro":
                continue
            line = f"  {trait:<8} {d['accuracy']:>6.3f} {d['f1']:>6.3f}"
            if has_auc:
                line += f" {d['roc_auc']:>6.3f}"
            print(line)
        d = m["macro"]
        line = f"  {'macro':<8} {d['accuracy']:>6.3f} {d['f1']:>6.3f}"
        if has_auc:
            line += f" {d['roc_auc']:>6.3f}"
        print(line)
    else:
        header = f"  {'trait':<18} {'rho':>6} {'r':>6} {'mae':>6} {'r2':>7}"
        print(header)
        for trait, d in m.items():
            if trait == "macro":
                continue
            print(
                f"  {trait:<18} {d['spearman']:>6.3f} {d['pearson']:>6.3f} "
                f"{d['mae']:>6.3f} {d['r2']:>7.3f}"
            )
        d = m["macro"]
        print(
            f"  {'macro':<18} {d['spearman']:>6.3f} {d['pearson']:>6.3f} "
            f"{d['mae']:>6.3f} {d['r2']:>7.3f}"
        )


# -----------------------------------------------------------------------------
# Dummy
# -----------------------------------------------------------------------------

def fit_dummy(
    train_y: np.ndarray, test_y: np.ndarray, task: str = "classification",
) -> tuple[np.ndarray, np.ndarray | None]:
    """Per-trait sanity-floor predictor.

    Classification: DummyClassifier(most_frequent) → (pred, prob).
    Regression: per-trait mean of train_y → (pred, None).
    """
    n_traits = train_y.shape[1]
    if task == "classification":
        pred = np.zeros_like(test_y)
        prob = np.zeros_like(test_y, dtype=float)
        dx_tr = np.zeros((len(train_y), 1))
        dx_te = np.zeros((len(test_y), 1))
        for i in range(n_traits):
            m = DummyClassifier(strategy="most_frequent", random_state=config.SEED)
            m.fit(dx_tr, train_y[:, i])
            pred[:, i] = m.predict(dx_te)
            p = m.predict_proba(dx_te)
            prob[:, i] = p[:, 1] if p.shape[1] == 2 else float(train_y[:, i].mean())
        return pred, prob
    else:
        pred = np.empty_like(test_y, dtype=float)
        for i in range(n_traits):
            pred[:, i] = float(train_y[:, i].mean())
        return pred, None


# -----------------------------------------------------------------------------
# TF-IDF + (Logistic Regression | Ridge)
# -----------------------------------------------------------------------------

_TFIDF_LR_GRID = [
    {"ngram_range": (1, 1), "C": 0.5},
    {"ngram_range": (1, 1), "C": 1.0},
    {"ngram_range": (1, 1), "C": 2.0},
    {"ngram_range": (1, 2), "C": 0.5},
    {"ngram_range": (1, 2), "C": 1.0},
    {"ngram_range": (1, 2), "C": 2.0},
]
_TFIDF_RIDGE_GRID = [
    {"ngram_range": (1, 1), "alpha": 0.1},
    {"ngram_range": (1, 1), "alpha": 1.0},
    {"ngram_range": (1, 1), "alpha": 10.0},
    {"ngram_range": (1, 1), "alpha": 100.0},
    {"ngram_range": (1, 2), "alpha": 0.1},
    {"ngram_range": (1, 2), "alpha": 1.0},
    {"ngram_range": (1, 2), "alpha": 10.0},
    {"ngram_range": (1, 2), "alpha": 100.0},
]


def _vec(ngram_range):
    return TfidfVectorizer(
        ngram_range=ngram_range, min_df=5, max_df=0.95,
        sublinear_tf=True, lowercase=True,
    )


def fit_tfidf(
    train_text, train_y, val_text, val_y, test_text, test_y,
    task: str = "classification",
) -> tuple[np.ndarray, np.ndarray | None]:
    """TF-IDF features + LogReg (classification) or Ridge (regression).

    Tunes (n-gram range, C/alpha) on val by macro accuracy (classif) or macro
    Spearman ρ (regression); refits on train+val; predicts test.
    """
    is_clf = task == "classification"
    grid = _TFIDF_LR_GRID if is_clf else _TFIDF_RIDGE_GRID
    score_label = "macro-acc" if is_clf else "macro-spearman"

    best_score, best_g = -np.inf, None
    for g in grid:
        vec = _vec(g["ngram_range"])
        X_train = vec.fit_transform(train_text)
        X_val = vec.transform(val_text)
        scores = []
        for i in range(train_y.shape[1]):
            if is_clf:
                clf = LogisticRegression(C=g["C"], max_iter=2000, random_state=config.SEED)
                clf.fit(X_train, train_y[:, i])
                scores.append(accuracy_score(val_y[:, i], clf.predict(X_val)))
            else:
                clf = Ridge(alpha=g["alpha"], random_state=config.SEED)
                clf.fit(X_train, train_y[:, i])
                scores.append(_safe_corr(spearmanr, val_y[:, i], clf.predict(X_val)))
        avg = float(np.mean(scores))
        hp_str = f"C={g['C']:>4}" if is_clf else f"alpha={g['alpha']:>6}"
        print(f"  ngram={g['ngram_range']} {hp_str}  val {score_label}={avg:.3f}")
        if avg > best_score:
            best_score, best_g = avg, g
    print(f"  best: ngram={best_g['ngram_range']} "
          f"{'C='+str(best_g['C']) if is_clf else 'alpha='+str(best_g['alpha'])} "
          f"(val={best_score:.3f})")

    # Refit on train+val, predict test.
    vec = _vec(best_g["ngram_range"])
    X_all = vec.fit_transform(list(train_text) + list(val_text))
    X_test = vec.transform(test_text)
    all_y = np.concatenate([train_y, val_y], axis=0)

    n_traits = train_y.shape[1]
    pred = np.empty_like(test_y, dtype=float)
    prob = np.empty_like(test_y, dtype=float) if is_clf else None
    for i in range(n_traits):
        if is_clf:
            clf = LogisticRegression(C=best_g["C"], max_iter=2000, random_state=config.SEED)
            clf.fit(X_all, all_y[:, i])
            pred[:, i] = clf.predict(X_test)
            prob[:, i] = clf.predict_proba(X_test)[:, 1]
        else:
            clf = Ridge(alpha=best_g["alpha"], random_state=config.SEED)
            clf.fit(X_all, all_y[:, i])
            pred[:, i] = clf.predict(X_test)
    return pred, prob


# -----------------------------------------------------------------------------
# LIWC-style hand-crafted features
# -----------------------------------------------------------------------------

PRONOUNS_I = {"i", "me", "my", "mine", "myself", "im", "ive", "id", "ill"}
PRONOUNS_WE = {"we", "us", "our", "ours", "ourselves"}
PRONOUNS_YOU = {"you", "your", "yours", "yourself", "yourselves"}
PRONOUNS_3P = {
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "they", "them", "their", "theirs", "themselves", "it", "its", "itself",
}
NEGATIONS = {
    "no", "not", "nor", "never", "nothing", "none", "nobody", "neither",
    "nowhere", "cant", "wont", "dont", "shouldnt", "wouldnt", "couldnt",
    "didnt", "doesnt", "isnt", "arent", "wasnt", "werent",
}
NRC_KEYS = (
    "anger", "anticipation", "disgust", "fear", "joy",
    "sadness", "surprise", "trust", "positive", "negative",
)
WORD_RE = re.compile(r"[a-z']+")
SENT_RE = re.compile(r"[.!?]+\s+|\n+")


def _ensure_nltk() -> None:
    import nltk
    for pkg, path in [("punkt", "tokenizers/punkt"), ("punkt_tab", "tokenizers/punkt_tab")]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


def liwc_features(text: str) -> dict[str, float]:
    """Extract LIWC-style features for a single document."""
    from nrclex import NRCLex
    text_lower = text.lower()
    tokens = WORD_RE.findall(text_lower)
    n_words = max(len(tokens), 1)
    n_sent = max(len([s for s in SENT_RE.split(text) if s.strip()]), 1)

    feats: dict[str, float] = {
        "word_count_log":   float(np.log1p(n_words)),
        "avg_sentence_len": n_words / n_sent,
        "avg_word_len":     float(np.mean([len(t) for t in tokens])) if tokens else 0.0,
        "ttr":              len(set(tokens)) / n_words,
        "pron_i":           sum(t in PRONOUNS_I for t in tokens) / n_words,
        "pron_we":          sum(t in PRONOUNS_WE for t in tokens) / n_words,
        "pron_you":         sum(t in PRONOUNS_YOU for t in tokens) / n_words,
        "pron_3p":          sum(t in PRONOUNS_3P for t in tokens) / n_words,
        "negation":         sum(t in NEGATIONS for t in tokens) / n_words,
        "q_rate":           text.count("?") / n_sent,
        "excl_rate":        text.count("!") / n_sent,
    }
    try:
        freqs = NRCLex(text).affect_frequencies
    except Exception:
        freqs = {}
    for k in NRC_KEYS:
        feats[f"emo_{k}"] = float(freqs.get(k, 0.0))
    return feats


def extract_liwc_matrix(texts: list[str], desc: str) -> tuple[np.ndarray, list[str]]:
    rows = [liwc_features(t) for t in tqdm(texts, desc=desc, leave=False)]
    df = pd.DataFrame(rows)
    return df.to_numpy(dtype=float), list(df.columns)


_LIWC_LR_GRID = [0.1, 0.5, 1.0, 2.0, 5.0]
_LIWC_RIDGE_GRID = [0.1, 1.0, 10.0, 100.0]


def fit_liwc(
    train_text, train_y, val_text, val_y, test_text, test_y,
    task: str = "classification",
) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
    """LIWC-style features + LogReg (classification) or Ridge (regression)."""
    is_clf = task == "classification"
    grid = _LIWC_LR_GRID if is_clf else _LIWC_RIDGE_GRID
    score_label = "macro-acc" if is_clf else "macro-spearman"

    _ensure_nltk()
    X_train, feat_names = extract_liwc_matrix(train_text, "LIWC train")
    X_val, _ = extract_liwc_matrix(val_text, "LIWC val")
    X_test, _ = extract_liwc_matrix(test_text, "LIWC test")

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)

    best_score, best_hp = -np.inf, None
    for hp in grid:
        scores = []
        for i in range(train_y.shape[1]):
            if is_clf:
                clf = LogisticRegression(C=hp, max_iter=2000, random_state=config.SEED)
                clf.fit(X_train_s, train_y[:, i])
                scores.append(accuracy_score(val_y[:, i], clf.predict(X_val_s)))
            else:
                clf = Ridge(alpha=hp, random_state=config.SEED)
                clf.fit(X_train_s, train_y[:, i])
                scores.append(_safe_corr(spearmanr, val_y[:, i], clf.predict(X_val_s)))
        avg = float(np.mean(scores))
        hp_name = "C" if is_clf else "alpha"
        print(f"  {hp_name}={hp:>4}  val {score_label}={avg:.3f}")
        if avg > best_score:
            best_score, best_hp = avg, hp
    print(f"  best: {'C' if is_clf else 'alpha'}={best_hp} (val={best_score:.3f})")

    # Refit on train+val with a fresh scaler.
    X_all = np.concatenate([X_train, X_val], axis=0)
    all_y = np.concatenate([train_y, val_y], axis=0)
    scaler2 = StandardScaler().fit(X_all)
    X_all_s = scaler2.transform(X_all)
    X_test_s = scaler2.transform(X_test)

    n_traits = train_y.shape[1]
    pred = np.empty_like(test_y, dtype=float)
    prob = np.empty_like(test_y, dtype=float) if is_clf else None
    for i in range(n_traits):
        if is_clf:
            clf = LogisticRegression(C=best_hp, max_iter=2000, random_state=config.SEED)
            clf.fit(X_all_s, all_y[:, i])
            pred[:, i] = clf.predict(X_test_s)
            prob[:, i] = clf.predict_proba(X_test_s)[:, 1]
        else:
            clf = Ridge(alpha=best_hp, random_state=config.SEED)
            clf.fit(X_all_s, all_y[:, i])
            pred[:, i] = clf.predict(X_test_s)
    return pred, prob, feat_names


# -----------------------------------------------------------------------------
# I/O
# -----------------------------------------------------------------------------

def save_results(
    name: str,
    pred: np.ndarray,
    prob: np.ndarray | None,
    y_test: np.ndarray,
    test_ids: list[str],
    metrics: dict,
    feat_names: list[str] | None = None,
    trait_cols: list[str] | None = None,
    id_col_name: str = "AUTHID",
    extra_cols: dict[str, list] | None = None,
) -> None:
    """Persist predictions + metrics; backward-compatible defaults are essays."""
    if trait_cols is None:
        trait_cols = config.TRAIT_COLS
    out_dir = config.RESULTS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    cols: dict[str, list] = {id_col_name: test_ids}
    if extra_cols:
        for k, v in extra_cols.items():
            if k not in cols:
                cols[k] = v
    for i, t in enumerate(trait_cols):
        cols[f"true_{t}"] = y_test[:, i].tolist()
        cols[f"pred_{t}"] = pred[:, i].tolist()
        if prob is not None:
            cols[f"prob_{t}"] = prob[:, i].tolist()
    pd.DataFrame(cols).to_csv(out_dir / "test_predictions.csv", index=False)
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    if feat_names is not None:
        (out_dir / "feature_names.txt").write_text(
            "\n".join(feat_names) + "\n", encoding="utf-8"
        )


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dataset", choices=["essays", "recruitview"], default="essays",
        help="Which dataset to run baselines on.",
    )
    parser.add_argument(
        "--model", choices=["all", "dummy", "tfidf", "liwc"], default="all",
    )
    args = parser.parse_args()

    config.ensure_dirs()

    if args.dataset == "essays":
        from .data import load_essays, load_splits
        df = load_essays()
        splits = load_splits(df)
        trait_cols = config.TRAIT_COLS
        task = "classification"
        text_col = "TEXT"
        id_col = "AUTHID"
        suffix = ""
        # Name suffix mapping for output dirs.
        name_map = {"dummy": "dummy", "tfidf": "tfidf-lr", "liwc": "liwc-lr"}
    else:
        from .data_recruitview import load_recruitview, load_recruitview_splits
        df = load_recruitview()
        splits = load_recruitview_splits(df)
        trait_cols = config.RECRUITVIEW_TRAIT_COLS
        task = "regression"
        text_col = "transcript"
        id_col = "id"
        suffix = "_recruitview"
        name_map = {"dummy": "dummy", "tfidf": "tfidf-ridge", "liwc": "liwc-ridge"}

    print(f"dataset : {args.dataset} ({task}, {len(trait_cols)} traits)")
    print(f"train/val/test sizes: "
          f"{len(splits['train'])} / {len(splits['val'])} / {len(splits['test'])}")

    y = {k: splits[k][trait_cols].to_numpy(dtype=float) for k in splits}
    txt = {k: splits[k][text_col].tolist() for k in splits}
    test_ids = splits["test"][id_col].astype(str).tolist()

    extra_test: dict[str, list] = {}
    if args.dataset == "recruitview":
        extra_test["user_no"] = splits["test"]["user_no"].astype(str).tolist()
        extra_test["question_id"] = splits["test"]["question_id"].astype(str).tolist()

    def _save(short_name: str, pred, prob, m, feat_names=None):
        save_results(
            name_map[short_name] + suffix, pred, prob, y["test"], test_ids, m,
            feat_names=feat_names, trait_cols=trait_cols,
            id_col_name=id_col, extra_cols=extra_test,
        )

    if args.model in ("all", "dummy"):
        print(f"\n[dummy{suffix}] {task} sanity floor")
        pred, prob = fit_dummy(y["train"], y["test"], task=task)
        m = metrics_per_trait(y["test"], pred, prob, task=task, trait_cols=trait_cols)
        print_metrics(name_map["dummy"] + suffix, m)
        _save("dummy", pred, prob, m)

    if args.model in ("all", "tfidf"):
        print(f"\n[{name_map['tfidf']}{suffix}] TF-IDF + "
              f"{'LogReg' if task == 'classification' else 'Ridge'}")
        pred, prob = fit_tfidf(
            txt["train"], y["train"], txt["val"], y["val"], txt["test"], y["test"],
            task=task,
        )
        m = metrics_per_trait(y["test"], pred, prob, task=task, trait_cols=trait_cols)
        print_metrics(name_map["tfidf"] + suffix, m)
        _save("tfidf", pred, prob, m)

    if args.model in ("all", "liwc"):
        print(f"\n[{name_map['liwc']}{suffix}] LIWC features + "
              f"{'LogReg' if task == 'classification' else 'Ridge'}")
        pred, prob, feat_names = fit_liwc(
            txt["train"], y["train"], txt["val"], y["val"], txt["test"], y["test"],
            task=task,
        )
        m = metrics_per_trait(y["test"], pred, prob, task=task, trait_cols=trait_cols)
        print_metrics(name_map["liwc"] + suffix, m)
        _save("liwc", pred, prob, m, feat_names=feat_names)

    if args.model == "all":
        print(f"\n=== Summary ({args.dataset}, test macro-averaged) ===")
        for short in ("dummy", "tfidf", "liwc"):
            full = name_map[short] + suffix
            path = config.RESULTS_DIR / full / "metrics.json"
            if not path.exists():
                continue
            macro = json.loads(path.read_text(encoding="utf-8"))["macro"]
            if "accuracy" in macro:
                auc = macro.get("roc_auc", float("nan"))
                print(f"  {full:<28} acc={macro['accuracy']:.3f}  "
                      f"f1={macro['f1']:.3f}  auc={auc:.3f}")
            else:
                print(f"  {full:<28} rho={macro['spearman']:+.3f}  "
                      f"r={macro['pearson']:+.3f}  mae={macro['mae']:.3f}  "
                      f"r2={macro['r2']:+.3f}")


if __name__ == "__main__":
    main()
