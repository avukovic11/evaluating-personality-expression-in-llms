"""Lower-bound baselines on the Essays dataset.

Three multi-label binary baselines over the five Big Five traits:

- **dummy** — DummyClassifier(most_frequent) per trait. Sanity floor.
- **tfidf** — TF-IDF + Logistic Regression. Tunes (C, ngram_range) on val by macro
  accuracy, then refits on train+val and predicts test.
- **liwc**  — Hand-crafted features + Logistic Regression. Features: word/sentence
  length stats, lexical diversity, pronoun rates, negation rate, punctuation rates,
  and NRC EmoLex emotion frequencies. Tunes C on val, then refits and predicts test.

Per-baseline outputs go to `code/datasets/results/<name>/`:
    metrics.json           — per-trait + macro accuracy / F1 / AUC
    test_predictions.csv   — AUTHID, true/pred/prob per trait
    feature_names.txt      — LIWC only

Run from `code/`:
    python -m src.baselines
    python -m src.baselines --model tfidf
"""

from __future__ import annotations

import argparse
import json
import re

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from . import config
from .data import load_essays, load_splits


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def metrics_per_trait(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None
) -> dict:
    """Return per-trait accuracy / F1 / AUC plus macro averages."""
    out: dict = {}
    for i, trait in enumerate(config.TRAIT_COLS):
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
    macro = {
        "accuracy": float(np.mean([out[t]["accuracy"] for t in config.TRAIT_COLS])),
        "f1": float(np.mean([out[t]["f1"] for t in config.TRAIT_COLS])),
    }
    if y_prob is not None:
        macro["roc_auc"] = float(
            np.nanmean([out[t]["roc_auc"] for t in config.TRAIT_COLS])
        )
    out["macro"] = macro
    return out


def print_metrics(name: str, m: dict) -> None:
    has_auc = "roc_auc" in m[config.TRAIT_COLS[0]]
    header = f"  {'trait':<8} {'acc':>6} {'f1':>6}" + (f" {'auc':>6}" if has_auc else "")
    print(header)
    for trait in config.TRAIT_COLS:
        d = m[trait]
        line = f"  {trait:<8} {d['accuracy']:>6.3f} {d['f1']:>6.3f}"
        if has_auc:
            line += f" {d['roc_auc']:>6.3f}"
        print(line)
    d = m["macro"]
    line = f"  {'macro':<8} {d['accuracy']:>6.3f} {d['f1']:>6.3f}"
    if has_auc:
        line += f" {d['roc_auc']:>6.3f}"
    print(line)


# -----------------------------------------------------------------------------
# Dummy / majority baseline
# -----------------------------------------------------------------------------

def fit_dummy(train_y: np.ndarray, test_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred = np.zeros_like(test_y)
    prob = np.zeros_like(test_y, dtype=float)
    dummy_X_train = np.zeros((len(train_y), 1))
    dummy_X_test = np.zeros((len(test_y), 1))
    for i in range(train_y.shape[1]):
        m = DummyClassifier(strategy="most_frequent", random_state=config.SEED)
        m.fit(dummy_X_train, train_y[:, i])
        pred[:, i] = m.predict(dummy_X_test)
        # predict_proba returns columns sorted by class label (0, 1).
        p = m.predict_proba(dummy_X_test)
        prob[:, i] = p[:, 1] if p.shape[1] == 2 else float(train_y[:, i].mean())
    return pred, prob


# -----------------------------------------------------------------------------
# TF-IDF + Logistic Regression
# -----------------------------------------------------------------------------

TFIDF_GRID = [
    {"ngram_range": (1, 1), "C": 0.5},
    {"ngram_range": (1, 1), "C": 1.0},
    {"ngram_range": (1, 1), "C": 2.0},
    {"ngram_range": (1, 2), "C": 0.5},
    {"ngram_range": (1, 2), "C": 1.0},
    {"ngram_range": (1, 2), "C": 2.0},
]


def fit_tfidf_lr(
    train_text: list[str], train_y: np.ndarray,
    val_text: list[str], val_y: np.ndarray,
    test_text: list[str], test_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    best_score, best_g = -1.0, None
    for g in TFIDF_GRID:
        vec = TfidfVectorizer(
            ngram_range=g["ngram_range"], min_df=5, max_df=0.95,
            sublinear_tf=True, lowercase=True,
        )
        X_train = vec.fit_transform(train_text)
        X_val = vec.transform(val_text)
        accs = []
        for i in range(train_y.shape[1]):
            clf = LogisticRegression(
                C=g["C"], max_iter=2000, random_state=config.SEED
            )
            clf.fit(X_train, train_y[:, i])
            accs.append(accuracy_score(val_y[:, i], clf.predict(X_val)))
        avg = float(np.mean(accs))
        print(f"  ngram={g['ngram_range']} C={g['C']:>4}  val macro-acc={avg:.3f}")
        if avg > best_score:
            best_score, best_g = avg, g
    assert best_g is not None
    print(f"  best: ngram={best_g['ngram_range']} C={best_g['C']} (val={best_score:.3f})")

    vec = TfidfVectorizer(
        ngram_range=best_g["ngram_range"], min_df=5, max_df=0.95,
        sublinear_tf=True, lowercase=True,
    )
    X_all = vec.fit_transform(list(train_text) + list(val_text))
    X_test = vec.transform(test_text)
    all_y = np.concatenate([train_y, val_y], axis=0)

    pred = np.zeros_like(test_y)
    prob = np.zeros_like(test_y, dtype=float)
    for i in range(test_y.shape[1]):
        clf = LogisticRegression(C=best_g["C"], max_iter=2000, random_state=config.SEED)
        clf.fit(X_all, all_y[:, i])
        pred[:, i] = clf.predict(X_test)
        prob[:, i] = clf.predict_proba(X_test)[:, 1]
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
    """Make sure tokenizer data nrclex relies on is available."""
    import nltk
    for pkg, path in [("punkt", "tokenizers/punkt"), ("punkt_tab", "tokenizers/punkt_tab")]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


def liwc_features(text: str) -> dict[str, float]:
    """Extract hand-crafted features for a single essay."""
    from nrclex import NRCLex

    text_lower = text.lower()
    tokens = WORD_RE.findall(text_lower)
    n_words = max(len(tokens), 1)
    n_sent = max(len([s for s in SENT_RE.split(text) if s.strip()]), 1)

    feats: dict[str, float] = {
        "word_count_log": float(np.log1p(n_words)),
        "avg_sentence_len": n_words / n_sent,
        "avg_word_len": float(np.mean([len(t) for t in tokens])) if tokens else 0.0,
        "ttr": len(set(tokens)) / n_words,
        "pron_i": sum(t in PRONOUNS_I for t in tokens) / n_words,
        "pron_we": sum(t in PRONOUNS_WE for t in tokens) / n_words,
        "pron_you": sum(t in PRONOUNS_YOU for t in tokens) / n_words,
        "pron_3p": sum(t in PRONOUNS_3P for t in tokens) / n_words,
        "negation": sum(t in NEGATIONS for t in tokens) / n_words,
        "q_rate": text.count("?") / n_sent,
        "excl_rate": text.count("!") / n_sent,
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


def fit_liwc_lr(
    train_text: list[str], train_y: np.ndarray,
    val_text: list[str], val_y: np.ndarray,
    test_text: list[str], test_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    _ensure_nltk()
    X_train, feat_names = extract_liwc_matrix(train_text, "LIWC train")
    X_val, _ = extract_liwc_matrix(val_text, "LIWC val")
    X_test, _ = extract_liwc_matrix(test_text, "LIWC test")

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)

    grid = [0.1, 0.5, 1.0, 2.0, 5.0]
    best_score, best_C = -1.0, None
    for C in grid:
        accs = []
        for i in range(train_y.shape[1]):
            clf = LogisticRegression(C=C, max_iter=2000, random_state=config.SEED)
            clf.fit(X_train_s, train_y[:, i])
            accs.append(accuracy_score(val_y[:, i], clf.predict(X_val_s)))
        avg = float(np.mean(accs))
        print(f"  C={C:>4}  val macro-acc={avg:.3f}")
        if avg > best_score:
            best_score, best_C = avg, C
    print(f"  best: C={best_C} (val={best_score:.3f})")

    # Refit scaler on train+val, then LR per trait
    X_all = np.concatenate([X_train, X_val], axis=0)
    all_y = np.concatenate([train_y, val_y], axis=0)
    scaler2 = StandardScaler().fit(X_all)
    X_all_s = scaler2.transform(X_all)
    X_test_s = scaler2.transform(X_test)

    pred = np.zeros_like(test_y)
    prob = np.zeros_like(test_y, dtype=float)
    for i in range(test_y.shape[1]):
        clf = LogisticRegression(C=best_C, max_iter=2000, random_state=config.SEED)
        clf.fit(X_all_s, all_y[:, i])
        pred[:, i] = clf.predict(X_test_s)
        prob[:, i] = clf.predict_proba(X_test_s)[:, 1]
    return pred, prob, feat_names


# -----------------------------------------------------------------------------
# I/O
# -----------------------------------------------------------------------------

def save_results(
    name: str,
    pred: np.ndarray,
    prob: np.ndarray,
    y_test: np.ndarray,
    test_ids: list[str],
    metrics: dict,
    feat_names: list[str] | None = None,
) -> None:
    out_dir = config.RESULTS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    cols: dict[str, list] = {"AUTHID": test_ids}
    for i, t in enumerate(config.TRAIT_COLS):
        cols[f"true_{t}"] = y_test[:, i].tolist()
        cols[f"pred_{t}"] = pred[:, i].tolist()
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
        "--model", choices=["all", "dummy", "tfidf", "liwc"], default="all"
    )
    args = parser.parse_args()

    config.ensure_dirs()
    df = load_essays()
    splits = load_splits(df)
    y = {k: splits[k][config.TRAIT_COLS].to_numpy() for k in splits}
    txt = {k: splits[k]["TEXT"].tolist() for k in splits}
    test_ids = splits["test"]["AUTHID"].tolist()

    if args.model in ("all", "dummy"):
        print("\n[dummy] majority class per trait")
        pred, prob = fit_dummy(y["train"], y["test"])
        m = metrics_per_trait(y["test"], pred, prob)
        print_metrics("dummy", m)
        save_results("dummy", pred, prob, y["test"], test_ids, m)

    if args.model in ("all", "tfidf"):
        print("\n[tfidf-lr] TF-IDF + Logistic Regression")
        pred, prob = fit_tfidf_lr(
            txt["train"], y["train"], txt["val"], y["val"], txt["test"], y["test"]
        )
        m = metrics_per_trait(y["test"], pred, prob)
        print_metrics("tfidf-lr", m)
        save_results("tfidf-lr", pred, prob, y["test"], test_ids, m)

    if args.model in ("all", "liwc"):
        print("\n[liwc-lr] LIWC-style features + Logistic Regression")
        pred, prob, feat_names = fit_liwc_lr(
            txt["train"], y["train"], txt["val"], y["val"], txt["test"], y["test"]
        )
        m = metrics_per_trait(y["test"], pred, prob)
        print_metrics("liwc-lr", m)
        save_results("liwc-lr", pred, prob, y["test"], test_ids, m, feat_names=feat_names)

    if args.model == "all":
        print("\n=== Summary (test macro-averaged) ===")
        for name in ("dummy", "tfidf-lr", "liwc-lr"):
            with open(config.RESULTS_DIR / name / "metrics.json", encoding="utf-8") as f:
                macro = json.load(f)["macro"]
            auc = macro.get("roc_auc", float("nan"))
            print(
                f"  {name:<10}  acc={macro['accuracy']:.3f}  "
                f"f1={macro['f1']:.3f}  auc={auc:.3f}"
            )


if __name__ == "__main__":
    main()
