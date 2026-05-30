"""Fine-tune a transformer probe for personality prediction on either dataset.

Two tracks share this script via `--dataset`:

  Track 1 — Pennebaker Essays (default; `--dataset essays`):
    Multi-label binary classification over 5 Big Five traits. Sigmoid head,
    BCEWithLogitsLoss. After training, per-trait threshold is tuned on val
    (∈[0.3, 0.7], maximizing accuracy). Early stopping on val macro-accuracy.

  Track 2 — RECRUITVIEW (`--dataset recruitview`):
    Multi-target regression over 5 OCEAN z-scored traits. Linear head, MSE.
    No threshold tuning. Early stopping on val macro Spearman ρ.

Multi-seed: `--seeds 42,43,44` runs each seed independently and writes an
ensemble row at the dataset's natural aggregator (averaged probs + averaged
thresholds for classification; averaged raw predictions for regression).

Outputs:
  Track 1: <CHECKPOINTS_DIR>/<model>_seed<N>/,  <RESULTS_DIR>/<model>/
  Track 2: <CHECKPOINTS_DIR>/<model>_recruitview_seed<N>/,
           <RESULTS_DIR>/<model>_recruitview/

Run from `code/`:
    python -m src.classifier --train                                          # essays + default model
    python -m src.classifier --train --dataset recruitview
    python -m src.classifier --train --model answerdotai/ModernBERT-base
    python -m src.classifier --train --dataset recruitview --model answerdotai/ModernBERT-base
    python -m src.classifier --train --seeds 42,43,44
    python -m src.classifier --train --smoke
    python -m src.classifier --predict-file essay.txt
    cat essay.txt | python -m src.classifier --predict-stdin --dataset recruitview
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from . import config
from .baselines import (
    metrics_per_trait,
    metrics_per_trait_grouped,
    print_metrics,
    save_results,
)


# -----------------------------------------------------------------------------
# Device + helpers
# -----------------------------------------------------------------------------

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _result_dir_name(model_slug: str, dataset_slug: str) -> str:
    """Output dir name. essays → `<model>` (unchanged); recruitview → `<model>_recruitview`."""
    return model_slug if dataset_slug == "essays" else f"{model_slug}_{dataset_slug}"


def _ckpt_dir_name(model_slug: str, dataset_slug: str, seed: int) -> str:
    base = _result_dir_name(model_slug, dataset_slug)
    return f"{base}_seed{seed}"


# -----------------------------------------------------------------------------
# Dataset construction (parametrized over text + trait columns)
# -----------------------------------------------------------------------------

def make_dataset(
    df: pd.DataFrame, tokenizer, text_col: str, trait_cols: list[str],
    max_seq_len: int | None = None,
) -> Dataset:
    """HF Dataset of `{input_ids, attention_mask, labels}` for either track."""
    if max_seq_len is None:
        max_seq_len = config.MAX_SEQ_LEN
    work = df[[text_col] + trait_cols].reset_index(drop=True)
    ds = Dataset.from_pandas(work, preserve_index=False)

    def tokenize(batch):
        out = tokenizer(
            batch[text_col],
            truncation=True,
            max_length=max_seq_len,
        )
        out["labels"] = [
            [float(batch[c][i]) for c in trait_cols]
            for i in range(len(batch[text_col]))
        ]
        return out

    return ds.map(tokenize, batched=True, remove_columns=ds.column_names)


# -----------------------------------------------------------------------------
# HF Trainer compute_metrics — one per task
# -----------------------------------------------------------------------------

def _hf_metrics_classification(eval_pred) -> dict[str, float]:
    """Macro-accuracy + macro-F1 at threshold 0.5. Accuracy drives early stopping."""
    logits, labels = eval_pred
    preds = (_sigmoid(logits) > 0.5).astype(np.int8)
    labels = np.asarray(labels, dtype=np.int8)
    accs = [(labels[:, i] == preds[:, i]).mean() for i in range(labels.shape[1])]
    f1s = [
        f1_score(labels[:, i], preds[:, i], zero_division=0)
        for i in range(labels.shape[1])
    ]
    return {
        "macro_accuracy": float(np.mean(accs)),
        "macro_f1": float(np.mean(f1s)),
    }


def _hf_metrics_regression(eval_pred) -> dict[str, float]:
    """Macro Spearman ρ / Pearson r / MAE. Spearman drives early stopping."""
    preds, labels = eval_pred
    preds = np.asarray(preds, dtype=float)
    labels = np.asarray(labels, dtype=float)
    spearmans, pearsons, maes = [], [], []
    for i in range(labels.shape[1]):
        yt, yp = labels[:, i], preds[:, i]
        if np.std(yp) == 0 or np.std(yt) == 0:
            sp = pr = 0.0
        else:
            sp = float(spearmanr(yt, yp)[0])
            pr = float(pearsonr(yt, yp)[0])
            sp = 0.0 if np.isnan(sp) else sp
            pr = 0.0 if np.isnan(pr) else pr
        spearmans.append(sp)
        pearsons.append(pr)
        maes.append(float(np.mean(np.abs(yt - yp))))
    return {
        "macro_spearman": float(np.mean(spearmans)),
        "macro_pearson": float(np.mean(pearsons)),
        "macro_mae": float(np.mean(maes)),
    }


# -----------------------------------------------------------------------------
# Threshold tuning (classification only)
# -----------------------------------------------------------------------------

def tune_thresholds(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-trait threshold ∈ [0.3, 0.7] maximizing per-trait accuracy on val."""
    candidates = np.linspace(0.3, 0.7, 17)
    thresholds = np.full(y.shape[1], 0.5)
    for i in range(y.shape[1]):
        best_acc, best_t = -1.0, 0.5
        for t in candidates:
            preds = (probs[:, i] >= t).astype(np.int8)
            acc = float((y[:, i] == preds).mean())
            if acc > best_acc:
                best_acc, best_t = acc, float(t)
        thresholds[i] = best_t
    return thresholds


# -----------------------------------------------------------------------------
# Train one seed
# -----------------------------------------------------------------------------

def fit_one_seed(
    seed: int,
    train_ds: Dataset,
    val_ds: Dataset,
    test_ds: Dataset,
    val_y: np.ndarray,
    tokenizer,
    epochs: int,
    batch_size: int,
    lr: float,
    model_name: str,
    task: str,
    trait_cols: list[str],
    dataset_slug: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Train one model.

    Returns:
      - classification: (test_probs after sigmoid, per-trait thresholds)
      - regression:     (test_predictions raw z-scores, None)
    """
    set_seed(seed)
    device = get_device()
    # bf16 on Ampere+ (A100, RTX 30xx, ...); fp32 on Turing (T4) and CPU.
    # fp16 is intentionally disabled: transformers v5 + accelerate hits
    # "Attempting to unscale FP16 gradients" in clip_grad_norm_.
    use_bf16 = (
        device == "cuda" and torch.cuda.get_device_capability()[0] >= 8
    )

    problem_type = (
        "multi_label_classification" if task == "classification" else "regression"
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(trait_cols),
        problem_type=problem_type,
    )

    model_slug = model_name.split("/")[-1]
    out_dir = config.CHECKPOINTS_DIR / _ckpt_dir_name(model_slug, dataset_slug, seed)

    if task == "classification":
        compute_metrics_fn = _hf_metrics_classification
        metric_for_best = "macro_accuracy"
    else:
        compute_metrics_fn = _hf_metrics_regression
        metric_for_best = "macro_spearman"

    args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=max(batch_size * 2, 32),
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best,
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=50,
        seed=seed,
        bf16=use_bf16,
        report_to=[],
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    trainer.save_model(str(out_dir))

    if task == "classification":
        val_probs = _sigmoid(trainer.predict(val_ds).predictions)
        thresholds = tune_thresholds(val_probs, val_y)
        test_probs = _sigmoid(trainer.predict(test_ds).predictions)
        return test_probs, thresholds

    test_preds = np.asarray(trainer.predict(test_ds).predictions, dtype=float)
    return test_preds, None


# -----------------------------------------------------------------------------
# Inference on a single text (both tracks)
# -----------------------------------------------------------------------------

def predict_text(
    text: str, checkpoint_dir: Path, task: str, trait_cols: list[str],
) -> dict:
    """Return per-trait score for a single document.

    Classification: sigmoid probability in [0, 1].
    Regression: raw z-score (typically ~[-2, +2], can go further on outliers).
    """
    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_dir}. Train first with --train."
        )
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint_dir))
    device = get_device()
    model.to(device).eval()

    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=config.MAX_SEQ_LEN,
    ).to(device)
    with torch.no_grad():
        raw = model(**inputs).logits[0].cpu().numpy()
    scores = _sigmoid(raw) if task == "classification" else raw
    return {t: float(scores[i]) for i, t in enumerate(trait_cols)}


# -----------------------------------------------------------------------------
# Per-seed save
# -----------------------------------------------------------------------------

def _save_seed_artifacts(
    name: str,
    seed: int,
    predictions: np.ndarray,
    thresholds: np.ndarray | None,
    y_test: np.ndarray,
    test_ids: list[str],
    metrics: dict,
    trait_cols: list[str],
    task: str,
    id_col_name: str,
    extra_cols: dict[str, list] | None = None,
) -> None:
    """Per-seed predictions + metrics. `predictions` is probs for classif, raw preds for regr."""
    seed_dir = config.RESULTS_DIR / name / "seeds" / str(seed)
    seed_dir.mkdir(parents=True, exist_ok=True)
    cols: dict[str, list] = {id_col_name: test_ids}
    if extra_cols:
        for k, v in extra_cols.items():
            if k not in cols:
                cols[k] = v
    if task == "classification":
        assert thresholds is not None
        probs = predictions
        pred = (probs >= thresholds[None, :]).astype(np.int8)
        for i, t in enumerate(trait_cols):
            cols[f"true_{t}"] = y_test[:, i].tolist()
            cols[f"pred_{t}"] = pred[:, i].tolist()
            cols[f"prob_{t}"] = probs[:, i].tolist()
    else:
        for i, t in enumerate(trait_cols):
            cols[f"true_{t}"] = y_test[:, i].tolist()
            cols[f"pred_{t}"] = predictions[:, i].tolist()
    pd.DataFrame(cols).to_csv(seed_dir / "test_predictions.csv", index=False)
    (seed_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    if task == "classification" and thresholds is not None:
        (seed_dir / "thresholds.json").write_text(
            json.dumps(
                {t: float(thresholds[i]) for i, t in enumerate(trait_cols)},
                indent=2,
            ),
            encoding="utf-8",
        )


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def _load_dataset_bundle(dataset: str):
    """Returns a dict with df, splits, trait_cols, text_col, id_col, task, slug, names."""
    if dataset == "essays":
        from .data import load_essays, load_splits
        df = load_essays()
        splits = load_splits(df)
        return {
            "df": df,
            "splits": splits,
            "task": "classification",
            "trait_cols": config.TRAIT_COLS,
            "trait_names": config.TRAIT_NAMES,
            "text_col": "TEXT",
            "id_col": "AUTHID",
            "dataset_slug": "essays",
            "extra_cols_keys": [],
        }
    if dataset == "recruitview":
        from .data_recruitview import load_recruitview, load_recruitview_splits
        df = load_recruitview()
        splits = load_recruitview_splits(df)
        return {
            "df": df,
            "splits": splits,
            "task": "regression",
            "trait_cols": config.RECRUITVIEW_TRAIT_COLS,
            "trait_names": config.RECRUITVIEW_TRAIT_NAMES,
            "text_col": "transcript",
            "id_col": "id",
            "dataset_slug": "recruitview",
            "extra_cols_keys": ["user_no", "question_id"],
        }
    raise ValueError(f"Unknown dataset: {dataset!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", action="store_true", required=False)
    parser.add_argument(
        "--dataset", choices=["essays", "recruitview"], default="essays",
        help="Which dataset/task to train on.",
    )
    parser.add_argument(
        "--seeds", type=str, default="42",
        help="Comma-separated seeds, e.g. '42,43,44'.",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument(
        "--smoke", action="store_true",
        help="Quick sanity run: 1 epoch on 64 train examples (CPU-tolerable).",
    )
    parser.add_argument(
        "--max-train-samples", type=int, default=None,
        help="Subsample train to N rows for quick iteration.",
    )
    parser.add_argument(
        "--predict-file", type=str, default=None, metavar="PATH",
        help="Skip training; predict per-trait scores for the document in PATH.",
    )
    parser.add_argument(
        "--predict-stdin", action="store_true",
        help="Skip training; read document from stdin and predict.",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default=None, metavar="PATH",
        help="Override checkpoint dir for --predict-*.",
    )
    parser.add_argument(
        "--model", type=str, default=config.CLASSIFIER_MODEL, metavar="HF_ID",
        help=f"HuggingFace model id. Default {config.CLASSIFIER_MODEL}.",
    )
    parser.add_argument(
        "--max-seq-len", type=int, default=config.MAX_SEQ_LEN,
        help=(
            f"Max tokens per input. Default {config.MAX_SEQ_LEN}. Encoders "
            f"like ModernBERT support up to 8192; longer contexts help on "
            f"essays whose mean length (~850 tokens) exceeds 512. The "
            f"checkpoint dir is suffixed with _seqN when N != default."
        ),
    )
    args = parser.parse_args()

    bundle = _load_dataset_bundle(args.dataset)
    task = bundle["task"]
    trait_cols = bundle["trait_cols"]
    trait_names = bundle["trait_names"]
    text_col = bundle["text_col"]
    id_col = bundle["id_col"]
    dataset_slug = bundle["dataset_slug"]

    if args.predict_file or args.predict_stdin:
        if args.predict_file:
            text = Path(args.predict_file).read_text(encoding="utf-8")
        else:
            import sys
            text = sys.stdin.read()
        if not text.strip():
            raise SystemExit("Empty input.")

        model_slug = args.model.split("/")[-1]
        default_ckpt = (
            config.CHECKPOINTS_DIR / _ckpt_dir_name(model_slug, dataset_slug, 42)
        )
        ckpt = Path(args.checkpoint_dir) if args.checkpoint_dir else default_ckpt
        scores = predict_text(text, ckpt, task=task, trait_cols=trait_cols)

        n_words = len(text.split())
        preview = text.strip().replace("\n", " ")[:100]
        print(f"Text ({n_words} words):")
        print(f"  {preview}{'...' if len(text) > 100 else ''}")
        print()
        if task == "classification":
            print(f"  {'trait':<6}  {'name':<18}  {'prob':>5}  {'pred@0.5':>8}")
            for t, p in scores.items():
                pred_label = "y" if p >= 0.5 else "n"
                print(
                    f"  {t:<6}  {trait_names[t]:<18}  {p:>5.3f}  {pred_label:>8}"
                )
        else:
            print(f"  {'trait':<18}  {'name':<18}  {'z-score':>+8}")
            for t, z in scores.items():
                print(f"  {t:<18}  {trait_names[t]:<18}  {z:+8.3f}")
        return

    if not args.train:
        parser.error(
            "Pass --train to fine-tune, or --predict-file / --predict-stdin to score text."
        )

    if args.smoke:
        args.max_train_samples = args.max_train_samples or 64
        args.epochs = 1

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    config.ensure_dirs()

    print(f"device  : {get_device()}")
    print(f"dataset : {args.dataset} ({task}, {len(trait_cols)} traits)")
    print(f"model   : {args.model}")
    print(f"seeds   : {seeds}")
    print(f"epochs  : {args.epochs}  batch={args.batch_size}  lr={args.lr}  max_seq_len={args.max_seq_len}")
    if args.max_train_samples:
        print(f"NOTE: subsampling train to {args.max_train_samples} examples")

    splits = bundle["splits"]
    if args.max_train_samples is not None:
        splits["train"] = splits["train"].head(args.max_train_samples)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train_ds = make_dataset(splits["train"], tokenizer, text_col, trait_cols, max_seq_len=args.max_seq_len)
    val_ds = make_dataset(splits["val"], tokenizer, text_col, trait_cols, max_seq_len=args.max_seq_len)
    test_ds = make_dataset(splits["test"], tokenizer, text_col, trait_cols, max_seq_len=args.max_seq_len)

    val_y = splits["val"][trait_cols].to_numpy(dtype=float)
    test_y = splits["test"][trait_cols].to_numpy(dtype=float)
    test_ids = splits["test"][id_col].astype(str).tolist()
    extra_cols: dict[str, list] = {}
    for k in bundle["extra_cols_keys"]:
        extra_cols[k] = splits["test"][k].astype(str).tolist()

    model_slug = args.model.split("/")[-1]
    # Suffix the model slug with _seqN whenever the user picks a non-default
    # max_seq_len, so a longer-context retraining doesn't overwrite the
    # original 512-token results / checkpoints.
    if args.max_seq_len != config.MAX_SEQ_LEN:
        model_slug = f"{model_slug}_seq{args.max_seq_len}"
    name = _result_dir_name(model_slug, dataset_slug)

    all_outputs: list[np.ndarray] = []
    all_thresholds: list[np.ndarray] = [] if task == "classification" else []
    for seed in seeds:
        print(f"\n=== seed {seed} ===")
        outputs, thresholds = fit_one_seed(
            seed, train_ds, val_ds, test_ds, val_y, tokenizer,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            model_name=args.model, task=task, trait_cols=trait_cols,
            dataset_slug=dataset_slug,
        )
        all_outputs.append(outputs)

        # Per-seed metrics + save
        if task == "classification":
            all_thresholds.append(thresholds)
            pred = (outputs >= thresholds[None, :]).astype(np.int8)
            m = metrics_per_trait(
                test_y, pred, outputs, task=task, trait_cols=trait_cols,
            )
        else:
            pred = outputs
            m = metrics_per_trait(
                test_y, pred, None, task=task, trait_cols=trait_cols,
            )
        print_metrics(f"{name} seed {seed}", m)
        _save_seed_artifacts(
            name, seed, outputs, thresholds, test_y, test_ids, m,
            trait_cols=trait_cols, task=task, id_col_name=id_col,
            extra_cols=extra_cols,
        )

    # Ensemble (or single-seed): average outputs (and thresholds for classif).
    ens_outputs = np.mean(all_outputs, axis=0)
    if task == "classification":
        ens_thresholds = np.mean(all_thresholds, axis=0)
        ens_pred = (ens_outputs >= ens_thresholds[None, :]).astype(np.int8)
        ens_metrics = metrics_per_trait(
            test_y, ens_pred, ens_outputs, task=task, trait_cols=trait_cols,
        )
    else:
        ens_pred = ens_outputs
        ens_metrics = metrics_per_trait(
            test_y, ens_pred, None, task=task, trait_cols=trait_cols,
        )
    label = "single-seed" if len(seeds) == 1 else f"ensemble (n={len(seeds)})"
    print_metrics(f"{name} {label}", ens_metrics)

    # For RECRUITVIEW: compute user-aggregated metrics. Labels are user-level
    # (every clip from the same user has the same z-score), so the clip-level
    # rho is dragged down by within-user noise. The user-aggregated rho is
    # the metric that matches the dataset's annotation unit and the number
    # we report alongside it in the paper.
    if task == "regression" and "user_no" in extra_cols:
        user_metrics = metrics_per_trait_grouped(
            test_y, ens_outputs, extra_cols["user_no"], trait_cols,
        )
        print_metrics(
            f"{name} {label} — user-aggregated (n_users={user_metrics['n_groups']})",
            {k: v for k, v in user_metrics.items() if k != "n_groups"},
        )
        # Attach as a sibling key so downstream tooling can read both.
        ens_metrics = dict(ens_metrics)
        ens_metrics["user_aggregated"] = user_metrics

    save_results(
        name,
        ens_pred if task == "classification" else ens_outputs,
        ens_outputs if task == "classification" else None,
        test_y, test_ids, ens_metrics,
        trait_cols=trait_cols, id_col_name=id_col, extra_cols=extra_cols,
    )
    if task == "classification":
        (config.RESULTS_DIR / name / "thresholds.json").write_text(
            json.dumps(
                {t: float(ens_thresholds[i]) for i, t in enumerate(trait_cols)},
                indent=2,
            ),
            encoding="utf-8",
        )

    # Multi-seed std reporting (different metric per task).
    if len(seeds) > 1:
        print("\nper-seed std on the chosen early-stopping metric:")
        if task == "classification":
            per_seed = np.array([
                [
                    float(((p >= t[None, :]).astype(np.int8)[:, i] == test_y[:, i]).mean())
                    for i in range(test_y.shape[1])
                ]
                for p, t in zip(all_outputs, all_thresholds)
            ])
            label_text = "test-acc"
        else:
            per_seed = np.array([
                [
                    spearmanr(test_y[:, i], p[:, i])[0] if np.std(p[:, i]) > 0 else 0.0
                    for i in range(test_y.shape[1])
                ]
                for p in all_outputs
            ])
            per_seed = np.nan_to_num(per_seed, nan=0.0)
            label_text = "test-spearman"
        stds = per_seed.std(axis=0, ddof=0)
        for i, t in enumerate(trait_cols):
            print(
                f"  {t:<18}  mean {label_text}={per_seed[:, i].mean():+.3f}  "
                f"std={stds[i]:.3f}"
            )


if __name__ == "__main__":
    main()
