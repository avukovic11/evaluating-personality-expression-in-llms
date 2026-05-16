"""Fine-tune DeBERTa-v3-base as a multi-label binary classifier for Big Five traits.

Pipeline per seed:
  1. Tokenize train/val/test essays (max_length=512).
  2. Fine-tune `microsoft/deberta-v3-base` with a 5-output sigmoid head
     (`problem_type="multi_label_classification"` -> BCEWithLogitsLoss).
  3. Early-stop on val macro-F1 (patience 2); reload best checkpoint.
  4. Tune per-trait sigmoid thresholds on val (maximize per-trait F1).
  5. Predict on test; save probs / preds / thresholds / metrics.

If `--seeds 42,43,44` is given, run each seed independently and write an extra
ensemble row (probabilities averaged across seeds, then thresholded with the
mean of per-seed thresholds).

Outputs:
  code/datasets/checkpoints/deberta-v3-base_seed<N>/         # best model
  code/datasets/results/deberta-v3-base/seeds/<N>/           # per-seed artifacts
  code/datasets/results/deberta-v3-base/{metrics.json, test_predictions.csv}
                                                             # ensemble (or single-seed)

Run from `code/`:
    python -m src.classifier --train
    python -m src.classifier --train --seeds 42,43,44
    python -m src.classifier --train --smoke         # 1 epoch on 64 examples (CPU-safe)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
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
from .baselines import metrics_per_trait, print_metrics, save_results
from .data import load_essays, load_splits


# -----------------------------------------------------------------------------
# Device
# -----------------------------------------------------------------------------

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

def make_dataset(df: pd.DataFrame, tokenizer) -> Dataset:
    """Build a HF Dataset of `{input_ids, attention_mask, labels}` from a DataFrame."""
    work = df[["TEXT"] + config.TRAIT_COLS].reset_index(drop=True)
    ds = Dataset.from_pandas(work, preserve_index=False)

    def tokenize(batch):
        out = tokenizer(
            batch["TEXT"],
            truncation=True,
            max_length=config.MAX_SEQ_LEN,
        )
        out["labels"] = [
            [float(batch[c][i]) for c in config.TRAIT_COLS]
            for i in range(len(batch["TEXT"]))
        ]
        return out

    return ds.map(tokenize, batched=True, remove_columns=ds.column_names)


# -----------------------------------------------------------------------------
# Trainer metric (for early stopping / checkpoint selection)
# -----------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def hf_compute_metrics(eval_pred) -> dict[str, float]:
    """Macro-F1 at 0.5 threshold — used to select the best checkpoint."""
    logits, labels = eval_pred
    preds = (_sigmoid(logits) > 0.5).astype(np.int8)
    labels = np.asarray(labels, dtype=np.int8)
    f1s = [
        f1_score(labels[:, i], preds[:, i], zero_division=0)
        for i in range(labels.shape[1])
    ]
    return {"macro_f1": float(np.mean(f1s))}


# -----------------------------------------------------------------------------
# Threshold tuning
# -----------------------------------------------------------------------------

def tune_thresholds(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-trait threshold ∈ [0.2, 0.8] maximizing F1 on `y`."""
    candidates = np.linspace(0.2, 0.8, 31)
    thresholds = np.full(y.shape[1], 0.5)
    for i in range(y.shape[1]):
        best_f1, best_t = -1.0, 0.5
        for t in candidates:
            preds = (probs[:, i] >= t).astype(np.int8)
            f1 = f1_score(y[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
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
) -> tuple[np.ndarray, np.ndarray]:
    """Train one model, return (test_probs, thresholds)."""
    set_seed(seed)
    device = get_device()
    fp16 = device == "cuda"

    model = AutoModelForSequenceClassification.from_pretrained(
        config.CLASSIFIER_MODEL,
        num_labels=len(config.TRAIT_COLS),
        problem_type="multi_label_classification",
    )

    out_dir = config.CHECKPOINTS_DIR / f"deberta-v3-base_seed{seed}"

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
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=50,
        seed=seed,
        fp16=fp16,
        report_to=[],
        dataloader_num_workers=0,        # safer on Windows; Colab Linux is unaffected
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,      # transformers v5 renamed from `tokenizer`
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=hf_compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    trainer.save_model(str(out_dir))     # persist best model + tokenizer

    val_probs = _sigmoid(trainer.predict(val_ds).predictions)
    thresholds = tune_thresholds(val_probs, val_y)

    test_probs = _sigmoid(trainer.predict(test_ds).predictions)
    return test_probs, thresholds


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def _save_seed_artifacts(
    name: str,
    seed: int,
    probs: np.ndarray,
    pred: np.ndarray,
    thresholds: np.ndarray,
    y_test: np.ndarray,
    test_ids: list[str],
    metrics: dict,
) -> None:
    seed_dir = config.RESULTS_DIR / name / "seeds" / str(seed)
    seed_dir.mkdir(parents=True, exist_ok=True)
    cols: dict[str, list] = {"AUTHID": test_ids}
    for i, t in enumerate(config.TRAIT_COLS):
        cols[f"true_{t}"] = y_test[:, i].tolist()
        cols[f"pred_{t}"] = pred[:, i].tolist()
        cols[f"prob_{t}"] = probs[:, i].tolist()
    pd.DataFrame(cols).to_csv(seed_dir / "test_predictions.csv", index=False)
    (seed_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (seed_dir / "thresholds.json").write_text(
        json.dumps(
            {t: float(thresholds[i]) for i, t in enumerate(config.TRAIT_COLS)},
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", action="store_true", required=False)
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
        help="Subsample train to N rows for quick iteration. Implies --epochs 1 unless set.",
    )
    args = parser.parse_args()

    if not args.train:
        parser.error("Pass --train to fine-tune the classifier.")

    if args.smoke:
        args.max_train_samples = args.max_train_samples or 64
        args.epochs = 1

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    name = "deberta-v3-base"
    config.ensure_dirs()

    print(f"device : {get_device()}")
    print(f"model  : {config.CLASSIFIER_MODEL}")
    print(f"seeds  : {seeds}")
    print(f"epochs : {args.epochs}  batch={args.batch_size}  lr={args.lr}")
    if args.max_train_samples:
        print(f"NOTE: subsampling train to {args.max_train_samples} examples")

    df = load_essays()
    splits = load_splits(df)
    if args.max_train_samples is not None:
        splits["train"] = splits["train"].head(args.max_train_samples)

    tokenizer = AutoTokenizer.from_pretrained(config.CLASSIFIER_MODEL)
    train_ds = make_dataset(splits["train"], tokenizer)
    val_ds = make_dataset(splits["val"], tokenizer)
    test_ds = make_dataset(splits["test"], tokenizer)

    val_y = splits["val"][config.TRAIT_COLS].to_numpy()
    test_y = splits["test"][config.TRAIT_COLS].to_numpy()
    test_ids = splits["test"]["AUTHID"].tolist()

    all_probs: list[np.ndarray] = []
    all_thresholds: list[np.ndarray] = []
    for seed in seeds:
        print(f"\n=== seed {seed} ===")
        probs, thresholds = fit_one_seed(
            seed, train_ds, val_ds, test_ds, val_y, tokenizer,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        )
        pred = (probs >= thresholds[None, :]).astype(np.int8)
        m = metrics_per_trait(test_y, pred, probs)
        print_metrics(f"{name} seed {seed}", m)
        _save_seed_artifacts(name, seed, probs, pred, thresholds, test_y, test_ids, m)
        all_probs.append(probs)
        all_thresholds.append(thresholds)

    # Ensemble (or single-seed): average probs, average thresholds
    ens_probs = np.mean(all_probs, axis=0)
    ens_thresholds = np.mean(all_thresholds, axis=0)
    ens_pred = (ens_probs >= ens_thresholds[None, :]).astype(np.int8)
    ens_metrics = metrics_per_trait(test_y, ens_pred, ens_probs)
    label = "single-seed" if len(seeds) == 1 else f"ensemble (n={len(seeds)})"
    print_metrics(f"{name} {label}", ens_metrics)

    save_results(name, ens_pred, ens_probs, test_y, test_ids, ens_metrics)
    (config.RESULTS_DIR / name / "thresholds.json").write_text(
        json.dumps(
            {t: float(ens_thresholds[i]) for i, t in enumerate(config.TRAIT_COLS)},
            indent=2,
        ),
        encoding="utf-8",
    )

    if len(seeds) > 1:
        # Per-trait std across seeds for the paper's mean±std reporting
        seed_accs = np.array(
            [
                [
                    float(((p >= t[None, :]).astype(np.int8)[:, i] == test_y[:, i]).mean())
                    for i in range(test_y.shape[1])
                ]
                for p, t in zip(all_probs, all_thresholds)
            ]
        )
        stds = seed_accs.std(axis=0, ddof=0)
        print("\nper-trait test-acc std across seeds:")
        for i, t in enumerate(config.TRAIT_COLS):
            print(f"  {t:<6}  mean={seed_accs[:, i].mean():.3f}  std={stds[i]:.3f}")


if __name__ == "__main__":
    main()
