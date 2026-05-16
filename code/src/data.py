"""Load essays.csv, encode labels, and produce stratified train/val/test splits.

Splits are saved as one AUTHID per line in `code/datasets/splits/{train,val,test}.txt`.
We persist IDs (not row indices, not pickled DataFrames) so the splits are robust to
pandas-version churn and remain inspectable in a diff. Reconstructing the per-split
DataFrames is done in-memory at training time via `load_splits(load_essays())`.

The raw CSV is Windows-1252 (smart quotes in essay text); we read it explicitly so
teammates on macOS / Linux (utf-8 default) do not hit decode errors.

Run from `code/`:
    python -m src.data
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from . import config


def load_essays(csv_path: Path = config.ESSAYS_CSV) -> pd.DataFrame:
    """Read the essays CSV, decode labels, and return a clean DataFrame.

    Returns columns: `AUTHID`, `TEXT`, and the five trait columns from
    `config.TRAIT_COLS` as int8 {0, 1}.
    """
    df = pd.read_csv(csv_path, encoding="cp1252")
    df = df.rename(columns={c: c.lstrip("#") for c in df.columns})
    df["TEXT"] = df["TEXT"].astype(str).str.strip()
    for col in config.TRAIT_COLS:
        df[col] = df[col].map({"y": 1, "n": 0}).astype("int8")
    if df[config.TRAIT_COLS].isna().any().any():
        raise ValueError("Unexpected label values in essays.csv — expected y/n only.")
    return df.reset_index(drop=True)


def make_splits(
    df: pd.DataFrame,
    val_size: float = 0.10,
    test_size: float = 0.10,
    seed: int = config.SEED,
) -> dict[str, np.ndarray]:
    """Stratified multi-label split. Returns sorted row-index arrays into `df`."""
    n = len(df)
    y = df[config.TRAIT_COLS].to_numpy()
    indices = np.arange(n).reshape(-1, 1)

    rest_size = val_size + test_size
    splitter1 = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=rest_size, random_state=seed
    )
    train_idx, rest_idx = next(splitter1.split(indices, y))

    rel_test = test_size / rest_size
    splitter2 = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=rel_test, random_state=seed
    )
    rel_val_idx, rel_test_idx = next(
        splitter2.split(rest_idx.reshape(-1, 1), y[rest_idx])
    )
    val_idx = rest_idx[rel_val_idx]
    test_idx = rest_idx[rel_test_idx]

    assert set(train_idx).isdisjoint(val_idx)
    assert set(train_idx).isdisjoint(test_idx)
    assert set(val_idx).isdisjoint(test_idx)
    assert len(train_idx) + len(val_idx) + len(test_idx) == n

    return {
        "train": np.sort(train_idx),
        "val": np.sort(val_idx),
        "test": np.sort(test_idx),
    }


def save_splits(
    splits: dict[str, np.ndarray],
    df: pd.DataFrame,
    out_dir: Path = config.SPLITS_DIR,
) -> None:
    """Write split AUTHIDs as one ID per line in `<out_dir>/{train,val,test}.txt`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, idx in splits.items():
        authids = df.iloc[idx]["AUTHID"].tolist()
        (out_dir / f"{name}.txt").write_text("\n".join(authids) + "\n", encoding="utf-8")


def load_splits(
    df: pd.DataFrame,
    out_dir: Path = config.SPLITS_DIR,
) -> dict[str, pd.DataFrame]:
    """Read split files and return DataFrame slices keyed by split name."""
    id_to_idx = {aid: i for i, aid in enumerate(df["AUTHID"])}
    out: dict[str, pd.DataFrame] = {}
    for name in ("train", "val", "test"):
        path = out_dir / f"{name}.txt"
        if not path.exists():
            raise FileNotFoundError(
                f"Split file {path} not found — run `python -m src.data` first."
            )
        authids = [a for a in path.read_text(encoding="utf-8").splitlines() if a]
        idx = np.array([id_to_idx[a] for a in authids])
        out[name] = df.iloc[idx].reset_index(drop=True)
    return out


def _report(df: pd.DataFrame, splits: dict[str, np.ndarray]) -> None:
    print(f"Total essays: {len(df)}")
    print("Trait y-rates (full set):")
    for col in config.TRAIT_COLS:
        print(f"  {col} ({config.TRAIT_NAMES[col]:<17}): {df[col].mean():.3f}")
    print()
    for name, idx in splits.items():
        sub = df.iloc[idx]
        rates = ", ".join(f"{c}={sub[c].mean():.3f}" for c in config.TRAIT_COLS)
        print(f"{name:<5} n={len(idx):>4} ({len(idx) / len(df):>5.1%})  {rates}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--val-size", type=float, default=0.10)
    parser.add_argument("--test-size", type=float, default=0.10)
    args = parser.parse_args()

    config.ensure_dirs()
    df = load_essays()
    splits = make_splits(
        df, val_size=args.val_size, test_size=args.test_size, seed=args.seed
    )
    save_splits(splits, df)
    _report(df, splits)
    print(f"\nWrote splits to {config.SPLITS_DIR}")


if __name__ == "__main__":
    main()
