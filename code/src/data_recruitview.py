"""Load RECRUITVIEW from HuggingFace and produce user-stratified train/val/test splits.

Loads `AI4A-lab/RecruitView` via `datasets.load_dataset`, keeps only the OCEAN
trait scores + question/user metadata + transcript, and drops the 6 multimodal
performance metrics, the gemini summary, and the video/audio columns.

**Splits are user-level, not row-level.** The dataset has ~6 clips per user on
average (331 users / 2011 rows), so a row-level split would leak the same person
across train and test. We hash-shuffle the user list with a fixed seed and
assign each user *entirely* to one split (70 / 15 / 15 by user count).

User-level splits are committed at `code/datasets/splits_recruitview/{train,val,test}_users.txt`
as one `user_no` per line. Each module that needs the split slices the full
DataFrame at load time with `load_recruitview_splits(df)`.

Run from `code/`:
    python -m src.data_recruitview
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

DATASET_ID = "AI4A-lab/RecruitView"
DEFAULT_MIN_WORDS = 5  # drop rows with near-empty transcripts (silence / refusals)

_KEEP_COLS = [
    "id", "question_id", "question", "user_no", "transcript",
    *config.RECRUITVIEW_TRAIT_COLS,
]


def load_recruitview(min_words: int = DEFAULT_MIN_WORDS) -> pd.DataFrame:
    """Load RECRUITVIEW from HuggingFace; keep OCEAN + metadata only.

    Drops the video/audio columns *before* materializing to pandas so we never
    pull video binaries into memory. Also filters rows whose transcript has
    fewer than `min_words` words (silence / refusals).
    """
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID)

    # RECRUITVIEW ships as a single split (typically "train"); we recompute
    # our own train/val/test, so combine whatever splits exist.
    if hasattr(ds, "keys"):
        parts: list[pd.DataFrame] = []
        for split_name in ds.keys():
            sub = ds[split_name]
            drop_cols = [c for c in sub.column_names if c not in _KEEP_COLS]
            sub = sub.remove_columns(drop_cols)
            parts.append(sub.to_pandas())
        df = pd.concat(parts, ignore_index=True)
    else:
        drop_cols = [c for c in ds.column_names if c not in _KEEP_COLS]
        df = ds.remove_columns(drop_cols).to_pandas()

    missing = set(_KEEP_COLS) - set(df.columns)
    if missing:
        raise ValueError(
            f"RECRUITVIEW is missing expected columns: {sorted(missing)}. "
            f"Got: {sorted(df.columns)}"
        )

    df = df[_KEEP_COLS].copy()
    df["transcript"] = df["transcript"].astype(str).str.strip()
    df["user_no"] = df["user_no"].astype(str)

    word_counts = df["transcript"].str.split().str.len()
    n_before = len(df)
    df = df[word_counts >= min_words].reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"  dropped {n_dropped} rows with <{min_words} words in transcript")

    return df


def make_user_splits(
    df: pd.DataFrame,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = config.SEED,
) -> dict[str, list[str]]:
    """Partition unique `user_no` values into train/val/test (each user → one split)."""
    if val_frac + test_frac >= 1.0:
        raise ValueError("val_frac + test_frac must be < 1.0")

    users = sorted(df["user_no"].unique())
    rng = np.random.default_rng(seed)
    order = np.arange(len(users))
    rng.shuffle(order)
    users_shuffled = [users[i] for i in order]

    n = len(users_shuffled)
    n_test = int(round(test_frac * n))
    n_val = int(round(val_frac * n))

    test_users = users_shuffled[:n_test]
    val_users = users_shuffled[n_test:n_test + n_val]
    train_users = users_shuffled[n_test + n_val:]

    # Sanity: disjoint sets, covers all users.
    s_train, s_val, s_test = set(train_users), set(val_users), set(test_users)
    assert s_train.isdisjoint(s_val)
    assert s_train.isdisjoint(s_test)
    assert s_val.isdisjoint(s_test)
    assert len(s_train) + len(s_val) + len(s_test) == n

    return {"train": train_users, "val": val_users, "test": test_users}


def save_user_splits(
    splits: dict[str, list[str]],
    out_dir: Path | None = None,
) -> None:
    out_dir = out_dir or config.RECRUITVIEW_SPLITS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, users in splits.items():
        path = out_dir / f"{name}_users.txt"
        path.write_text("\n".join(users) + "\n", encoding="utf-8")


def load_recruitview_splits(
    df: pd.DataFrame,
    out_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Read user_no files and return DataFrame slices keyed by split name."""
    out_dir = out_dir or config.RECRUITVIEW_SPLITS_DIR
    out: dict[str, pd.DataFrame] = {}
    for name in ("train", "val", "test"):
        path = out_dir / f"{name}_users.txt"
        if not path.exists():
            raise FileNotFoundError(
                f"Split file {path} not found. "
                f"Run `python -m src.data_recruitview` first."
            )
        users = {u for u in path.read_text(encoding="utf-8").splitlines() if u}
        out[name] = df[df["user_no"].isin(users)].reset_index(drop=True)
    return out


def _report(df: pd.DataFrame, splits: dict[str, list[str]]) -> None:
    print(f"\nTotal rows    : {len(df)}")
    print(f"Total users   : {df['user_no'].nunique()}")
    print(f"Total questions: {df['question_id'].nunique()}")

    print("\nPer-trait stats (z-scored):")
    for col in config.RECRUITVIEW_TRAIT_COLS:
        vals = df[col]
        print(
            f"  {col:<18} mean={vals.mean():+.3f}  std={vals.std():.3f}  "
            f"min={vals.min():+.2f}  max={vals.max():+.2f}"
        )

    word_counts = df["transcript"].str.split().str.len()
    print(
        f"\nTranscript words: mean={word_counts.mean():.1f}  "
        f"median={word_counts.median():.0f}  "
        f"min={word_counts.min()}  max={word_counts.max()}"
    )

    rows_per_user = df.groupby("user_no").size()
    n_total_users = df["user_no"].nunique()
    print(f"\nSplits (each user assigned entirely to one split):")
    for name, users in splits.items():
        n_users = len(users)
        n_rows = int(rows_per_user.loc[rows_per_user.index.isin(users)].sum())
        print(
            f"  {name:<5} {n_users:>4} users ({n_users / n_total_users:.1%})  "
            f"→ {n_rows:>5} rows ({n_rows / len(df):.1%})"
        )

    # Hard assertion against identity leakage.
    s_train, s_val, s_test = (set(splits[k]) for k in ("train", "val", "test"))
    assert s_train.isdisjoint(s_val)
    assert s_train.isdisjoint(s_test)
    assert s_val.isdisjoint(s_test)
    print("\n  ✓ zero user_no overlap across splits")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument(
        "--min-words", type=int, default=DEFAULT_MIN_WORDS,
        help=f"Drop transcripts with < N words. Default {DEFAULT_MIN_WORDS}.",
    )
    args = parser.parse_args()

    config.ensure_dirs()
    print(f"Loading {DATASET_ID} from HuggingFace …")
    df = load_recruitview(min_words=args.min_words)

    splits = make_user_splits(
        df,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )
    save_user_splits(splits)
    _report(df, splits)
    print(f"\nWrote user splits to {config.RECRUITVIEW_SPLITS_DIR}")


if __name__ == "__main__":
    main()
