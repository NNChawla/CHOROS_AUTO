"""
Deterministic stratified train / val / test splits for DEVCOM and FAB.

DEVCOM: 40 run groups  → 28 train / 6 val / 6 test  (≈70/15/15)
  Stratified on firing_accuracy_AOBJ_s3 (per-run mean, overall median).
  Runs 12 and 16 have no D3 objective scores; they are forced to train.

FAB:    96 participants → 67 train / 14 val / 15 test (≈70/15/15)
  Stratified on portScore (overall median, 1 row per PID).

Split membership is loaded from training/split_manifest.json.  The helper
that originally computed the manifest is kept below for auditing/regeneration,
but runtime code must use the frozen manifest for stable evaluation.

CRITICAL: DEVCOM permutation calls must be made before FAB permutation calls
from the same RNG instance.  Reordering these calls changes all splits.
"""

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

SPLIT_SEED = 42
DATA_ROOT  = Path(os.environ.get("CHOROS_DATA_ROOT", "/srv/CHOROS/data"))
MANIFEST_PATH = Path(__file__).with_name("split_manifest.json")

_DEVCOM_RUN_RE  = re.compile(r"DEVCOM_run_(\d+)_session_")
_FAB_PID_RE     = re.compile(r"FAB_(\w+)_Build[AB]_")
_DEVCOM_FILE_RE = re.compile(r"DEVCOM_run_(\d+)_session_")


def _stratified_split(
    ids: np.ndarray,
    labels: np.ndarray,
    n_val: int,
    n_test: int,
    rng: np.random.Generator,
) -> tuple[list, list, list]:
    """
    Proportional stratified split into train / val / test.

    Each class is shuffled independently with `rng` (preserving the shared
    RNG call sequence), then allocated proportionally to val and test.
    The remainder goes to train.
    """
    idx0 = np.where(labels == 0)[0]
    idx1 = np.where(labels == 1)[0]

    idx0 = idx0[rng.permutation(len(idx0))]
    idx1 = idx1[rng.permutation(len(idx1))]

    n_total = len(ids)
    p0 = len(idx0) / n_total

    n_val_0, n_val_1   = int(p0 * n_val),  n_val  - int(p0 * n_val)
    n_test_0, n_test_1 = int(p0 * n_test), n_test - int(p0 * n_test)

    val_idx   = np.concatenate([idx0[:n_val_0],
                                 idx1[:n_val_1]])
    test_idx  = np.concatenate([idx0[n_val_0:n_val_0 + n_test_0],
                                 idx1[n_val_1:n_val_1 + n_test_1]])
    train_idx = np.concatenate([idx0[n_val_0 + n_test_0:],
                                 idx1[n_val_1 + n_test_1:]])

    return (sorted(ids[train_idx].tolist()),
            sorted(ids[val_idx].tolist()),
            sorted(ids[test_idx].tolist()))


def _compute_splits() -> tuple[dict, dict]:
    # ── DEVCOM ────────────────────────────────────────────────────────────────
    devcom_dir = DATA_ROOT / "aligned" / "target_DEVCOM_s2"
    run_ids = sorted({int(m.group(1))
                      for f in devcom_dir.glob("DEVCOM_run_*_ego.parquet")
                      if (m := _DEVCOM_RUN_RE.match(f.name))})

    # Per-run firing_accuracy (mean across roles BG/BL/BR/BS)
    devcom_csv = DATA_ROOT / "target_objective_D3" / "DEVCOM_s3_metrics.csv"
    devcom_meta = pd.read_csv(devcom_csv)
    devcom_meta['_run_id'] = (devcom_meta['PID'].str.extract(r'-(\d+)$')[0]
                              .astype(float).astype('Int64'))
    per_run = (devcom_meta.groupby('_run_id')['firing_accuracy_AOBJ_s3']
               .mean()
               .reindex(run_ids))

    # Runs with no D3 score (currently 12 and 16) are forced to train
    scored_mask  = per_run.notna().values
    scored_ids   = np.array(run_ids)[scored_mask]
    unscored_ids = np.array(run_ids)[~scored_mask]

    devcom_median = per_run.dropna().median()
    scored_labels = (per_run.dropna().values >= devcom_median).astype(int)

    # DEVCOM permutation calls MUST precede FAB calls — do NOT reorder
    rng = np.random.default_rng(SPLIT_SEED)

    # Stratify the 38 scored runs: 26 train, 6 val, 6 test
    d_train_scored, d_val, d_test = _stratified_split(
        scored_ids, scored_labels, n_val=6, n_test=6, rng=rng)
    # Append unscored runs to train (they have no probe labels)
    d_train = sorted(d_train_scored + unscored_ids.tolist())

    devcom = {'train': d_train, 'val': d_val, 'test': d_test}

    # ── FAB ───────────────────────────────────────────────────────────────────
    fab_dir  = DATA_ROOT / "aligned" / "target_FAB"
    fab_csv  = DATA_ROOT / "target_objective_FAB" / "metadataFAB.csv"
    fab_meta = pd.read_csv(fab_csv)

    meta_pids = set(fab_meta["PID"].dropna().astype(str))
    file_pids = {m.group(1)
                 for f in fab_dir.glob("FAB_*_ego.parquet")
                 if (m := _FAB_PID_RE.match(f.name))}
    pids = sorted(meta_pids & file_pids)

    # portScore is constant per PID (same across tasks A and B)
    pid_score = (fab_meta.set_index('PID')['portScore']
                 .reindex(pids)
                 .astype(float))
    fab_median = pid_score.median()
    fab_labels = (pid_score.values >= fab_median).astype(int)

    f_train, f_val, f_test = _stratified_split(
        np.array(pids), fab_labels, n_val=14, n_test=15, rng=rng)

    fab = {'train': f_train, 'val': f_val, 'test': f_test}
    return devcom, fab


def _load_split_manifest(path: Path = MANIFEST_PATH) -> tuple[dict, dict]:
    with open(path) as f:
        manifest = json.load(f)

    if manifest.get("seed") != SPLIT_SEED:
        raise ValueError(
            f"Split manifest seed {manifest.get('seed')} does not match "
            f"SPLIT_SEED={SPLIT_SEED}"
        )

    devcom = {k: sorted(map(int, manifest["devcom"][k]))
              for k in ("train", "val", "test")}
    fab = {k: sorted(map(str, manifest["fab"][k]))
           for k in ("train", "val", "test")}
    return devcom, fab


DEVCOM_SPLITS, FAB_SPLITS = _load_split_manifest()

_DEVCOM_RUN_TO_SPLIT: dict[int, str] = {
    run_id: split
    for split, ids in DEVCOM_SPLITS.items()
    for run_id in ids
}
_FAB_PID_TO_SPLIT: dict[str, str] = {
    pid: split
    for split, pids in FAB_SPLITS.items()
    for pid in pids
}


# ---------------------------------------------------------------------------
# Public lookup helpers
# ---------------------------------------------------------------------------

def get_devcom_split(run_id: int) -> str:
    """Return 'train', 'val', 'test', or 'unknown' for a DEVCOM run ID."""
    return _DEVCOM_RUN_TO_SPLIT.get(run_id, 'unknown')


def get_fab_split(pid: str) -> str:
    """Return 'train', 'val', 'test', or 'unknown' for a FAB participant ID."""
    return _FAB_PID_TO_SPLIT.get(pid, 'unknown')


# ---------------------------------------------------------------------------
# File-list filtering
# ---------------------------------------------------------------------------

def filter_devcom_files(files: list[Path], split_keys: list[str]) -> list[Path]:
    """Return only the files whose run_id belongs to one of the given splits."""
    target_ids: set[int] = set()
    for k in split_keys:
        target_ids.update(DEVCOM_SPLITS.get(k, []))
    out = []
    for f in files:
        m = _DEVCOM_FILE_RE.match(f.name)
        if m and int(m.group(1)) in target_ids:
            out.append(f)
    return out


def filter_fab_files(files: list[Path], split_keys: list[str]) -> list[Path]:
    """Return only the files whose PID belongs to one of the given splits."""
    target_pids: set[str] = set()
    for k in split_keys:
        target_pids.update(FAB_SPLITS.get(k, []))
    out = []
    for f in files:
        m = _FAB_PID_RE.match(f.name)
        if m and m.group(1) in target_pids:
            out.append(f)
    return out


# ---------------------------------------------------------------------------
# DataFrame annotation helpers
# ---------------------------------------------------------------------------

def _fab_pid_from_stem(stem: str) -> str | None:
    m = _FAB_PID_RE.match(stem) or re.match(r"FAB_(\w+)_Build[AB]_", stem)
    return m.group(1) if m else None


def _devcom_run_id_from_stem(stem: str) -> int | None:
    m = _DEVCOM_FILE_RE.match(stem) or re.match(r"DEVCOM_run_(\d+)_session_", stem)
    return int(m.group(1)) if m else None


def annotate_fab_df(df: pd.DataFrame, filename_col: str = 'filename') -> pd.DataFrame:
    """Add a '_split' column to df based on FAB PID extracted from filename_col."""
    df = df.copy()
    df['_split'] = df[filename_col].apply(
        lambda fn: get_fab_split(_fab_pid_from_stem(fn) or '')
    )
    return df


def annotate_devcom_df(df: pd.DataFrame, filename_col: str = 'filename') -> pd.DataFrame:
    """Add a '_split' column to df based on DEVCOM run_id extracted from filename_col."""
    df = df.copy()
    def _lookup(fn):
        rid = _devcom_run_id_from_stem(fn)
        return get_devcom_split(rid) if rid is not None else 'unknown'
    df['_split'] = df[filename_col].apply(_lookup)
    return df


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    devcom_csv = DATA_ROOT / "target_objective_D3" / "DEVCOM_s3_metrics.csv"
    dm = pd.read_csv(devcom_csv)
    dm['_run_id'] = dm['PID'].str.extract(r'-(\d+)$')[0].astype(float).astype('Int64')
    per_run = dm.groupby('_run_id')['firing_accuracy_AOBJ_s3'].mean()

    print("DEVCOM splits (stratified on firing_accuracy_AOBJ_s3):")
    for k, v in DEVCOM_SPLITS.items():
        scored = [r for r in v if r in per_run.index]
        if scored:
            accs = per_run.reindex(scored)
            print(f"  {k:5s}: {len(v):2d} runs  "
                  f"firing_acc mean={accs.mean():.3f}  min={accs.min():.3f}  max={accs.max():.3f}")
        else:
            print(f"  {k:5s}: {len(v):2d} runs")

    fab_csv = DATA_ROOT / "target_objective_FAB" / "metadataFAB.csv"
    fm = pd.read_csv(fab_csv).set_index('PID')['portScore']
    print("\nFAB splits (stratified on portScore):")
    for k, v in FAB_SPLITS.items():
        scores = fm.reindex(v).dropna()
        print(f"  {k:5s}: {len(v):2d} participants  "
              f"portScore mean={scores.mean():.2f}  "
              f"class_0={(scores < fm.median()).sum()}  "
              f"class_1={(scores >= fm.median()).sum()}")
