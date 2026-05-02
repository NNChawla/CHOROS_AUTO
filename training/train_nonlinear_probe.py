"""
Nonlinear probe over CHOROS sequence embeddings -> target label (RBF SVM).

Objectives:
  FAB  – portScore (0-16) from data/target_objective_FAB/metadataFAB.csv
         aligned data: data/aligned/target_FAB/
  D2   – per-metric columns from data/target_objective_D2/DEVCOM_s2_metrics.csv
         aligned data: data/aligned/target_DEVCOM_s2/
  D3   – per-metric columns from data/target_objective_D3/DEVCOM_s3_metrics.csv
         aligned data: data/aligned/target_DEVCOM_s3/

Modes:
  regression   – raw score (SVR, RBF kernel)
  median       – binary: above vs. below median (SVC, RBF kernel)
  quartile     – 3-class: bottom Q1, middle Q2-Q3, top Q4 (SVC, RBF kernel)

Embeddings: outputs/embeddings/<run>/sequence_embeddings.npy  (N x 128)
"""

import re
import warnings
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (LeaveOneOut, LeaveOneGroupOut,
                                     StratifiedKFold, cross_val_predict, cross_val_score)
from sklearn.metrics import (r2_score, mean_absolute_error,
                             accuracy_score, balanced_accuracy_score,
                             roc_auc_score, classification_report)
from scipy.stats import pearsonr, spearmanr

# ── args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--emb-dir", type=str, default=None,
                    help="Name of embedding run directory under outputs/embeddings/ "
                         "(default: most recent)")
parser.add_argument("--objective", type=str, default="FAB",
                    choices=["FAB", "D2", "D3"],
                    help="Which target objective to probe (default: FAB)")
parser.add_argument("--target-col", type=str, default=None,
                    help="Target column for D2/D3 objectives. "
                         "Defaults to first metric column. "
                         "Pass 'all' to iterate over every metric column.")
args = parser.parse_args()

# ── objective config ──────────────────────────────────────────────────────────
_ROOT   = Path(__file__).parent.parent
EMB_DIR = _ROOT / "outputs" / "embeddings"
RUN_DIR = EMB_DIR / args.emb_dir if args.emb_dir else sorted(EMB_DIR.iterdir())[-1]
EMB_CSV = RUN_DIR / "sequence_embeddings.csv"

_DEVCOM_ROLE_ABBR = {
    "BravoGrenadier": "BG",
    "BravoLeader":    "BL",
    "BravoRifleman":  "BR",
    "BravoSaw":       "BS",
}

def _devcom_parse_key(session_pat, fn):
    m = re.match(rf"DEVCOM_run_(\d+)_{session_pat}_(Bravo\w+)_", fn)
    if not m:
        return None
    abbr = _DEVCOM_ROLE_ABBR.get(m.group(2))
    return f"{abbr}-{m.group(1)}" if abbr else None

def _devcom_parse_run(session_pat, fn):
    m = re.match(rf"DEVCOM_run_(\d+)_{session_pat}_", fn)
    return int(m.group(1)) if m else None

OBJECTIVE_CFG = {
    "FAB": {
        "csv":        _ROOT / "data" / "target_objective_FAB" / "metadataFAB.csv",
        "aligned":    _ROOT / "data" / "aligned" / "target_FAB",
        "join_key":   "PID_cohort",
        "target_cols": ["portScore"],
        "parse_key":  lambda fn: (
            lambda m: f"{m.group(1)}_{m.group(2)}" if m else None
        )(re.match(r"FAB_(\w+)_Build([AB])_", fn)),
    },
    "D2": {
        "csv":        _ROOT / "data" / "target_objective_D2" / "DEVCOM_s2_metrics.csv",
        "aligned":    _ROOT / "data" / "aligned" / "target_DEVCOM_s2",
        "join_key":   "PID",
        "target_cols": None,
        "parse_key":  lambda fn: _devcom_parse_key("session_2", fn),
        "parse_run":  lambda fn: _devcom_parse_run("session_2", fn),
    },
    "D3": {
        "csv":        _ROOT / "data" / "target_objective_D3" / "DEVCOM_s3_metrics.csv",
        "aligned":    _ROOT / "data" / "aligned" / "target_DEVCOM_s3",
        "join_key":   "PID",
        "target_cols": None,
        "parse_key":  lambda fn: _devcom_parse_key("session_[23]", fn),
        "parse_run":  lambda fn: _devcom_parse_run("session_[23]", fn),
    },
}

cfg = OBJECTIVE_CFG[args.objective]

print(f"Using embeddings from : {RUN_DIR.name}")
print(f"Objective             : {args.objective}")

# ── load metadata ─────────────────────────────────────────────────────────────
meta = pd.read_csv(cfg["csv"])
meta.columns = meta.columns.str.strip()

join_key = cfg["join_key"]
if cfg["target_cols"] is None:
    cfg["target_cols"] = [c for c in meta.columns if c != join_key]

if args.objective == "FAB":
    target_cols_to_run = cfg["target_cols"]
elif args.target_col is None:
    target_cols_to_run = [cfg["target_cols"][0]]
elif args.target_col == "all":
    target_cols_to_run = cfg["target_cols"]
else:
    if args.target_col not in cfg["target_cols"]:
        raise ValueError(f"--target-col '{args.target_col}' not found. "
                         f"Available: {cfg['target_cols']}")
    target_cols_to_run = [args.target_col]

# ── load & join embeddings ────────────────────────────────────────────────────
emb_csv = pd.read_csv(EMB_CSV)
emb_csv = emb_csv.assign(**{join_key: emb_csv["filename"].apply(cfg["parse_key"])})
emb_csv = emb_csv.dropna(subset=[join_key])

feat_cols = [c for c in emb_csv.columns if c.startswith("e") and c[1:].isdigit()]

# ── hyperparameter grids ──────────────────────────────────────────────────────
Cs_grid = np.logspace(-1, 3, 15)

# ── helpers ───────────────────────────────────────────────────────────────────

def _select_svr_logo(X, y_raw, groups):
    logo = LeaveOneGroupOut()
    best_C, best_score = Cs_grid[0], -np.inf
    for C in Cs_grid:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_val_score(SVR(kernel="rbf", C=C), X, y_raw,
                                     cv=logo, groups=groups, scoring="r2")
        if scores.mean() > best_score:
            best_score = scores.mean()
            best_C = C
    return best_C


def _select_svr_loo(X, y_raw):
    best_C, best_score = Cs_grid[0], -np.inf
    for C in Cs_grid:
        scores = cross_val_score(SVR(kernel="rbf", C=C), X, y_raw,
                                 cv=LeaveOneOut(), scoring="r2")
        if scores.mean() > best_score:
            best_score = scores.mean()
            best_C = C
    return best_C


def _select_svc_logo(X, y_cls, groups):
    logo = LeaveOneGroupOut()
    best_C, best_score = Cs_grid[0], -np.inf
    for C in Cs_grid:
        model = SVC(kernel="rbf", C=C, class_weight="balanced", random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            scores = cross_val_score(model, X, y_cls, cv=logo, groups=groups,
                                     scoring="balanced_accuracy")
        if scores.mean() > best_score:
            best_score = scores.mean()
            best_C = C
    return best_C


def _select_svc_loo(X, y_cls):
    actual_classes = np.unique(y_cls)
    n_splits = min(5, min((y_cls == c).sum() for c in actual_classes))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    best_C, best_score = Cs_grid[0], -np.inf
    for C in Cs_grid:
        model = SVC(kernel="rbf", C=C, class_weight="balanced", random_state=42)
        scores = cross_val_score(model, X, y_cls, cv=skf,
                                 scoring="balanced_accuracy")
        if scores.mean() > best_score:
            best_score = scores.mean()
            best_C = C
    return best_C


def run_regression(X, y_raw, groups=None):
    if groups is not None:
        best_C = _select_svr_logo(X, y_raw, groups)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_pred = cross_val_predict(SVR(kernel="rbf", C=best_C), X, y_raw,
                                       cv=LeaveOneGroupOut(), groups=groups)
        cv_label = "LOGO"
    else:
        best_C = _select_svr_loo(X, y_raw)
        y_pred = cross_val_predict(SVR(kernel="rbf", C=best_C), X, y_raw,
                                   cv=LeaveOneOut())
        cv_label = "LOO"

    r2    = r2_score(y_raw, y_pred)
    mae   = mean_absolute_error(y_raw, y_pred)
    pear  = pearsonr(y_raw, y_pred)
    spear = spearmanr(y_raw, y_pred)

    print(f"\n── Regression {cv_label} Results ───────────────────────────────────────────────")
    print(f"  Best C          : {best_C:.4f}")
    print(f"  R²              : {r2:.4f}")
    print(f"  MAE             : {mae:.4f}")
    print(f"  Pearson  r      : {pear.statistic:.4f}  (p={pear.pvalue:.4e})")
    print(f"  Spearman rho    : {spear.statistic:.4f}  (p={spear.pvalue:.4e})")
    return y_pred


def run_classification(label, X, y_cls, n_classes, groups=None):
    actual_classes = np.unique(y_cls)
    actual_n = len(actual_classes)

    if actual_n < 2:
        print(f"\n── {label} SKIPPED (only {actual_n} class present) ──────────────────────")
        return np.full_like(y_cls, actual_classes[0]), None

    if actual_n < n_classes:
        print(f"  Warning: expected {n_classes} classes, found {actual_n} — treating as {actual_n}-class")

    if groups is not None:
        best_C = _select_svc_logo(X, y_cls, groups)
        cv_splitter = LeaveOneGroupOut()
        cv_label = "LOGO"
    else:
        best_C = _select_svc_loo(X, y_cls)
        cv_splitter = LeaveOneOut()
        cv_label = "LOO"

    model = SVC(kernel="rbf", C=best_C, class_weight="balanced",
                probability=True, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        y_pred = cross_val_predict(model, X, y_cls, cv=cv_splitter, groups=groups)
        y_prob = cross_val_predict(model, X, y_cls, cv=cv_splitter, groups=groups,
                                   method="predict_proba")

    acc  = accuracy_score(y_cls, y_pred)
    bacc = balanced_accuracy_score(y_cls, y_pred)
    if actual_n == 2:
        auc = roc_auc_score(y_cls, y_prob[:, 1])
        auc_str = f"  ROC-AUC         : {auc:.4f}"
    else:
        auc = roc_auc_score(y_cls, y_prob, multi_class="ovr", average="macro")
        auc_str = f"  ROC-AUC (macro) : {auc:.4f}"

    print(f"\n── {label} {cv_label} Results ──────────────────────────────────────────────────")
    print(f"  Best C          : {best_C:.4f}")
    print(f"  Accuracy        : {acc:.4f}")
    print(f"  Balanced Acc.   : {bacc:.4f}")
    print(auc_str)
    print(f"\n  Classification report ({cv_label}):")
    print(classification_report(y_cls, y_pred, labels=actual_classes, digits=3))

    return y_pred, y_prob


# ── probe each target column ──────────────────────────────────────────────────
all_outputs = []

for target_col in target_cols_to_run:
    print(f"\n{'='*72}")
    print(f"Target column: {target_col}")
    print(f"{'='*72}")

    keep_cols = [join_key] + [target_col]
    merged = emb_csv.merge(meta[keep_cols], on=join_key, how="inner")
    merged = merged.dropna(subset=[target_col])

    print(f"Matched samples : {len(merged)}  (embeddings={len(emb_csv)}, metadata={len(meta)})")

    if len(merged) == 0:
        print("  No matched samples — skipping this target column.")
        continue

    X_raw = merged[feat_cols].values.astype(np.float32)
    y_raw = merged[target_col].values.astype(np.float32)

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    if "parse_run" in cfg:
        groups = merged["filename"].apply(cfg["parse_run"]).values
        n_groups = len(np.unique(groups))
        print(f"CV strategy      : Leave-One-Group-Out  ({n_groups} run groups)")
    else:
        groups = None
        print(f"CV strategy      : Leave-One-Out")

    # 1. Regression
    y_pred_reg = run_regression(X, y_raw, groups=groups)

    # 2. Median split (binary)
    median = np.median(y_raw)
    y_med  = (y_raw >= median).astype(int)
    print(f"\nMedian threshold : {median:.3f}  →  low={(y_med==0).sum()}  high={(y_med==1).sum()}")
    pred_med, prob_med = run_classification("Median-split (binary)", X, y_med, n_classes=2, groups=groups)

    # 3. Quartile split (3-class)
    q1 = np.percentile(y_raw, 25)
    q3 = np.percentile(y_raw, 75)
    y_qrt = np.where(y_raw <= q1, 0, np.where(y_raw <= q3, 1, 2))
    print(f"\nQuartile thresholds: Q1≤{q1:.3f} / Q2-Q3 / Q4>{q3:.3f}")
    print(f"  Class counts: low={(y_qrt==0).sum()}  mid={(y_qrt==1).sum()}  high={(y_qrt==2).sum()}")
    pred_qrt, prob_qrt = run_classification("Quartile-split (3-class)", X, y_qrt, n_classes=3, groups=groups)

    out = merged[[join_key, target_col]].copy()
    out["pred_regression"] = y_pred_reg
    out["label_median"]    = y_med
    out["pred_median"]     = pred_med
    out["label_quartile"]  = y_qrt
    out["pred_quartile"]   = pred_qrt
    out["target_col"]      = target_col
    all_outputs.append(out)

# ── save predictions ──────────────────────────────────────────────────────────
out_dir = _ROOT / "outputs" / "predictions"
out_dir.mkdir(parents=True, exist_ok=True)

combined = pd.concat(all_outputs, ignore_index=True)
out_path = out_dir / f"nonlinear_probe_{args.objective}_{RUN_DIR.name}.csv"
combined.to_csv(out_path, index=False)
print(f"\nPredictions saved to: {out_path}")
