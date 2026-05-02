"""
Linear probe over CHOROS sequence embeddings -> target label.

Objectives:
  FAB  – portScore (0-16) from data/target_objective_FAB/metadataFAB.csv
         aligned data: data/aligned/target_FAB/
  D2   – per-metric columns from data/target_objective_D2/DEVCOM_s2_metrics.csv
         aligned data: data/aligned/target_DEVCOM_s2/
  D3   – per-metric columns from data/target_objective_D3/DEVCOM_s3_metrics.csv
         aligned data: data/aligned/target_DEVCOM_s3/

Modes:
  regression   – raw score (Ridge)
  median       – binary: above vs. below median (LogisticRegression)
  quartile     – 3-class: bottom Q1, middle Q2-Q3, top Q4 (LogisticRegression)

Embeddings: outputs/embeddings/<run>/sequence_embeddings.npy  (N x 128)
"""

import re
import warnings
import argparse
import os
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from sklearn.linear_model import Ridge, RidgeCV, LogisticRegressionCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (LeaveOneOut, LeaveOneGroupOut,
                                     StratifiedKFold, cross_val_predict, cross_val_score)
from sklearn.metrics import (r2_score, mean_absolute_error,
                             accuracy_score, balanced_accuracy_score,
                             roc_auc_score, classification_report,
                             matthews_corrcoef, f1_score)
from scipy.stats import pearsonr, spearmanr

_ROOT     = Path(__file__).parent.parent
DATA_ROOT = Path(os.environ.get("CHOROS_DATA_ROOT", "/srv/CHOROS/data"))
EMB_DIR   = _ROOT / "outputs" / "embeddings"

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
        "csv":        DATA_ROOT / "target_objective_FAB" / "metadataFAB.csv",
        "aligned":    DATA_ROOT / "aligned" / "target_FAB",
        "join_key":   "PID_cohort",
        "target_cols": ["portScore"],
        "parse_key":  lambda fn: (
            lambda m: f"{m.group(1)}_{m.group(2)}" if m else None
        )(re.match(r"FAB_(\w+)_Build([AB])_", fn)),
    },
    "D2": {
        "csv":        DATA_ROOT / "target_objective_D2" / "DEVCOM_s2_metrics.csv",
        "aligned":    DATA_ROOT / "aligned" / "target_DEVCOM_s2",
        "join_key":   "PID",
        "target_cols": None,
        "parse_key":  lambda fn: _devcom_parse_key("session_2", fn),
        "parse_run":  lambda fn: _devcom_parse_run("session_2", fn),
    },
    "D3": {
        "csv":        DATA_ROOT / "target_objective_D3" / "DEVCOM_s3_metrics.csv",
        "aligned":    DATA_ROOT / "aligned" / "target_DEVCOM_s3",
        "join_key":   "PID",
        "target_cols": None,
        "parse_key":  lambda fn: _devcom_parse_key("session_[23]", fn),
        "parse_run":  lambda fn: _devcom_parse_run("session_[23]", fn),
    },
}

alphas  = np.logspace(-2, 4, 50)
Cs_grid = np.logspace(-3, 3, 20)


def _score_alpha(alpha, X, y_raw, groups, logo):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cross_val_score(Ridge(alpha=alpha), X, y_raw,
                               cv=logo, groups=groups, scoring="r2").mean()

def _select_alpha_logo(X, y_raw, groups):
    logo = LeaveOneGroupOut()
    scores = Parallel(n_jobs=-1)(
        delayed(_score_alpha)(a, X, y_raw, groups, logo) for a in alphas
    )
    return alphas[int(np.argmax(scores))]


def _score_C(C, X, y_cls, groups, logo):
    model = LogisticRegression(C=C, max_iter=2000, class_weight="balanced",
                               solver="lbfgs", random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return cross_val_score(model, X, y_cls, cv=logo, groups=groups,
                               scoring="balanced_accuracy").mean()

def _select_C_logo(X, y_cls, groups):
    logo = LeaveOneGroupOut()
    scores = Parallel(n_jobs=-1)(
        delayed(_score_C)(C, X, y_cls, groups, logo) for C in Cs_grid
    )
    return Cs_grid[int(np.argmax(scores))]


def run_regression(X, y_raw, groups=None):
    if groups is not None:
        logo = LeaveOneGroupOut()
        best_alpha = _select_alpha_logo(X, y_raw, groups)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_pred = cross_val_predict(Ridge(alpha=best_alpha), X, y_raw,
                                       cv=logo, groups=groups)
        cv_label = "LOGO"
    else:
        ridge_cv = RidgeCV(alphas=alphas, cv=5, scoring="r2")
        ridge_cv.fit(X, y_raw)
        best_alpha = ridge_cv.alpha_
        y_pred = cross_val_predict(Ridge(alpha=best_alpha), X, y_raw,
                                   cv=LeaveOneOut(), n_jobs=-1)
        cv_label = "LOO"

    r2    = r2_score(y_raw, y_pred)
    mae   = mean_absolute_error(y_raw, y_pred)
    pear  = pearsonr(y_raw, y_pred)
    spear = spearmanr(y_raw, y_pred)

    print(f"\n── Regression {cv_label} Results ───────────────────────────────────────────────")
    print(f"  Best alpha      : {best_alpha:.4f}")
    print(f"  R²              : {r2:.4f}")
    print(f"  MAE             : {mae:.4f}")
    print(f"  Pearson  r      : {pear.statistic:.4f}  (p={pear.pvalue:.4e})")
    print(f"  Spearman rho    : {spear.statistic:.4f}  (p={spear.pvalue:.4e})")
    return y_pred, {
        'r2': r2, 'mae': mae,
        'pearson_r': pear.statistic, 'pearson_p': pear.pvalue,
        'spearman_r': spear.statistic, 'spearman_p': spear.pvalue,
    }


def run_classification(label, X, y_cls, n_classes, groups=None):
    actual_classes = np.unique(y_cls)
    actual_n = len(actual_classes)

    if actual_n < 2:
        print(f"\n── {label} SKIPPED (only {actual_n} class present) ──────────────────────")
        return np.full_like(y_cls, actual_classes[0]), None

    if actual_n < n_classes:
        print(f"  Warning: expected {n_classes} classes, found {actual_n} — treating as {actual_n}-class")

    if groups is not None:
        cv_splitter = LeaveOneGroupOut()
        best_C = _select_C_logo(X, y_cls, groups)
        cv_label = "LOGO"
    else:
        skf = StratifiedKFold(n_splits=min(5, actual_n * min((y_cls == c).sum() for c in actual_classes)),
                              shuffle=True, random_state=42)
        clf = LogisticRegressionCV(
            Cs=20, cv=skf, max_iter=2000,
            scoring="balanced_accuracy",
            class_weight="balanced",
            solver="lbfgs",
            n_jobs=-1,
            random_state=42,
        )
        clf.fit(X, y_cls)
        best_C = clf.C_[0]
        cv_splitter = LeaveOneOut()
        cv_label = "LOO"

    model = LogisticRegression(
        C=best_C, max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        y_prob = cross_val_predict(model, X, y_cls, cv=cv_splitter, groups=groups,
                                   method="predict_proba", n_jobs=-1)
    classes = np.unique(y_cls)
    y_pred = classes[y_prob.argmax(axis=1)]

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


def run_regression_split(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    groups_train: np.ndarray | None,
    eval_split_name: str,
) -> tuple[np.ndarray, dict]:
    """Fit Ridge on train (alpha selected by LOGO-CV or 5-fold), evaluate on eval."""
    if groups_train is not None:
        best_alpha = _select_alpha_logo(X_train, y_train, groups_train)
        cv_label = "LOGO"
    else:
        rc = RidgeCV(alphas=alphas, cv=5, scoring="r2")
        rc.fit(X_train, y_train)
        best_alpha = rc.alpha_
        cv_label = "5-fold"

    model = Ridge(alpha=best_alpha)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_eval)

    r2    = r2_score(y_eval, y_pred)
    mae   = mean_absolute_error(y_eval, y_pred)
    pear  = pearsonr(y_eval, y_pred)
    spear = spearmanr(y_eval, y_pred)

    print(f"\n── Regression [{eval_split_name}] (alpha via {cv_label} on train) ─────────────")
    print(f"  Best alpha      : {best_alpha:.4f}")
    print(f"  R²              : {r2:.4f}")
    print(f"  MAE             : {mae:.4f}")
    print(f"  Pearson  r      : {pear.statistic:.4f}  (p={pear.pvalue:.4e})")
    print(f"  Spearman rho    : {spear.statistic:.4f}  (p={spear.pvalue:.4e})")
    return y_pred, {
        'r2': r2, 'mae': mae,
        'pearson_r': pear.statistic, 'pearson_p': pear.pvalue,
        'spearman_r': spear.statistic, 'spearman_p': spear.pvalue,
    }


def run_classification_split(
    label: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    n_classes: int,
    groups_train: np.ndarray | None,
    eval_split_name: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Fit logistic probe on train (C selected by LOGO-CV or stratified k-fold),
    evaluate on eval.  Returns (y_pred, y_prob, metrics_dict).
    """
    actual_train_classes = np.unique(y_train)
    actual_eval_classes  = np.unique(y_eval)
    all_classes          = np.union1d(actual_train_classes, actual_eval_classes)
    actual_n             = len(all_classes)

    if len(actual_train_classes) < 2:
        print(f"\n── {label} [{eval_split_name}] SKIPPED (only {len(actual_train_classes)} class in train) ──")
        dummy = np.full(len(y_eval), actual_train_classes[0])
        return dummy, None, {'bacc': float('nan'), 'mcc': float('nan'),
                             'f1': float('nan'), 'auc': float('nan')}

    if groups_train is not None:
        best_C   = _select_C_logo(X_train, y_train, groups_train)
        cv_label = "LOGO"
    else:
        n_splits = min(5, int(min(np.bincount(y_train.astype(int)))))
        n_splits = max(2, n_splits)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        clf = LogisticRegressionCV(
            Cs=20, cv=skf, max_iter=2000,
            scoring="balanced_accuracy",
            class_weight="balanced",
            solver="lbfgs",
            n_jobs=-1,
            random_state=42,
        )
        clf.fit(X_train, y_train)
        best_C   = clf.C_[0]
        cv_label = "StratKF"

    model = LogisticRegression(
        C=best_C, max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_eval)
    # model.classes_ may differ from all_classes if some classes missing in train
    classes_seen = model.classes_
    y_pred = classes_seen[y_prob.argmax(axis=1)]

    bacc = balanced_accuracy_score(y_eval, y_pred)
    mcc  = matthews_corrcoef(y_eval, y_pred)
    f1   = f1_score(y_eval, y_pred, average='macro', zero_division=0)

    if actual_n == 2:
        col_idx = list(classes_seen).index(1) if 1 in classes_seen else 0
        auc = roc_auc_score(y_eval, y_prob[:, col_idx]) if len(np.unique(y_eval)) == 2 else float('nan')
        auc_str = f"  ROC-AUC         : {auc:.4f}"
    else:
        try:
            auc = roc_auc_score(y_eval, y_prob, multi_class="ovr", average="macro",
                                labels=classes_seen)
        except ValueError:
            auc = float('nan')
        auc_str = f"  ROC-AUC (macro) : {auc:.4f}"

    acc = accuracy_score(y_eval, y_pred)
    print(f"\n── {label} [{eval_split_name}] (C via {cv_label} on train) ──────────────────────")
    print(f"  Best C          : {best_C:.4f}")
    print(f"  Accuracy        : {acc:.4f}")
    print(f"  Balanced Acc.   : {bacc:.4f}")
    print(f"  MCC             : {mcc:.4f}")
    print(f"  F1 (macro)      : {f1:.4f}")
    print(auc_str)
    print(f"\n  Classification report (train→{eval_split_name}):")
    print(classification_report(y_eval, y_pred, labels=all_classes, digits=3))

    return y_pred, y_prob, {'bacc': bacc, 'mcc': mcc, 'f1': f1, 'auc': auc}


def run_probe(
    emb_dir: str | None,
    objective: str = "FAB",
    target_col: str | None = None,
    train_split: str | None = None,
    eval_split: str | None = None,
) -> dict:
    """
    Run linear probes on a sequence-embedding directory.

    Parameters
    ----------
    emb_dir      : absolute path or name under outputs/embeddings/
    objective    : "FAB", "D2", or "D3"
    target_col   : specific column to probe, "all", or None (→ first / all for FAB)
    train_split  : 'train' or 'train+val' — activates split-aware mode
    eval_split   : 'val' or 'test' — split to evaluate on (required with train_split)

    Returns
    -------
    {target_col: {'r2', 'mae', 'pearson_r', 'pearson_p',
                  'spearman_r', 'spearman_p', 'n_samples'}}
    """
    if (train_split is None) != (eval_split is None):
        raise ValueError("--train_split and --eval_split must be provided together")

    emb_path = Path(emb_dir) if emb_dir else None
    if emb_path is None or not emb_path.is_absolute():
        RUN_DIR = EMB_DIR / emb_dir if emb_dir else sorted(EMB_DIR.iterdir())[-1]
    else:
        RUN_DIR = emb_path
    EMB_CSV = RUN_DIR / "sequence_embeddings.csv"

    cfg = dict(OBJECTIVE_CFG[objective])   # shallow copy so we can mutate target_cols

    print(f"Using embeddings from : {RUN_DIR.name}")
    print(f"Objective             : {objective}")

    meta = pd.read_csv(cfg["csv"])
    meta.columns = meta.columns.str.strip()

    join_key = cfg["join_key"]
    if cfg["target_cols"] is None:
        cfg["target_cols"] = [c for c in meta.columns if c != join_key]

    if objective == "FAB":
        target_cols_to_run = cfg["target_cols"]
    elif target_col is None:
        target_cols_to_run = [cfg["target_cols"][0]]
    elif target_col == "all":
        target_cols_to_run = cfg["target_cols"]
    else:
        if target_col not in cfg["target_cols"]:
            raise ValueError(f"--target-col '{target_col}' not found. "
                             f"Available: {cfg['target_cols']}")
        target_cols_to_run = [target_col]

    emb_csv = pd.read_csv(EMB_CSV)
    new_col = emb_csv["filename"].apply(cfg["parse_key"]).rename(join_key)
    emb_csv = pd.concat([emb_csv, new_col], axis=1).dropna(subset=[join_key])

    feat_cols = [c for c in emb_csv.columns if c.startswith("e") and c[1:].isdigit()]

    split_mode = bool(train_split and eval_split)
    if split_mode:
        import sys as _sys
        _training_dir = str(_ROOT / 'training')
        if _training_dir not in _sys.path:
            _sys.path.insert(0, _training_dir)
        from splits import annotate_fab_df, annotate_devcom_df
        if objective == "FAB":
            emb_csv = annotate_fab_df(emb_csv)
        else:
            emb_csv = annotate_devcom_df(emb_csv)
        train_keys = set(train_split.split('+'))

    all_outputs = []
    results: dict = {}

    for tcol in target_cols_to_run:
        print(f"\n{'='*72}")
        print(f"Target column: {tcol}")
        print(f"{'='*72}")

        keep_cols = [join_key] + [tcol]
        merged = emb_csv.merge(meta[keep_cols], on=join_key, how="inner")
        merged = merged.dropna(subset=[tcol])

        print(f"Matched samples : {len(merged)}  (embeddings={len(emb_csv)}, metadata={len(meta)})")

        if len(merged) == 0:
            print("  No matched samples — skipping this target column.")
            continue

        target_display = tcol
        if objective == "FAB" and tcol == "portScore":
            target_display = "expertiseScore"

        if split_mode:
            # Re-annotate merged (join may have changed index)
            if objective == "FAB":
                merged = annotate_fab_df(merged)
            else:
                merged = annotate_devcom_df(merged)

            merged_train = merged[merged['_split'].isin(train_keys)].copy()
            merged_eval  = merged[merged['_split'] == eval_split].copy()

            print(f"  Train samples : {len(merged_train)}  "
                  f"Eval ({eval_split}) samples : {len(merged_eval)}")

            if len(merged_train) == 0 or len(merged_eval) == 0:
                print("  Insufficient split data — skipping this target column.")
                continue

            X_train_raw = merged_train[feat_cols].values.astype(np.float32)
            X_eval_raw  = merged_eval[feat_cols].values.astype(np.float32)
            y_train_raw = merged_train[tcol].values.astype(np.float32)
            y_eval_raw  = merged_eval[tcol].values.astype(np.float32)

            if objective == "FAB" and tcol == "portScore":
                y_train_raw = 16.0 - y_train_raw
                y_eval_raw  = 16.0 - y_eval_raw
                print(f"  (FAB: portScore inverted → expertiseScore; higher = more expert)")

            # Scaler fit on train only
            scaler  = StandardScaler()
            X_train = scaler.fit_transform(X_train_raw)
            X_eval  = scaler.transform(X_eval_raw)

            if "parse_run" in cfg:
                groups_train = merged_train["filename"].apply(cfg["parse_run"]).values
                n_groups = len(np.unique(groups_train))
                print(f"C/alpha selection: LOGO-CV on train  ({n_groups} run groups)")
            else:
                groups_train = None
                print(f"C/alpha selection: StratKF on train")

            # Regression
            y_pred_reg, reg_metrics = run_regression_split(
                X_train, y_train_raw, X_eval, y_eval_raw, groups_train, eval_split)
            results[target_display] = {**reg_metrics, 'n_samples': len(merged_eval)}

            # Median-split classification (population median across train+eval)
            median = np.median(np.concatenate([y_train_raw, y_eval_raw]))
            y_train_med = (y_train_raw >= median).astype(int)
            y_eval_med  = (y_eval_raw  >= median).astype(int)
            if objective == "FAB" and tcol == "portScore":
                print(f"\nMedian (population): {median:.3f}  → "
                      f"train novice={(y_train_med==0).sum()} expert={(y_train_med==1).sum()}  "
                      f"eval novice={(y_eval_med==0).sum()} expert={(y_eval_med==1).sum()}")
            else:
                print(f"\nMedian (population): {median:.3f}  → "
                      f"train low={(y_train_med==0).sum()} high={(y_train_med==1).sum()}  "
                      f"eval low={(y_eval_med==0).sum()} high={(y_eval_med==1).sum()}")

            pred_med, prob_med, med_metrics = run_classification_split(
                "Median-split (binary)", X_train, y_train_med,
                X_eval, y_eval_med, n_classes=2, groups_train=groups_train,
                eval_split_name=eval_split)

            # Quartile-split classification (thresholds from train)
            q1 = np.percentile(y_train_raw, 25)
            q3 = np.percentile(y_train_raw, 75)
            y_train_qrt = np.where(y_train_raw <= q1, 0, np.where(y_train_raw <= q3, 1, 2))
            y_eval_qrt  = np.where(y_eval_raw  <= q1, 0, np.where(y_eval_raw  <= q3, 1, 2))
            print(f"\nQuartile (train): Q1≤{q1:.3f} / Q2-Q3 / Q4>{q3:.3f}")
            pred_qrt, prob_qrt, qrt_metrics = run_classification_split(
                "Quartile-split (3-class)", X_train, y_train_qrt,
                X_eval, y_eval_qrt, n_classes=3, groups_train=groups_train,
                eval_split_name=eval_split)

            # Compact machine-parseable summary line (median-split metrics for Optuna).
            # Only emitted when all metrics are finite; a missing line causes the
            # hparam trial to fail explicitly rather than silently substituting chance level.
            m = med_metrics
            import math as _math
            if any(_math.isnan(v) for v in m.values()):
                print(f"[Probe] WARNING: nan metric for objective={objective} "
                      f"target={tcol} split={eval_split} — summary line suppressed",
                      flush=True)
            else:
                print(
                    f"[Probe] objective={objective} target={tcol} split={eval_split} "
                    f"Balanced Acc.: {m['bacc']:.4f}  "
                    f"MCC: {m['mcc']:.4f}  "
                    f"F1(macro): {m['f1']:.4f}  "
                    f"ROC-AUC: {m['auc']:.4f}",
                    flush=True,
                )

            out = merged_eval[[join_key]].copy()
            out[target_display]    = y_eval_raw
            out["pred_regression"] = y_pred_reg
            out["label_median"]    = y_eval_med
            out["pred_median"]     = pred_med
            out["label_quartile"]  = y_eval_qrt
            out["pred_quartile"]   = pred_qrt
            out["target_col"]      = target_display
            all_outputs.append(out)

        else:
            # ── Original CV-based path (unchanged) ──────────────────────────
            X_raw = merged[feat_cols].values.astype(np.float32)
            y_raw = merged[tcol].values.astype(np.float32)
            if objective == "FAB" and tcol == "portScore":
                y_raw = 16.0 - y_raw
                print(f"  (FAB: portScore inverted → expertiseScore; higher = more expert)")

            scaler = StandardScaler()
            X = scaler.fit_transform(X_raw)

            if "parse_run" in cfg:
                groups = merged["filename"].apply(cfg["parse_run"]).values
                n_groups = len(np.unique(groups))
                print(f"CV strategy      : Leave-One-Group-Out  ({n_groups} run groups)")
            else:
                groups = None
                print(f"CV strategy      : Leave-One-Out")

            y_pred_reg, reg_metrics = run_regression(X, y_raw, groups=groups)
            results[target_display] = {**reg_metrics, 'n_samples': len(merged)}

            median = np.median(y_raw)
            y_med  = (y_raw >= median).astype(int)
            if objective == "FAB" and tcol == "portScore":
                print(f"\nMedian threshold : {median:.3f}  →  novice-like={(y_med==0).sum()}  expert-like={(y_med==1).sum()}")
            else:
                print(f"\nMedian threshold : {median:.3f}  →  low={(y_med==0).sum()}  high={(y_med==1).sum()}")
            pred_med, prob_med = run_classification(
                "Median-split (binary)", X, y_med, n_classes=2, groups=groups)

            q1 = np.percentile(y_raw, 25)
            q3 = np.percentile(y_raw, 75)
            y_qrt = np.where(y_raw <= q1, 0, np.where(y_raw <= q3, 1, 2))
            print(f"\nQuartile thresholds: Q1≤{q1:.3f} / Q2-Q3 / Q4>{q3:.3f}")
            if objective == "FAB" and tcol == "portScore":
                print(f"  Class counts: novice-like={(y_qrt==0).sum()}  intermediate={(y_qrt==1).sum()}  expert-like={(y_qrt==2).sum()}")
            else:
                print(f"  Class counts: low={(y_qrt==0).sum()}  mid={(y_qrt==1).sum()}  high={(y_qrt==2).sum()}")
            pred_qrt, prob_qrt = run_classification(
                "Quartile-split (3-class)", X, y_qrt, n_classes=3, groups=groups)

            out = merged[[join_key]].copy()
            out[target_display]    = y_raw
            out["pred_regression"] = y_pred_reg
            out["label_median"]    = y_med
            out["pred_median"]     = pred_med
            out["label_quartile"]  = y_qrt
            out["pred_quartile"]   = pred_qrt
            out["target_col"]      = target_display
            all_outputs.append(out)

    out_dir = _ROOT / "outputs" / "predictions" / RUN_DIR.name
    out_dir.mkdir(parents=True, exist_ok=True)

    if all_outputs:
        combined = pd.concat(all_outputs, ignore_index=True)
        out_path = out_dir / f"linear_probe_{objective}.csv"
        combined.to_csv(out_path, index=False)
        print(f"\nPredictions saved to: {out_path}")

    return results


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--emb-dir",   type=str, default=None,
                   help="Name or absolute path of embedding run directory "
                        "(default: most recent under outputs/embeddings/)")
    p.add_argument("--objective", type=str, default="FAB",
                   choices=["FAB", "D2", "D3"],
                   help="Which target objective to probe (default: FAB)")
    p.add_argument("--target-col", type=str, default=None,
                   help="Target column for D2/D3 objectives. "
                        "Defaults to first metric column. "
                        "Pass 'all' to iterate over every metric column.")
    p.add_argument("--train_split", type=str, default=None,
                   choices=["train", "train+val"],
                   help="Fit probe on this split subset. Activates split-aware mode.")
    p.add_argument("--eval_split", type=str, default=None,
                   choices=["val", "test"],
                   help="Evaluate probe on this split subset (required with --train_split).")
    args = p.parse_args()
    run_probe(args.emb_dir, args.objective, args.target_col,
              train_split=args.train_split, eval_split=args.eval_split)
