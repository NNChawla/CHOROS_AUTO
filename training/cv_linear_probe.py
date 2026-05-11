"""
Cross-validation linear probe over CHOROS sequence embeddings -> target label.
Full CV mode: no separate test set; predictions produced by proper nested CV.

For split-path mode (separate train/eval splits), see train_linear_probe.py.

Classification CV:
  FAB    – nested LOO/SKF: outer LOO for predictions, inner StratifiedKFold for C
  DEVCOM – nested LOGO/SGKF: outer LOGO for predictions, inner StratifiedGroupKFold for C

Regression CV:
  FAB    – 5-fold RidgeCV for alpha selection, then LOO cross_val_predict
  DEVCOM – LOGO for alpha selection, then LOGO cross_val_predict
"""

import re
import warnings
import argparse
import os
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from sklearn.linear_model import Ridge, RidgeCV, LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (LeaveOneOut, LeaveOneGroupOut,
                                     StratifiedKFold, StratifiedGroupKFold,
                                     cross_val_predict, cross_val_score)
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


def _score_alpha(alpha, X, y_raw, groups, cv):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cross_val_score(Ridge(alpha=alpha), X, y_raw,
                               cv=cv, groups=groups, scoring="r2").mean()

def _select_alpha_logo(X, y_raw, groups):
    logo = LeaveOneGroupOut()
    scores = Parallel(n_jobs=-1)(
        delayed(_score_alpha)(a, X, y_raw, groups, logo) for a in alphas
    )
    return alphas[int(np.argmax(scores))]


def _score_C(C, X, y_cls, groups, cv):
    model = LogisticRegression(C=C, max_iter=2000, class_weight="balanced",
                               solver="lbfgs", random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return cross_val_score(model, X, y_cls, cv=cv, groups=groups,
                               scoring="balanced_accuracy").mean()

def _select_C_skf(X, y_cls, n_splits=5):
    min_class_count = min((y_cls == c).sum() for c in np.unique(y_cls))
    actual_splits = max(2, min(n_splits, int(min_class_count)))
    skf = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=42)
    scores = Parallel(n_jobs=-1)(
        delayed(_score_C)(C, X, y_cls, None, skf) for C in Cs_grid
    )
    return Cs_grid[int(np.argmax(scores))]

def _select_C_sgkf(X, y_cls, groups, n_splits=5):
    n_groups = len(np.unique(groups))
    actual_splits = max(2, min(n_splits, n_groups))
    sgkf = StratifiedGroupKFold(n_splits=actual_splits)
    scores = Parallel(n_jobs=-1)(
        delayed(_score_C)(C, X, y_cls, groups, sgkf) for C in Cs_grid
    )
    return Cs_grid[int(np.argmax(scores))]


def _nested_loo_predict_proba(X, y_cls):
    """Outer LOO, inner StratifiedKFold C selection. No data leakage."""
    loo_outer = LeaveOneOut()
    classes = np.unique(y_cls)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_prob = np.zeros((len(y_cls), len(classes)))

    for train_idx, test_idx in loo_outer.split(X, y_cls):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr = y_cls[train_idx]
        best_C = _select_C_skf(X_tr, y_tr)
        model = LogisticRegression(C=best_C, max_iter=2000, class_weight="balanced",
                                   solver="lbfgs", random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            model.fit(X_tr, y_tr)
        prob = model.predict_proba(X_te)
        for j, c in enumerate(model.classes_):
            y_prob[test_idx[0], class_to_idx[c]] = prob[0, j]

    return y_prob


def _nested_logo_predict_proba(X, y_cls, groups):
    """Outer LOGO, inner StratifiedGroupKFold C selection. No data leakage."""
    logo_outer = LeaveOneGroupOut()
    classes = np.unique(y_cls)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_prob = np.zeros((len(y_cls), len(classes)))

    for train_idx, test_idx in logo_outer.split(X, y_cls, groups):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr = y_cls[train_idx]
        groups_tr = groups[train_idx]
        best_C = _select_C_sgkf(X_tr, y_tr, groups_tr)
        model = LogisticRegression(C=best_C, max_iter=2000, class_weight="balanced",
                                   solver="lbfgs", random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            model.fit(X_tr, y_tr)
        prob = model.predict_proba(X_te)
        for j, c in enumerate(model.classes_):
            for k, te_idx in enumerate(test_idx):
                y_prob[te_idx, class_to_idx[c]] = prob[k, j]

    return y_prob


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
        y_prob = _nested_logo_predict_proba(X, y_cls, groups)
        cv_label = "nested-LOGO/SGKF"
    else:
        y_prob = _nested_loo_predict_proba(X, y_cls)
        cv_label = "nested-LOO/SKF"

    y_pred = actual_classes[y_prob.argmax(axis=1)]

    acc  = accuracy_score(y_cls, y_pred)
    bacc = balanced_accuracy_score(y_cls, y_pred)
    if actual_n == 2:
        auc = roc_auc_score(y_cls, y_prob[:, 1])
        auc_str = f"  ROC-AUC         : {auc:.4f}"
    else:
        auc = roc_auc_score(y_cls, y_prob, multi_class="ovr", average="macro")
        auc_str = f"  ROC-AUC (macro) : {auc:.4f}"

    print(f"\n── {label} {cv_label} Results ──────────────────────────────────────────────────")
    print(f"  Accuracy        : {acc:.4f}")
    print(f"  Balanced Acc.   : {bacc:.4f}")
    print(auc_str)
    print(f"\n  Classification report ({cv_label}):")
    print(classification_report(y_cls, y_pred, labels=actual_classes, digits=3))

    return y_pred, y_prob


def run_identity_probe(emb_dirs_labels: list[tuple[str, str]]) -> dict:
    """Diagnostic: predict which dataset each session came from using stratified k-fold."""
    frames = []
    for emb_dir, label in emb_dirs_labels:
        csv_path = Path(emb_dir) / 'sequence_embeddings.csv'
        df = pd.read_csv(csv_path)
        df['_dataset'] = label
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    feat_cols = [c for c in combined.columns if c.startswith('e') and c[1:].isdigit()]

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(combined['_dataset'].values)
    n_classes = len(le.classes_)

    print(f"\n{'='*72}")
    print(f"Dataset Identity Probe  ({n_classes} classes, {len(combined)} samples)")
    print(f"{'='*72}")
    for i, cls in enumerate(le.classes_):
        print(f"  Class {i} '{cls}': n={int(np.sum(y == i))}")

    X = StandardScaler().fit_transform(combined[feat_cols].values.astype(np.float32))

    n_splits = max(2, min(5, int(min(np.bincount(y)))))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        clf = LogisticRegressionCV(
            Cs=20, cv=skf, max_iter=2000,
            scoring="balanced_accuracy",
            class_weight="balanced",
            solver="lbfgs",
            n_jobs=-1,
            random_state=42,
        )
        clf.fit(X, y)
        best_C = clf.C_[0]

        skf_pred = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        model = LogisticRegression(C=best_C, max_iter=2000, class_weight="balanced",
                                   solver="lbfgs", random_state=42)
        y_prob = cross_val_predict(model, X, y, cv=skf_pred,
                                   method="predict_proba", n_jobs=-1)

    y_pred = y_prob.argmax(axis=1)

    bacc = balanced_accuracy_score(y, y_pred)
    mcc  = matthews_corrcoef(y, y_pred)
    f1   = f1_score(y, y_pred, average='macro', zero_division=0)
    try:
        auc = (roc_auc_score(y, y_prob, multi_class="ovr", average="macro")
               if n_classes > 2 else roc_auc_score(y, y_prob[:, 1]))
    except ValueError:
        auc = float('nan')

    auc_str = f"  ROC-AUC {'(macro) ' if n_classes > 2 else ''}       : {auc:.4f}"
    print(f"\n── Dataset Identity ({n_splits}-fold CV) ────────────────────────────────────────")
    print(f"  Best C          : {best_C:.4f}")
    print(f"  Balanced Acc.   : {bacc:.4f}")
    print(f"  MCC             : {mcc:.4f}")
    print(f"  F1 (macro)      : {f1:.4f}")
    print(auc_str)
    print(f"\n  Classification report ({n_splits}-fold CV):")
    print(classification_report(y, y_pred, target_names=le.classes_, digits=3))

    return {'bacc': bacc, 'mcc': mcc, 'f1': f1, 'auc': auc}


def run_probe(
    emb_dir: str | None,
    objective: str = "FAB",
    target_col: str | None = None,
) -> dict:
    emb_path = Path(emb_dir) if emb_dir else None
    if emb_path is None or not emb_path.is_absolute():
        RUN_DIR = EMB_DIR / emb_dir if emb_dir else sorted(EMB_DIR.iterdir())[-1]
    else:
        RUN_DIR = emb_path
    EMB_CSV = RUN_DIR / "sequence_embeddings.csv"

    cfg = dict(OBJECTIVE_CFG[objective])

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
            print(f"CV strategy      : Nested LOGO/SGKF  ({n_groups} run groups)")
        else:
            groups = None
            print(f"CV strategy      : Nested LOO/SKF")

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
        out["prob_med_pos"]    = (prob_med[:, 1]
                                  if prob_med is not None and prob_med.shape[1] >= 2
                                  else np.nan)
        out["label_quartile"]  = y_qrt
        out["pred_quartile"]   = pred_qrt
        out["target_col"]      = target_display
        all_outputs.append(out)

    out_dir = _ROOT / "outputs" / "predictions" / RUN_DIR.name
    out_dir.mkdir(parents=True, exist_ok=True)

    if all_outputs:
        combined = pd.concat(all_outputs, ignore_index=True)
        tcol_tag = f"_{target_cols_to_run[0]}" if len(target_cols_to_run) == 1 else ""
        out_path = out_dir / f"cv_probe_{objective}{tcol_tag}.csv"
        combined.to_csv(out_path, index=False)
        print(f"\nPredictions saved to: {out_path}")

    return results


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--emb-dir",   type=str, default=None,
                   help="Name or absolute path of embedding run directory "
                        "(default: most recent under outputs/embeddings/)")
    p.add_argument("--objective", type=str, default="FAB",
                   choices=["FAB", "D2", "D3"])
    p.add_argument("--target-col", type=str, default=None,
                   help="Target column for D2/D3. Defaults to first metric column. "
                        "Pass 'all' to iterate every column.")
    p.add_argument("--identity-dirs", nargs='+', default=None,
                   help="Run dataset identity probe. Provide 2+ 'path:Label' arguments, "
                        "e.g. /path/to/emb1:FAB /path/to/emb2:DEVCOM.")
    args = p.parse_args()

    if args.identity_dirs:
        pairs = []
        for item in args.identity_dirs:
            path, sep, label = item.partition(':')
            pairs.append((path.strip(), label.strip() if sep else Path(path.strip()).name))
        if len(pairs) < 2:
            p.error("--identity-dirs requires at least 2 entries")
        run_identity_probe(pairs)
    else:
        run_probe(args.emb_dir, args.objective, args.target_col)
