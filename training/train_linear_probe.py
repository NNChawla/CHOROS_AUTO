"""
Linear probe over CHOROS sequence embeddings -> target label.
Split-path mode: separate train/eval splits; hyperparameters selected on train only.

For full cross-validation (no separate test set), see cv_linear_probe.py.

C selection:
  FAB    – LeaveOneOut on train split
  DEVCOM – LeaveOneGroupOut (groups = run numbers) on train split
"""

import re
import itertools
import warnings
import argparse
import os
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from sklearn.linear_model import Ridge, RidgeCV, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut, cross_val_score, cross_val_predict
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

alphas        = np.logspace(-2, 4, 50)
Cs_grid       = np.logspace(-5, 5, 11)
penalties_grid = ['l1', 'l2']


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


def _make_classifier(C, classifier, penalty='l2'):
    if classifier == "svc":
        return LinearSVC(C=C, penalty=penalty, max_iter=5000, class_weight="balanced",
                         random_state=42, dual=(penalty == 'l2'))
    solver = 'liblinear' if penalty == 'l1' else 'lbfgs'
    return LogisticRegression(C=C, penalty=penalty, max_iter=2000, class_weight="balanced",
                              solver=solver, random_state=42)

def _score_C(C, penalty, X, y_cls, groups, cv, classifier="svc"):
    model = _make_classifier(C, classifier, penalty)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return cross_val_score(model, X, y_cls, cv=cv, groups=groups,
                               scoring="balanced_accuracy").mean()

def _select_C_logo(X, y_cls, groups, classifier="svc"):
    logo = LeaveOneGroupOut()
    param_grid = list(itertools.product(Cs_grid, penalties_grid))
    scores = Parallel(n_jobs=-1)(
        delayed(_score_C)(C, pen, X, y_cls, groups, logo, classifier) for C, pen in param_grid
    )
    return param_grid[int(np.argmax(scores))]  # (C, penalty)

def _select_C_loo(X, y_cls, classifier="svc"):
    loo = LeaveOneOut()
    param_grid = list(itertools.product(Cs_grid, penalties_grid))
    scores = Parallel(n_jobs=-1)(
        delayed(_score_C)(C, pen, X, y_cls, None, loo, classifier) for C, pen in param_grid
    )
    return param_grid[int(np.argmax(scores))]  # (C, penalty)


def run_regression_split(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    groups_train: np.ndarray | None,
    eval_split_name: str,
) -> tuple[np.ndarray, dict]:
    """Fit Ridge on train (alpha via LOGO or 5-fold RidgeCV), evaluate on eval."""
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
    classifier: str = "svc",
    fixed_C: float | None = None,
    fixed_penalty: str | None = None,
) -> tuple[np.ndarray, np.ndarray | None, dict, float | None, str | None]:
    """
    Fit linear classifier on train (C and penalty via LOGO or LOO), evaluate on eval.
    Returns (y_pred, pos_scores_1d, metrics_dict).
    pos_scores_1d is decision_function (SVC) or predict_proba[:,1] (LogReg) for binary,
    or None for the skipped case.
    """
    actual_train_classes = np.unique(y_train)
    actual_eval_classes  = np.unique(y_eval)
    all_classes          = np.union1d(actual_train_classes, actual_eval_classes)
    actual_n             = len(all_classes)

    if len(actual_train_classes) < 2:
        print(f"\n── {label} [{eval_split_name}] SKIPPED (only {len(actual_train_classes)} class in train) ──")
        dummy = np.full(len(y_eval), actual_train_classes[0])
        return dummy, None, {'acc': float('nan'), 'bacc': float('nan'),
                             'mcc': float('nan'), 'f1': float('nan'), 'auc': float('nan')}, None, None

    if fixed_C is not None:
        best_C       = fixed_C
        best_penalty = fixed_penalty if fixed_penalty is not None else 'l2'
        cv_label = "fixed"
    elif groups_train is not None:
        best_C, best_penalty = _select_C_logo(X_train, y_train, groups_train, classifier)
        cv_label = "LOGO"
    else:
        best_C, best_penalty = _select_C_loo(X_train, y_train, classifier)
        cv_label = "LOO"

    clf_label = "LinearSVC" if classifier == "svc" else "LogReg"
    model = _make_classifier(best_C, classifier, best_penalty)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model.fit(X_train, y_train)

    y_pred = model.predict(X_eval)

    # Ranking scores: decision_function for SVC, predict_proba for LogReg
    if classifier == "svc":
        scores = model.decision_function(X_eval)  # 1D binary or 2D multiclass
    else:
        proba        = model.predict_proba(X_eval)
        classes_seen = model.classes_
        if actual_n == 2:
            col_idx = list(classes_seen).index(1) if 1 in classes_seen else 0
            scores  = proba[:, col_idx]
        else:
            scores = proba

    bacc = balanced_accuracy_score(y_eval, y_pred)
    mcc  = matthews_corrcoef(y_eval, y_pred)
    f1   = f1_score(y_eval, y_pred, average='macro', zero_division=0)
    acc  = accuracy_score(y_eval, y_pred)

    if actual_n == 2:
        auc = roc_auc_score(y_eval, scores) if len(np.unique(y_eval)) == 2 else float('nan')
        auc_str = f"  ROC-AUC         : {auc:.4f}"
    else:
        try:
            auc = roc_auc_score(y_eval, scores, multi_class="ovr", average="macro")
        except ValueError:
            auc = float('nan')
        auc_str = f"  ROC-AUC (macro) : {auc:.4f}"

    print(f"\n── {label} [{eval_split_name}] ({clf_label}, C/penalty via {cv_label} on train) ──────────")
    print(f"  Best C          : {best_C:.4f}")
    print(f"  Best penalty    : {best_penalty}")
    print(f"  Accuracy        : {acc:.4f}")
    print(f"  Balanced Acc.   : {bacc:.4f}")
    print(f"  MCC             : {mcc:.4f}")
    print(f"  F1 (macro)      : {f1:.4f}")
    print(auc_str)
    print(f"\n  Classification report (train→{eval_split_name}):")
    print(classification_report(y_eval, y_pred, labels=all_classes, digits=3))

    # Return 1D positive-class score for output CSV
    pos_scores = scores if scores.ndim == 1 else scores[:, 1]
    return y_pred, pos_scores, {'acc': acc, 'bacc': bacc, 'mcc': mcc, 'f1': f1, 'auc': auc}, best_C, best_penalty


def run_probe(
    emb_dir: str | None,
    objective: str = "FAB",
    target_col: str | None = None,
    train_split: str = "train+val",
    eval_split: str = "test",
    train_median: bool = False,
    classifier: str = "svc",
    classification_only: bool = False,
    fixed_C: float | None = None,
    fixed_penalty: str | None = None,
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

        # Re-annotate after merge (index may have changed)
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

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_eval  = scaler.transform(X_eval_raw)

        if "parse_run" in cfg:
            groups_train = merged_train["filename"].apply(cfg["parse_run"]).values
            n_groups = len(np.unique(groups_train))
            print(f"C/alpha selection: LOGO on train  ({n_groups} run groups)")
        else:
            groups_train = None
            print(f"C/alpha selection: LOO on train")

        if classification_only:
            y_pred_reg = np.full(len(y_eval_raw), np.nan, dtype=np.float32)
            results[target_display] = {'n_samples': len(merged_eval)}
            print("\n── Regression skipped (classification-only probe) ─────────────")
        else:
            y_pred_reg, reg_metrics = run_regression_split(
                X_train, y_train_raw, X_eval, y_eval_raw, groups_train, eval_split)
            results[target_display] = {**reg_metrics, 'n_samples': len(merged_eval)}

        if train_median:
            median = np.median(y_train_raw)
            median_label = "train"
        else:
            median = np.median(np.concatenate([y_train_raw, y_eval_raw]))
            median_label = "population"
        y_train_med = (y_train_raw >= median).astype(int)
        y_eval_med  = (y_eval_raw  >= median).astype(int)
        if objective == "FAB" and tcol == "portScore":
            print(f"\nMedian ({median_label}): {median:.3f}  → "
                  f"train novice={(y_train_med==0).sum()} expert={(y_train_med==1).sum()}  "
                  f"eval novice={(y_eval_med==0).sum()} expert={(y_eval_med==1).sum()}")
        else:
            print(f"\nMedian ({median_label}): {median:.3f}  → "
                  f"train low={(y_train_med==0).sum()} high={(y_train_med==1).sum()}  "
                  f"eval low={(y_eval_med==0).sum()} high={(y_eval_med==1).sum()}")

        pred_med, prob_med, med_metrics, med_C, med_penalty = run_classification_split(
            "Median-split (binary)", X_train, y_train_med,
            X_eval, y_eval_med, n_classes=2, groups_train=groups_train,
            eval_split_name=eval_split, classifier=classifier,
            fixed_C=fixed_C, fixed_penalty=fixed_penalty)
        _cv_clf = _make_classifier(med_C, classifier, med_penalty)
        _cv_scheme = LeaveOneGroupOut() if groups_train is not None else LeaveOneOut()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            _y_train_cv = cross_val_predict(
                _cv_clf, X_train, y_train_med,
                cv=_cv_scheme, groups=groups_train)
            try:
                _score_method = 'decision_function' if classifier == 'svc' else 'predict_proba'
                _scores_cv = cross_val_predict(
                    _cv_clf, X_train, y_train_med,
                    cv=_cv_scheme, groups=groups_train, method=_score_method)
                if classifier != 'svc' and _scores_cv.ndim == 2:
                    _scores_cv = _scores_cv[:, 1]
                _auc_train = (roc_auc_score(y_train_med, _scores_cv)
                              if len(np.unique(y_train_med)) == 2 else float('nan'))
            except Exception:
                _auc_train = float('nan')
        _cv_label = "LOGO" if groups_train is not None else "LOO"
        print(f"\n── Median-split (binary) [train] CV ({_cv_label}) ──────────")
        train_med_metrics = {
            'acc':  accuracy_score(y_train_med, _y_train_cv),
            'bacc': balanced_accuracy_score(y_train_med, _y_train_cv),
            'mcc':  matthews_corrcoef(y_train_med, _y_train_cv),
            'f1':   f1_score(y_train_med, _y_train_cv, average='macro', zero_division=0),
            'auc':  _auc_train,
        }
        print(f"  Balanced Acc.   : {train_med_metrics['bacc']:.4f}")
        print(f"  Accuracy        : {train_med_metrics['acc']:.4f}")
        print(f"  MCC             : {train_med_metrics['mcc']:.4f}")
        print(f"  F1 (macro)      : {train_med_metrics['f1']:.4f}")
        print(f"  ROC-AUC         : {train_med_metrics['auc']:.4f}")

        if classification_only:
            y_eval_qrt = np.full(len(y_eval_raw), np.nan, dtype=np.float32)
            pred_qrt = np.full(len(y_eval_raw), np.nan, dtype=np.float32)
            print("\n── Quartile classification skipped (classification-only probe) ─────────────")
        else:
            q1 = np.percentile(y_train_raw, 25)
            q3 = np.percentile(y_train_raw, 75)
            y_train_qrt = np.where(y_train_raw <= q1, 0, np.where(y_train_raw <= q3, 1, 2))
            y_eval_qrt  = np.where(y_eval_raw  <= q1, 0, np.where(y_eval_raw  <= q3, 1, 2))
            print(f"\nQuartile (train): Q1≤{q1:.3f} / Q2-Q3 / Q4>{q3:.3f}")
            pred_qrt, prob_qrt, qrt_metrics, _, _ = run_classification_split(
                "Quartile-split (3-class)", X_train, y_train_qrt,
                X_eval, y_eval_qrt, n_classes=3, groups_train=groups_train,
                eval_split_name=eval_split, classifier=classifier,
                fixed_C=fixed_C, fixed_penalty=fixed_penalty)

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
                f"Acc.: {m['acc']:.4f}  "
                f"MCC: {m['mcc']:.4f}  "
                f"F1(macro): {m['f1']:.4f}  "
                f"ROC-AUC: {m['auc']:.4f}",
                flush=True,
            )
        tm = train_med_metrics
        if not any(_math.isnan(v) for v in tm.values()):
            print(
                f"[Probe] objective={objective} target={tcol} split=train "
                f"Balanced Acc.: {tm['bacc']:.4f}  "
                f"Acc.: {tm['acc']:.4f}  "
                f"MCC: {tm['mcc']:.4f}  "
                f"F1(macro): {tm['f1']:.4f}  "
                f"ROC-AUC: {tm['auc']:.4f}",
                flush=True,
            )

        out = merged_eval[[join_key]].copy()
        out[target_display]    = y_eval_raw
        out["pred_regression"] = y_pred_reg
        out["label_median"]    = y_eval_med
        out["pred_median"]     = pred_med
        out["prob_med_pos"]    = prob_med if prob_med is not None else np.nan
        out["label_quartile"]  = y_eval_qrt
        out["pred_quartile"]   = pred_qrt
        out["target_col"]      = target_display
        all_outputs.append(out)

    out_dir = _ROOT / "outputs" / "predictions" / RUN_DIR.name
    out_dir.mkdir(parents=True, exist_ok=True)

    if all_outputs:
        combined = pd.concat(all_outputs, ignore_index=True)
        tcol_tag = f"_{target_cols_to_run[0]}" if len(target_cols_to_run) == 1 else ""
        out_path = out_dir / f"linear_probe_{objective}{tcol_tag}.csv"
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
    p.add_argument("--train_split", type=str, default="train+val",
                   choices=["train", "train+val"])
    p.add_argument("--eval_split", type=str, default="test",
                   choices=["val", "test"])
    p.add_argument("--no_train_median", action='store_true', default=False,
                   help="Use population median as threshold (default: train-split median).")
    p.add_argument("--classifier", type=str, default="svc",
                   choices=["svc", "logreg"],
                   help="Classifier: svc=LinearSVC (default), logreg=LogisticRegression.")
    p.add_argument("--classification-only", action="store_true",
                   help="Only run the median-split classifier used by summary metrics.")
    p.add_argument("--fixed-C", type=float, default=None,
                   help="Use a fixed classifier C and skip C cross-validation.")
    p.add_argument("--fixed-penalty", type=str, default=None,
                   choices=["l1", "l2"],
                   help="Use a fixed penalty and skip penalty cross-validation "
                        "(only meaningful with --fixed-C; defaults to l2).")
    args = p.parse_args()

    run_probe(args.emb_dir, args.objective, args.target_col,
              train_split=args.train_split, eval_split=args.eval_split,
              train_median=not args.no_train_median,
              classifier=args.classifier,
              classification_only=args.classification_only,
              fixed_C=args.fixed_C,
              fixed_penalty=args.fixed_penalty)
