#!/usr/bin/env python3
"""
Run ablation evaluations and aggregate results.

For each run directory, parquet data for DEVCOM and FAB is loaded once per run,
then reused across all checkpoints (checkpoint_best, checkpoint_latest, and the
best-probe checkpoints).  For each checkpoint:
  - Both datasets are embedded in-process (reusing the loaded model for both).
  - Probes for D3/bot_dist, D3/firing, and FAB are run as subprocesses.

Up to len(GPU_IDS) checkpoints are processed concurrently via a thread pool;
each thread pins its model to a specific CUDA device.

Saves results incrementally to ablation_eval_results.json so runs can be resumed.
Writes a human-readable summary to ablation_eval_summary.txt when done.
"""

import itertools
import json
import os
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

# Project imports — add paths before importing
REPO = Path("/srv/CHOROS_AUTO")
sys.path.insert(0, str(REPO / 'pipeline'))
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'training'))

from embed_target_data import load_raw_sequences, embed_all_preloaded, load_checkpoint
from features import build_feature_cols
from splits import filter_devcom_files, filter_fab_files

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

BASE      = Path("/srv/CHOROS_AUTO/outputs/checkpoints/5_6_26_ablations")
EMBED_OUT = REPO / 'outputs' / 'embeddings'
JSON_OUT  = BASE / "ablation_eval_results.json"
TXT_OUT   = BASE / "ablation_eval_summary.txt"

DEVCOM_DIR  = Path("/srv/CHOROS/data/aligned/target_DEVCOM_s2")
FAB_DIR     = Path("/srv/CHOROS/data/aligned/target_FAB")
SPLIT_KEYS  = ['train', 'val', 'test']

# GPU slots: repeat an index for multiple concurrent jobs on that GPU.
# dim=256 / 6-layer model uses ~500 MB VRAM per job; 3090 (24 GB) fits 4+.
GPU_IDS = [0, 0, 0, 0, 1, 1]

_N_CPU        = os.cpu_count() or 4
_PROBE_N_JOBS = max(1, _N_CPU // len(GPU_IDS))

# (friendly_name, directory_basename)
RUNS = [
    ("spe256k_dp0.05_sf2",        "30601_20260507_002403_VR_npy_PVAJ_posejepa_e150_bs512_lr0.0004_dim256_l6_ml128_ps8_tr0.5_tmmasked_span_llsmooth_l1_poolmean_wu12_minlr1e-06_kinPVAJ_sa0.5"),
    ("spe256k_tr0.25_sf2",        "30602_20260507_035529_VR_npy_PVAJ_posejepa_e150_bs512_lr0.0004_dim256_l6_ml128_ps8_tr0.25_tmmasked_span_llsmooth_l1_poolmean_wu12_minlr1e-06_kinPVAJ_sa0.5"),
    ("spe256k_dp0.05_sf4",        "30901_20260507_002414_VR_npy_PVAJ_posejepa_e150_bs512_lr0.0004_dim256_l6_ml128_ps8_tr0.5_tmmasked_span_llsmooth_l1_poolmean_wu12_minlr1e-06_kinPVAJ_sa0.5"),
    ("spe256k_dp0.05_tr0.25_sf4", "30903_20260507_034326_VR_npy_PVAJ_posejepa_e150_bs512_lr0.0004_dim256_l6_ml128_ps8_tr0.25_tmmasked_span_llsmooth_l1_poolmean_wu12_minlr1e-06_kinPVAJ_sa0.5"),
    ("spe512k_dp0.05_sf4",        "40901_20260506_203027_VR_npy_PVAJ_posejepa_e150_bs512_lr0.0004_dim256_l6_ml128_ps8_tr0.5_tmmasked_span_llsmooth_l1_poolmean_wu12_minlr1e-06_kinPVAJ_sa0.5"),
    ("spe512k_tr0.25_sf4",        "40902_20260506_215133_VR_npy_PVAJ_posejepa_e150_bs512_lr0.0004_dim256_l6_ml128_ps8_tr0.25_tmmasked_span_llsmooth_l1_poolmean_wu12_minlr1e-06_kinPVAJ_sa0.5"),
    ("spe512k_dp0.05_tr0.25_sf4", "40903_20260506_233147_VR_npy_PVAJ_posejepa_e150_bs512_lr0.0004_dim256_l6_ml128_ps8_tr0.25_tmmasked_span_llsmooth_l1_poolmean_wu12_minlr1e-06_kinPVAJ_sa0.5"),
    ("spe512k_dp0.05_sf2",        "50901_20260507_020828_VR_npy_PVAJ_posejepa_e150_bs512_lr0.0004_dim256_l6_ml128_ps8_tr0.5_tmmasked_span_llsmooth_l1_poolmean_wu12_minlr1e-06_kinPVAJ_sa0.5"),
    ("spe512k_tr0.25_sf2",        "50902_20260507_030820_VR_npy_PVAJ_posejepa_e150_bs512_lr0.0004_dim256_l6_ml128_ps8_tr0.25_tmmasked_span_llsmooth_l1_poolmean_wu12_minlr1e-06_kinPVAJ_sa0.5"),
    ("spe512k_dp0.05_tr0.25_sf2", "50903_20260507_040849_VR_npy_PVAJ_posejepa_e150_bs512_lr0.0004_dim256_l6_ml128_ps8_tr0.25_tmmasked_span_llsmooth_l1_poolmean_wu12_minlr1e-06_kinPVAJ_sa0.5"),
]

def get_checkpoints(dir_path: Path) -> list[str]:
    return sorted(p.stem for p in dir_path.glob("checkpoint_*.pt"))

# Result cmd_ids in display order
CMD_ORDER = [
    ("D3_bot_dist", "D3/bot_dist"),
    ("D3_firing",   "D3/firing"),
    ("FAB",         "FAB"),
]

_TARGET_TO_CMD_ID = {
    'bot_dist_mean_s3':        'D3_bot_dist',
    'firing_accuracy_AOBJ_s3': 'D3_firing',
    'portScore':               'FAB',
}

# ──────────────────────────────────────────────────────────────────────────────
# Train-log parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_train_log(log_path: Path):
    """Return (batch_size, max_len, stride_factor, window_pool, session_pool, kinematics)."""
    batch_size = max_len = stride_factor = stride = window_pool = session_pool = kinematics = None
    with open(log_path) as f:
        for line in f:
            if batch_size is None:
                m = re.search(r'^\s*batch_size\s*:\s*(\d+)', line)
                if m:
                    batch_size = int(m.group(1))
            if max_len is None:
                m = re.search(r'^\s*max_len\s*:\s*(\d+)', line)
                if m:
                    max_len = int(m.group(1))
            if stride_factor is None:
                m = re.search(r'^\s*stride_factor\s*:\s*(\d+)', line)
                if m:
                    stride_factor = int(m.group(1))
            if stride is None:
                m = re.search(r'_stride(\d+)_', line)
                if m:
                    stride = int(m.group(1))
            if window_pool is None:
                m = re.search(r'_wp_(.+)_sp_([^.\s]+)\.pt', line)
                if m:
                    window_pool = m.group(1)
                    session_pool = m.group(2)
            if kinematics is None:
                m = re.search(r'^\s*kinematics\s*:\s*(\S+)', line)
                if m:
                    kinematics = m.group(1)
    if stride_factor is None and max_len and stride:
        stride_factor = max_len // stride
    return batch_size, max_len, stride_factor, window_pool, session_pool, kinematics


def _kinematics_from_checkpoint(ckpt_file: Path) -> str:
    """Read kinematics from checkpoint args without moving tensors to GPU."""
    ckpt = torch.load(ckpt_file, map_location='cpu', weights_only=False)
    return ckpt['args'].get('kinematics', 'P')

# ──────────────────────────────────────────────────────────────────────────────
# Probe subprocess
# ──────────────────────────────────────────────────────────────────────────────

def _probe_subprocess(emb_dir: Path, objective: str, target_col: str | None) -> str:
    """Run train_linear_probe.py for one emb_dir/objective, return stdout+stderr."""
    probe_script = REPO / 'training' / 'train_linear_probe.py'
    env = os.environ.copy()
    pj  = str(_PROBE_N_JOBS)
    env.update({
        'LOKY_MAX_CPU_COUNT':  pj,
        'OMP_NUM_THREADS':     pj,
        'MKL_NUM_THREADS':     pj,
        'OPENBLAS_NUM_THREADS': pj,
    })
    cmd = [sys.executable, str(probe_script),
           '--emb-dir',     str(emb_dir),
           '--objective',   objective,
           '--train_split', 'train+val',
           '--eval_split',  'test']
    if target_col:
        cmd += ['--target-col', target_col]
    r = subprocess.run(cmd, capture_output=True, text=True,
                       cwd=str(REPO), timeout=3600, env=env)
    return r.stdout + r.stderr

# ──────────────────────────────────────────────────────────────────────────────
# Probe output parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_all_probe_lines(output: str) -> dict[str, dict]:
    """Return {cmd_id: metrics_dict} for every [Probe] test line in output."""
    pat = (
        r'\[Probe\] objective=(\S+) target=(\S+) split=test '
        r'Balanced Acc\.: ([\d.]+)\s+MCC: ([-\d.]+)\s+'
        r'F1\(macro\): ([\d.]+)\s+ROC-AUC: ([\d.]+)'
    )
    results = {}
    for line in output.splitlines():
        m = re.search(pat, line)
        if m:
            target = m.group(2)
            cmd_id = _TARGET_TO_CMD_ID.get(target)
            if cmd_id:
                results[cmd_id] = {
                    'objective':    m.group(1),
                    'target':       target,
                    'balanced_acc': float(m.group(3)),
                    'mcc':          float(m.group(4)),
                    'f1_macro':     float(m.group(5)),
                    'roc_auc':      float(m.group(6)),
                }
    return results

# ──────────────────────────────────────────────────────────────────────────────
# Thread-safe logging
# ──────────────────────────────────────────────────────────────────────────────

_print_lock = threading.Lock()

def _log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)

# ──────────────────────────────────────────────────────────────────────────────
# Per-checkpoint worker (called from thread pool)
# ──────────────────────────────────────────────────────────────────────────────

def _eval_checkpoint(
    run_name:    str,
    ckpt_stem:   str,
    ckpt_file:   Path,
    gpu_id:      int,
    devcom_raw:  list[tuple[str, np.ndarray]],
    fab_raw:     list[tuple[str, np.ndarray]],
    bs:          int,
    wp:          str,
    sp:          str,
    stride:      int,
    tag:         str,
) -> dict[str, dict | None]:
    """Embed DEVCOM+FAB with one checkpoint (shared model load) then run probes.

    Returns {cmd_id: metrics_or_None} for D3_bot_dist, D3_firing, and FAB.
    """
    device = torch.device(f'cuda:{gpu_id}')
    _log(f"  START  {tag}  GPU={gpu_id}")

    model, ckpt_data = load_checkpoint(ckpt_file, device)

    devcom_emb = embed_all_preloaded(
        devcom_raw, DEVCOM_DIR.name, ckpt_file, EMBED_OUT,
        stride, bs, wp, sp, model=model, ckpt=ckpt_data,
    )
    fab_emb = embed_all_preloaded(
        fab_raw, FAB_DIR.name, ckpt_file, EMBED_OUT,
        stride, bs, wp, sp, model=model, ckpt=ckpt_data,
    )

    # Free GPU before CPU-bound probe
    del model
    torch.cuda.empty_cache()

    results: dict[str, dict | None] = {}

    for target_col, cmd_id in [('bot_dist_mean_s3',        'D3_bot_dist'),
                                ('firing_accuracy_AOBJ_s3', 'D3_firing')]:
        out     = _probe_subprocess(devcom_emb, 'D3', target_col)
        metrics = parse_all_probe_lines(out).get(cmd_id)
        results[cmd_id] = metrics
        if metrics:
            _log(f"  DONE   {tag}/{cmd_id:30s}  MCC={metrics['mcc']:.4f}  BAcc={metrics['balanced_acc']:.4f}")
        else:
            _log(f"  WARN   {tag}/{cmd_id:30s}  no [Probe] test line")
            for ln in [ln for ln in out.splitlines() if ln.strip()][-4:]:
                _log(f"    > {ln}")

    out     = _probe_subprocess(fab_emb, 'FAB', None)
    metrics = parse_all_probe_lines(out).get('FAB')
    results['FAB'] = metrics
    if metrics:
        _log(f"  DONE   {tag}/{'FAB':30s}  MCC={metrics['mcc']:.4f}  BAcc={metrics['balanced_acc']:.4f}")
    else:
        _log(f"  WARN   {tag}/{'FAB':30s}  no [Probe] test line")
        for ln in [ln for ln in out.splitlines() if ln.strip()][-4:]:
            _log(f"    > {ln}")

    return results

# ──────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ──────────────────────────────────────────────────────────────────────────────

def fmt_val(v, width=7):
    return f"{v:>{width}.4f}" if isinstance(v, float) else f"{'N/A':>{width}}"

def shorten_ckpt(name: str) -> str:
    return re.sub(r'_wp_.+', '', name)

CKPT_SHORT_ORDER = [
    "checkpoint_best",
    "checkpoint_latest",
    "checkpoint_best_probe_bot_dist",
    "checkpoint_best_probe_firing",
    "checkpoint_best_probe_port_score",
    "checkpoint_best_probe_val",
]

def format_per_run(results: dict) -> str:
    lines = []
    for run_name, _ in RUNS:
        run_data = results.get(run_name, {})
        ckpts = sorted(run_data.keys(),
                       key=lambda c: CKPT_SHORT_ORDER.index(shorten_ckpt(c))
                       if shorten_ckpt(c) in CKPT_SHORT_ORDER else 99)
        lines.append(f"\n{'='*90}")
        lines.append(f"RUN: {run_name}")
        lines.append('='*90)
        for ckpt in ckpts:
            cmd_data = run_data[ckpt]
            lines.append(f"\n  [{shorten_ckpt(ckpt)}]")
            lines.append(f"  {'Command':<22}  {'Bal.Acc':>8} {'MCC':>8} {'F1(mac)':>8} {'ROC-AUC':>8}")
            lines.append("  " + "-" * 58)
            for cmd_id, label in CMD_ORDER:
                m = cmd_data.get(cmd_id)
                if m:
                    lines.append(
                        f"  {label:<22}  "
                        f"{fmt_val(m['balanced_acc'], 8)} {fmt_val(m['mcc'], 8)} "
                        f"{fmt_val(m['f1_macro'], 8)} {fmt_val(m['roc_auc'], 8)}"
                    )
                else:
                    lines.append(f"  {label:<22}  {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8}")
    return "\n".join(lines)

def format_cross_run(results: dict, metric: str = 'mcc') -> str:
    col_keys = [(s, cmd_id) for s in CKPT_SHORT_ORDER for cmd_id, _ in CMD_ORDER]
    def abbrev(s):
        return s.replace("checkpoint_best_probe_", "probe_").replace("checkpoint_", "")
    col_labels = [f"{abbrev(s)}/{cmd_id}" for s, cmd_id in col_keys]
    col_w = max(len(l) for l in col_labels) + 1
    run_w  = max(len(n) for n, _ in RUNS) + 1
    lines = [f"\nCross-run comparison  metric={metric.upper()}"]
    lines.append(f"  {'Run':<{run_w}}  " + "  ".join(f"{l:>{col_w}}" for l in col_labels))
    lines.append("  " + "-" * (run_w + 2 + len(col_labels) * (col_w + 2)))
    for run_name, _ in RUNS:
        run_data = results.get(run_name, {})
        short_to_full = {shorten_ckpt(k): k for k in run_data}
        row = f"  {run_name:<{run_w}}  "
        vals = []
        for short, cmd_id in col_keys:
            full = short_to_full.get(short)
            m = run_data.get(full, {}).get(cmd_id) if full else None
            vals.append(f"{m[metric]:>{col_w}.4f}" if m else f"{'N/A':>{col_w}}")
        row += "  ".join(vals)
        lines.append(row)
    return "\n".join(lines)

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    if JSON_OUT.exists():
        with open(JSON_OUT) as f:
            all_results = json.load(f)
        n_cached = sum(
            1 for rd in all_results.values()
            for cd in rd.values()
            for v in cd.values()
            if v is not None
        )
        print(f"Resuming: {n_cached} cached results loaded from {JSON_OUT}")
    else:
        all_results = {}

    json_lock    = threading.Lock()
    job_counter  = itertools.count(1)
    gpu_cycle    = itertools.cycle(GPU_IDS)

    print(f"\n{len(GPU_IDS)} workers  GPU slots={GPU_IDS}  probe_n_jobs={_PROBE_N_JOBS}")

    with ThreadPoolExecutor(max_workers=len(GPU_IDS)) as executor:
        for run_name, dir_name in RUNS:
            dir_path = BASE / dir_name
            log_path = dir_path / "train.log"

            print(f"\n{'='*60}")
            print(f"RUN: {run_name}")

            if not log_path.exists():
                print(f"  SKIP — train.log missing")
                continue

            bs, ml, sf, wp, sp, kinematics = parse_train_log(log_path)
            if not all([bs, ml, sf, wp, sp]):
                print(f"  SKIP — could not parse params (bs={bs} ml={ml} sf={sf} wp={wp} sp={sp})")
                continue

            stride      = ml // sf
            checkpoints = get_checkpoints(dir_path)
            print(f"  bs={bs}  max_len={ml}  stride_factor={sf}  stride={stride}  wp={wp}  sp={sp}")
            print(f"  checkpoints ({len(checkpoints)}): {', '.join(checkpoints)}")

            # Resolve kinematics → feature_cols (try log first, fall back to checkpoint)
            if not kinematics:
                first_ckpt = next(
                    (dir_path / f"{c}.pt" for c in checkpoints if (dir_path / f"{c}.pt").exists()),
                    None,
                )
                kinematics = _kinematics_from_checkpoint(first_ckpt) if first_ckpt else 'P'
            feature_cols = build_feature_cols(kinematics)
            print(f"  kinematics={kinematics}  feature_cols={len(feature_cols)}")

            # Determine which files to embed (same split filter used by embed_and_probe)
            devcom_files = set(filter_devcom_files(sorted(DEVCOM_DIR.glob('*.parquet')), SPLIT_KEYS))
            fab_files    = set(filter_fab_files(sorted(FAB_DIR.glob('*.parquet')), SPLIT_KEYS))

            # Pre-load parquet data once for this entire run
            print(f"  Loading DEVCOM ({len(devcom_files)} files)...")
            devcom_raw = load_raw_sequences(DEVCOM_DIR, feature_cols, devcom_files)
            print(f"  Loading FAB    ({len(fab_files)} files)...")
            fab_raw    = load_raw_sequences(FAB_DIR, feature_cols, fab_files)

            all_results.setdefault(run_name, {})

            # Submit one future per checkpoint
            run_futures: dict = {}
            for ckpt_stem in checkpoints:
                ckpt_file = dir_path / f"{ckpt_stem}.pt"
                if not ckpt_file.exists():
                    print(f"  SKIP {ckpt_stem}.pt — file missing")
                    continue

                all_results[run_name].setdefault(ckpt_stem, {})

                all_cached = all(
                    all_results[run_name][ckpt_stem].get(cid) is not None
                    for cid in ['D3_bot_dist', 'D3_firing', 'FAB']
                )
                if all_cached:
                    for cid in ['D3_bot_dist', 'D3_firing', 'FAB']:
                        m = all_results[run_name][ckpt_stem][cid]
                        print(f"  CACHED  {ckpt_stem}/{cid:30s}  MCC={m['mcc']:.4f}  BAcc={m['balanced_acc']:.4f}")
                    continue

                jn  = next(job_counter)
                tag = f"[job {jn:3d}] {run_name}/{ckpt_stem}"
                f   = executor.submit(
                    _eval_checkpoint,
                    run_name, ckpt_stem, ckpt_file, next(gpu_cycle),
                    devcom_raw, fab_raw, bs, wp, sp, stride, tag,
                )
                run_futures[f] = ckpt_stem

            # Wait for this run's checkpoints before pre-loading the next run
            for future in as_completed(run_futures):
                ckpt_stem = run_futures[future]
                try:
                    results = future.result()
                    with json_lock:
                        all_results[run_name][ckpt_stem].update(results)
                        with open(JSON_OUT, 'w') as jf:
                            json.dump(all_results, jf, indent=2)
                except Exception as e:
                    _log(f"  ERROR  {run_name}/{ckpt_stem}: {e}")

    # ── Final summary ──────────────────────────────────────────────────────────
    summary = ["ABLATION EVALUATION RESULTS", "=" * 90,
               format_per_run(all_results), "\n\n"]
    for metric in ("mcc", "balanced_acc", "f1_macro", "roc_auc"):
        summary.append(format_cross_run(all_results, metric))
        summary.append("")

    summary_txt = "\n".join(summary)
    print("\n\n" + summary_txt)

    with open(TXT_OUT, "w") as f:
        f.write(summary_txt)

    print(f"\nJSON results → {JSON_OUT}")
    print(f"Text summary → {TXT_OUT}")


if __name__ == "__main__":
    main()
