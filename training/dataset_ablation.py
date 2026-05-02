"""
Leave-one-dataset-out ablation for VR motion encoder (MAE).

Runs 27 fixed-hyperparameter experiments (26 dataset exclusions + 1 control)
using the same 5-epoch → linear-probe evaluation pipeline as hparam_search.py.
Fixed hyperparameters come from the best hparam_search trial (t164, agg=0.5945).

Results are appended to outputs/dataset_ablation/results.tsv as each experiment
completes, so the script is safe to interrupt and resume.

Usage
-----
  # Single GPU — all 27 experiments sequentially:
  conda run -n CHOROS python training/dataset_ablation.py --gpu 0

  # Two GPUs in parallel (alternating experiments):
  conda run -n CHOROS python training/dataset_ablation.py --gpu 0 --stride 2 --start_idx 0 &
  conda run -n CHOROS python training/dataset_ablation.py --gpu 1 --stride 2 --start_idx 1

  # Smoke test with a single dataset:
  conda run -n CHOROS python training/dataset_ablation.py --gpu 0 --datasets GaitRecVR
"""

import argparse
import os
import re
import sys
import threading
import time
from pathlib import Path

import subprocess

_CHOROS_ROOT  = Path(__file__).parent.parent
_TRAINING_DIR = Path(__file__).parent
_DATA_ROOT    = Path(os.environ.get('CHOROS_DATA_ROOT', '/srv/CHOROS/data'))
sys.path.insert(0, str(_TRAINING_DIR))
from gpu_profiles import get_gpu_profile, print_gpu_profile, _HIGH_VRAM_GB, _detect_attn_backend
from hparam_search import parse_eval_output, aggregate_metric, _vram_monitor

import torch

# ---------------------------------------------------------------------------
# Dataset list (from docs/dataset_distribution.txt)
# ---------------------------------------------------------------------------

_DATASETS = [
    '3DArmGaze',
    'B100',
    'Circle',
    'GaitRecVR',
    'HMDPoser',
    'IbragimovDining',
    'IbragimovQueue',
    'InHARD-DT',
    'KinSigs',
    'Kreb24',
    'LBS21',
    'LBS22',
    'LBS23',
    'Liikkannen',
    'LocoVR',
    'MMWave',
    'MillerBallThrowing',
    'MoPs',
    'PrivacyGaze',
    'QUESTSET',
    'ShachRackMcMahan25',
    'WESTBROOK',
    'Wu17',
    'Wu25',
    'vr',
    'who',
]

# Sentinel for the control run (no exclusion).
_CONTROL = 'none'

# All experiments: control first, then each dataset exclusion.
_ALL_EXPERIMENTS = [_CONTROL] + _DATASETS

# ---------------------------------------------------------------------------
# Fixed hyperparameters (hparam_search trial 164, aggregate = 0.5945)
# ---------------------------------------------------------------------------

_BEST_PARAMS = {
    'embed_dim':          192,
    'n_heads':            4,
    'n_layers':           7,
    'ffn_dim':            768,
    'mask_type':          'span',
    'mask_ratio':         0.2413603627973692,
    'n_span_blocks':      2,
    'lr':                 0.002330097326156975,
    'sampling_alpha':     0.6227809257847557,
    'dropout':            0.07908855130716654,
    'max_len':            128,
    'feature_group_mask': ['A', 'J'],
}

# ---------------------------------------------------------------------------
# GPU profile
# ---------------------------------------------------------------------------

def _profile_for_gpu(gpu_index: int) -> dict:
    try:
        props   = torch.cuda.get_device_properties(gpu_index)
        vram_gb = props.total_memory / (1024 ** 3)
        high    = vram_gb >= _HIGH_VRAM_GB
        precision = 'bf16' if props.major >= 8 else 'fp16'
        backend   = _detect_attn_backend(props.major)
        return {
            'batch_size':   256 if high else 128,
            'num_workers':  6 if high else 4,
            'compile':      False,
            'precision':    precision,
            'attn_backend': backend,
            'profile_name': 'high-VRAM' if high else 'low-VRAM',
            'gpu_name':     props.name,
            'vram_gb':      vram_gb,
        }
    except Exception:
        return get_gpu_profile()

# ---------------------------------------------------------------------------
# Command builder
# ---------------------------------------------------------------------------

def build_cmd(excluded: str, run_idx: int, gpu: int, profile: dict) -> list[str]:
    """Build the train_vr_encoder.py subprocess command for one ablation run."""
    train_script = str(_CHOROS_ROOT / 'training' / 'train_vr_encoder.py')
    p = _BEST_PARAMS
    cmd = [
        'conda', 'run', '-n', 'CHOROS',
        'python', train_script,
        '--npy_dir',             str(_DATA_ROOT / 'kinematics' / 'VR_npy_PVAJ'),
        '--out_dir',             str(_CHOROS_ROOT / 'outputs' / 'checkpoints'),
        '--epochs',              '5',
        '--embed_eval_interval', '5',
        '--batch_size',          str(profile['batch_size']),
        '--num_workers',         str(profile['num_workers']),
        '--precision',           profile['precision'],
        '--warmup_epochs',       '1',
        '--min_lr',              '1e-6',
        '--kinematics',          'PVAJ',
        '--samples_per_epoch',   '65536',
        '--seed',                str(run_idx),
        '--no_compile',
        '--eval_window_pool',    'mean',
        '--eval_session_pool',   'mean',
        '--mask_ratio',          str(p['mask_ratio']),
        '--mask_type',           p['mask_type'],
        '--n_span_blocks',       str(p['n_span_blocks']),
        '--lr',                  str(p['lr']),
        '--sampling_alpha',      str(p['sampling_alpha']),
        '--dropout',             str(p['dropout']),
        '--embed_dim',           str(p['embed_dim']),
        '--n_heads',             str(p['n_heads']),
        '--n_layers',            str(p['n_layers']),
        '--ffn_dim',             str(p['ffn_dim']),
        '--max_len',             str(p['max_len']),
    ]
    if p['feature_group_mask']:
        cmd += ['--feature_group_mask'] + p['feature_group_mask']
    if excluded != _CONTROL:
        cmd += ['--exclude_datasets', excluded]
    return cmd

# ---------------------------------------------------------------------------
# Results TSV helpers
# ---------------------------------------------------------------------------

_TSV_HEADER = (
    'excluded_dataset\tportScore\tbot_dist_mean_s3\t'
    'firing_accuracy_AOBJ_s3\taggregate\telapsed_s\n'
)

def _load_done(results_path: Path) -> set[str]:
    """Return set of already-completed excluded_dataset values."""
    done: set[str] = set()
    if not results_path.exists():
        return done
    with open(results_path) as f:
        for line in f:
            if line.startswith('excluded_dataset'):
                continue
            parts = line.strip().split('\t')
            if parts:
                done.add(parts[0])
    return done


def _append_row(results_path: Path, excluded: str, metrics: dict[str, float], elapsed: float) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    if not results_path.exists():
        with open(results_path, 'w') as f:
            f.write(_TSV_HEADER)
    agg = aggregate_metric(metrics)
    row = (
        f"{excluded}\t"
        f"{metrics['portScore']:.4f}\t"
        f"{metrics['bot_dist_mean_s3']:.4f}\t"
        f"{metrics['firing_accuracy_AOBJ_s3']:.4f}\t"
        f"{agg:.4f}\t"
        f"{round(elapsed)}\n"
    )
    with open(results_path, 'a') as f:
        f.write(row)

# ---------------------------------------------------------------------------
# Single experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    excluded: str,
    run_idx:  int,
    gpu:      int,
    profile:  dict,
    results_path: Path,
) -> dict[str, float] | None:
    cmd  = build_cmd(excluded, run_idx, gpu, profile)
    env  = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)}
    label = f'excl={excluded}' if excluded != _CONTROL else 'control (no exclusion)'

    peak_mb    = [0]
    stop_event = threading.Event()
    monitor    = threading.Thread(
        target=_vram_monitor, args=(gpu, peak_mb, stop_event), daemon=True
    )
    monitor.start()

    print(f'\n[{run_idx:02d}/{len(_ALL_EXPERIMENTS)-1}] Starting {label}', flush=True)
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=1200,
            env=env,
        )
    except subprocess.TimeoutExpired:
        stop_event.set()
        elapsed = time.time() - t0
        print(f'  TIMEOUT after {elapsed:.0f}s — skipping {excluded}', flush=True)
        return None
    finally:
        stop_event.set()

    elapsed      = time.time() - t0
    peak_vram_gb = peak_mb[0] / 1024.0

    if result.returncode != 0:
        tail = (result.stderr or result.stdout)[-500:]
        print(f'  FAILED (rc={result.returncode}) after {elapsed:.0f}s — skipping {excluded}', flush=True)
        print(f'  {tail}', flush=True)
        return None

    metrics = parse_eval_output(result.stdout)
    missing = {'portScore', 'bot_dist_mean_s3', 'firing_accuracy_AOBJ_s3'} - metrics.keys()
    if missing:
        print(f'  MISSING metrics {missing} — skipping {excluded}', flush=True)
        return None

    agg = aggregate_metric(metrics)
    print(
        f'  Done {elapsed:.0f}s  vram={peak_vram_gb:.1f}GB  agg={agg:.4f}  '
        f'fab={metrics["portScore"]:.4f}  '
        f'bot={metrics["bot_dist_mean_s3"]:.4f}  '
        f'firing={metrics["firing_accuracy_AOBJ_s3"]:.4f}',
        flush=True,
    )

    _append_row(results_path, excluded, metrics, elapsed)
    return metrics

# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_summary(results_path: Path) -> None:
    if not results_path.exists():
        return
    rows = []
    with open(results_path) as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 6:
                rows.append(parts)
    if not rows:
        return

    rows.sort(key=lambda r: float(r[5]), reverse=True)
    print(f'\n{"="*72}')
    print('Dataset Ablation Results (sorted by aggregate score)')
    print(f'{"="*72}')
    print(f'  {"excluded_dataset":24s}  {"agg":6s}  {"fab":6s}  {"bot":6s}  {"firing":6s}  {"flag":6s}')
    print(f'  {"-"*64}')
    for r in rows:
        print(
            f'  {r[0]:24s}  {float(r[5]):.4f}  {float(r[1]):.4f}  '
            f'{float(r[2]):.4f}  {float(r[3]):.4f}  {float(r[4]):.4f}'
        )
    print('='*72)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Leave-one-dataset-out ablation for VR encoder')
    p.add_argument('--gpu',       type=int, default=0,
                   help='CUDA device index (default: 0)')
    p.add_argument('--start_idx', type=int, default=0,
                   help='Index into experiment list to start from (default: 0)')
    p.add_argument('--stride',    type=int, default=1,
                   help='Step size through experiment list, for multi-GPU splitting (default: 1)')
    p.add_argument('--datasets',  nargs='*', metavar='DATASET',
                   help='Run only these dataset exclusions (use "none" for control run). '
                        'Default: all 27 experiments.')
    p.add_argument('--out_dir',   type=str,
                   default=str(_CHOROS_ROOT / 'outputs' / 'dataset_ablation'),
                   help='Directory for results TSV (default: outputs/dataset_ablation)')
    return p.parse_args()


def main():
    args    = parse_args()
    profile = _profile_for_gpu(args.gpu)
    print_gpu_profile(profile)

    results_path = Path(args.out_dir) / 'results.tsv'
    done         = _load_done(results_path)

    # Determine experiment list.
    if args.datasets:
        experiments = [d for d in args.datasets if d in _ALL_EXPERIMENTS]
        invalid = [d for d in args.datasets if d not in _ALL_EXPERIMENTS]
        if invalid:
            print(f'Warning: unknown datasets ignored: {invalid}')
    else:
        experiments = _ALL_EXPERIMENTS[args.start_idx::args.stride]

    print(f'\nDataset ablation: {len(experiments)} experiments to run')
    print(f'  GPU: {args.gpu}   start_idx: {args.start_idx}   stride: {args.stride}')
    print(f'  Results: {results_path}')
    already_done = [e for e in experiments if e in done]
    if already_done:
        print(f'  Skipping {len(already_done)} already-done: {already_done}')
    print()

    t_start = time.time()
    completed = 0
    for run_idx, excluded in enumerate(experiments):
        if excluded in done:
            continue
        result = run_experiment(excluded, run_idx, args.gpu, profile, results_path)
        if result is not None:
            completed += 1

    total_s = time.time() - t_start
    h, m, s = int(total_s // 3600), int((total_s % 3600) // 60), int(total_s % 60)
    print(f'\nFinished {completed} experiments in {h:02d}:{m:02d}:{s:02d}')

    _print_summary(results_path)


if __name__ == '__main__':
    main()
