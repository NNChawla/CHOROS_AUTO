"""
Bayesian hyperparameter search for VR motion encoder (MAE, TS-JEPA, or Pose-JEPA).

Uses Optuna with TPE sampler.  Each trial runs the relevant training script as
an isolated subprocess, periodically embeds target datasets, then parses probe
output to extract downstream probe metrics across 3 downstream objectives.

Aggregate metric: harmonic mean of per-target val probe scores across
  FAB portScore / DEVCOM bot_dist_mean_s3 / firing_accuracy_AOBJ_s3
  (flag_incidents_s3 excluded: severe val-set class imbalance, 13/4 split)

By default, each per-target probe score is 0.5 * AUROC + 0.5 * rescaled MCC,
where rescaled MCC = (MCC + 1) / 2.
The cross-target aggregate blends harmonic mean with the weakest target:
0.75 * harmonic_mean(target_scores) + 0.25 * min(target_scores).

Trial selection defaults to a stability-aware score over recent probe evals:
mean(last K aggregate probe scores) - volatility_penalty * std(last K)
+ trend_weight * (final - first).  Use --score_mode last to reproduce the
old final-checkpoint-only trajectory handling.


Usage
-----
  # Single GPU (local SQLite):
  conda run -n CHOROS python training/hparam_search.py --gpu 0 --n_trials 40

  # TS-JEPA objective:
  conda run -n CHOROS python training/hparam_search.py --gpu 0 --n_trials 40 --objective tsjepa

  # Pose-JEPA objective:
  conda run -n CHOROS python training/hparam_search.py --gpu 0 --n_trials 40 --objective posejepa

  # Parallel workers on both GPUs (share one SQLite study):
  conda run -n CHOROS python training/hparam_search.py --gpu 0 --n_trials 20 &
  conda run -n CHOROS python training/hparam_search.py --gpu 1 --n_trials 20

  # Distributed across machines — coordinator (spaceboi, runs Phase 2+3):
  conda run -n CHOROS python training/hparam_search.py --gpu 0 --n_trials 40 \\
      --storage "postgresql://optuna:choroshps@app.arcadea.us/optuna_choros" \\
      --worker_offset 0

  # Remote worker (add --skip_final, unique --worker_offset per machine):
  conda run -n CHOROS python training/hparam_search.py --gpu 0 --n_trials 40 \\
      --storage "postgresql://optuna:choroshps@app.arcadea.us/optuna_choros" \\
      --worker_offset 1 --skip_final

  # Slower GPU (e.g. RTX 4070) — increase trial timeout so 100-epoch trials finish:
  conda run -n CHOROS python training/hparam_search.py --gpu 0 --n_trials 40 \\
      --storage "postgresql://optuna:choroshps@app.arcadea.us/optuna_choros" \\
      --worker_offset 4070 --skip_final --objective posejepa --trial_timeout 7200
"""

import argparse
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from statistics import harmonic_mean

import optuna
import subprocess
import numpy as np

optuna.logging.set_verbosity(optuna.logging.WARNING)

_CHOROS_ROOT  = Path(__file__).parent.parent
_TRAINING_DIR = Path(__file__).parent
_DATA_ROOT    = Path(os.environ.get('CHOROS_DATA_ROOT', '/srv/CHOROS/data'))
sys.path.insert(0, str(_TRAINING_DIR))
from gpu_profiles import get_gpu_profile, print_gpu_profile, _HIGH_VRAM_GB, _detect_attn_backend

import torch

# ---------------------------------------------------------------------------
# Search space definitions
# ---------------------------------------------------------------------------

# All embed_dim choices (64, 128, 192, 256) are divisible by both 4 and 8,
# so restricting to [4, 8] avoids Optuna's CategoricalDistribution dynamic-space error.
_N_HEADS_CHOICES = [4, 8]

# Architecture presets for Pose-JEPA model_shape hyperparameter.
_MODEL_SHAPES_POSEJEPA: dict[str, dict] = {
    # 'small':  {'embed_dim': 128, 'ffn_dim': 512,  'pred_ffn_dim': 256,  'n_heads': 4, 'n_layers': 4},
    # 'medium': {'embed_dim': 258, 'ffn_dim': 1032, 'pred_ffn_dim': 516,  'n_heads': 6, 'n_layers': 6},
    'medium_v2': {'embed_dim': 256, 'ffn_dim': 1024, 'pred_ffn_dim': 512,  'n_heads': 4, 'n_layers': 6},
    # 'large':  {'embed_dim': 512, 'ffn_dim': 2048, 'pred_ffn_dim': 1024, 'n_heads': 8, 'n_layers': 8},
}

# Translates feature_group_mask choice to --feature_group_mask CLI args
_FEATURE_MASK_ARGS = {
    'none': [],
    'P':    ['P'],
    'V':    ['V'],
    'A':    ['A'],
    'J':    ['J'],
    'PV':   ['P', 'V'],
    'PA':   ['P', 'A'],
    'PJ':   ['P', 'J'],
    'VA':   ['V', 'A'],
    'VJ':   ['V', 'J'],
    'AJ':   ['A', 'J'],
    'PVA':  ['P', 'V', 'A'],
    'PVJ':  ['P', 'V', 'J'],
    'VAJ':  ['V', 'A', 'J'],
}

# ---------------------------------------------------------------------------
# Hyperparameter sampling
# ---------------------------------------------------------------------------

def sample_hyperparams(trial: optuna.Trial) -> dict:
    embed_dim   = trial.suggest_categorical('embed_dim', [64, 128, 192, 256])
    n_heads     = trial.suggest_categorical('n_heads', _N_HEADS_CHOICES)
    mask_type   = trial.suggest_categorical('mask_type', ['random', 'span'])
    params = {
        'mask_ratio':          trial.suggest_float('mask_ratio', 0.25, 0.75),
        'mask_type':           mask_type,
        'n_span_blocks':       trial.suggest_int('n_span_blocks', 2, 8) if mask_type == 'span' else 4,
        # 'feature_group_mask':  trial.suggest_categorical('feature_group_mask', ['none', 'P', 'V', 'A', 'J', 'PV', 'PA', 'PJ', 'VA', 'VJ', 'AJ', 'PVA', 'PVJ', 'VAJ', 'PAJ']),
        'feature_group_mask':  trial.suggest_categorical('feature_group_mask', ['none']),
        'lr':                  trial.suggest_float('lr', 5e-5, 5e-3, log=True),
        'sampling_alpha':      trial.suggest_float('sampling_alpha', 0.35, 0.65),
        'dropout':             trial.suggest_float('dropout', 0.0, 0.25),
        'embed_dim':           embed_dim,
        'n_heads':             n_heads,
        'n_layers':            trial.suggest_int('n_layers', 2, 8),
        'ffn_dim':             trial.suggest_categorical('ffn_dim', [256, 512, 768, 1024]),
        'max_len':             trial.suggest_categorical('max_len', [64, 128, 256]),
    }
    return params


def sample_hyperparams_tsjepa(trial: optuna.Trial) -> dict:
    embed_dim = trial.suggest_categorical('embed_dim', [64, 128, 192, 256])
    n_heads   = trial.suggest_categorical('n_heads', _N_HEADS_CHOICES)
    mask_type = trial.suggest_categorical('mask_type', ['span', 'random'])
    params = {
        'target_ratio':        trial.suggest_float('target_ratio', 0.15, 0.50),
        'mask_type':           mask_type,
        'n_target_blocks':     trial.suggest_int('n_target_blocks', 1, 4) if mask_type == 'span' else 2,
        'feature_group_mask':  trial.suggest_categorical('feature_group_mask', ['none']),
        'lr':                  trial.suggest_float('lr', 5e-5, 5e-3, log=True),
        'sampling_alpha':      trial.suggest_float('sampling_alpha', 0.35, 0.65),
        'dropout':             trial.suggest_float('dropout', 0.0, 0.25),
        'embed_dim':           embed_dim,
        'n_heads':             n_heads,
        'n_layers':            trial.suggest_int('n_layers', 2, 8),
        'ffn_dim':             trial.suggest_categorical('ffn_dim', [256, 512, 768, 1024]),
        'max_len':             trial.suggest_categorical('max_len', [64, 128, 256]),
        'ema_start':           trial.suggest_float('ema_start', 0.990, 0.999),
        'pred_layers':         trial.suggest_int('pred_layers', 2, 4),
        'pred_ffn_dim':        trial.suggest_categorical('pred_ffn_dim', [256, 512, 768]),
    }
    return params


def sample_hyperparams_posejepa(trial: optuna.Trial) -> dict:
    model_shape = trial.suggest_categorical('model_shape', list(_MODEL_SHAPES_POSEJEPA))
    shape       = _MODEL_SHAPES_POSEJEPA[model_shape]
    target_mode = trial.suggest_categorical('target_mode', ['masked_span'])

    # future_min_gap and future_horizon only matter when future paths are possible.
    uses_future = target_mode in ('future', 'mixed')
    future_min_gap = (
        trial.suggest_categorical('future_min_gap', [1, 2])
        if uses_future else 2
    )
    future_horizon = (
        trial.suggest_categorical('future_horizon', ['short', 'medium', 'long'])
        if uses_future else 'medium'
    )

    return {
        'model_shape':      model_shape,
        'patch_size':       trial.suggest_int('patch_size', 4, 8, step=4),
        # 'target_ratio':     trial.suggest_float('target_ratio', 0.25, 0.75, step=0.25),
        'target_ratio':     trial.suggest_categorical('target_ratio', [0.75]),
        'target_mode':      target_mode,
        # 'n_target_blocks':  trial.suggest_int('n_target_blocks', 2, 4),
        'n_target_blocks':  trial.suggest_categorical('n_target_blocks', [2]),
        'future_min_gap':   future_min_gap,
        'future_horizon':   future_horizon,
        'ema_start':        trial.suggest_categorical('ema_start', [0.992, 0.994, 0.996, 0.998]),
        # 'pred_layers':      trial.suggest_int('pred_layers', 1, 3),
        'pred_layers':      trial.suggest_categorical('pred_layers', [1]),
        'pred_ffn_dim':     shape['pred_ffn_dim'],
        # 'latent_loss':        trial.suggest_categorical('latent_loss', ['smooth_l1']),
        'latent_loss':        trial.suggest_categorical('latent_loss', ['cosine']),
        # Use new Optuna names for continuous versions to avoid distribution
        # conflicts when a new run accidentally reuses an older categorical study.
        'lr':                 trial.suggest_float('lr_cont', 1e-5, 1e-3, log=True),
        'min_lr':             trial.suggest_categorical('min_lr', [1e-7, 1e-6, 1e-5]),
        'weight_decay':       trial.suggest_float('weight_decay_cont', 1e-4, 1e-1, log=True),
        'sampling_alpha':     trial.suggest_categorical('sampling_alpha', [0.5]),
        # 'dropout':            trial.suggest_float('dropout', 0.0, 0.15, step=0.05),
        'dropout':            trial.suggest_categorical('dropout', [0.0, 0.05, 0.1]),
        'embed_dim':          shape['embed_dim'],
        'n_heads':            shape['n_heads'],
        'n_layers':           shape['n_layers'],
        'ffn_dim':            shape['ffn_dim'],
        # max_len must be divisible by patch_size (8)
        'max_len':            trial.suggest_categorical('max_len', [64, 128]),
        'samples_per_epoch':  trial.suggest_categorical('samples_per_epoch', [256000]),
        'eval_window_pool':   trial.suggest_categorical('eval_window_pool', ['mean_std']),
    }


# ---------------------------------------------------------------------------
# Command builder
# ---------------------------------------------------------------------------

def _profile_for_gpu(gpu_index: int) -> dict:
    """Return GPU profile for a specific physical device index."""
    try:
        props   = torch.cuda.get_device_properties(gpu_index)
        vram_gb = props.total_memory / (1024 ** 3)
        high    = vram_gb >= _HIGH_VRAM_GB
        precision = 'bf16' if props.major >= 8 else 'fp16'
        backend   = _detect_attn_backend(props.major)
        return {
            'batch_size':   256 if high else 256,
            'num_workers':  4 if high else 4,
            'compile':      False,  # always off for hparam trials
            'precision':    precision,
            'attn_backend': backend,
            'profile_name': 'high-VRAM' if high else 'low-VRAM',
            'gpu_name':     props.name,
            'vram_gb':      vram_gb,
        }
    except Exception:
        return get_gpu_profile()  # fallback: query device 0


_FIXED_SEED = 42

_FUTURE_HORIZON_MAP: dict[str, tuple[int, int]] = {
    'short':  (1, 4),
    'medium': (2, 8),
    'long':   (4, 12),
}


def _warmup_epochs_posejepa(epochs: int) -> int:
    return round(epochs * 0.1)


def _param_alias(params: dict, primary: str, alias: str, default=None):
    """Read a parameter that may have a new Optuna name in newer studies."""
    if primary in params:
        return params[primary]
    if alias in params:
        return params[alias]
    return default


def build_cmd(params: dict, gpu: int, profile: dict, epochs: int = 10, eval_interval: int = 10) -> list[str]:
    train_script = str(_CHOROS_ROOT / 'training' / 'train_vr_encoder.py')
    cmd = [
        'conda', 'run', '-n', 'CHOROS',
        'python', train_script,
        '--npy_dir',            str(_DATA_ROOT / 'kinematics' / 'VR_npy_PVAJ'),
        '--out_dir',            str(_CHOROS_ROOT / 'outputs' / 'checkpoints'),
        '--epochs',             str(epochs),
        '--embed_eval_interval',str(eval_interval),
        '--batch_size',         str(profile['batch_size']),
        '--num_workers',        str(profile['num_workers']),
        '--precision',          profile['precision'],
        '--warmup_epochs',      '2',
        '--val_fraction',       '0.1',
        '--min_lr',             '1e-6',
        '--kinematics',         'PVAJ',
        '--samples_per_epoch',  '65536',
        '--seed',               str(_FIXED_SEED),
        '--no_compile',
        '--eval_window_pool',   'mean',
        '--eval_session_pool',  'mean',
        '--eval_split_mode',    'val',
        '--mask_ratio',         str(params['mask_ratio']),
        '--mask_type',          params['mask_type'],
        '--n_span_blocks',      str(params['n_span_blocks']),
        '--lr',                 str(params['lr']),
        '--sampling_alpha',     str(params['sampling_alpha']),
        '--dropout',            str(params['dropout']),
        '--embed_dim',          str(params['embed_dim']),
        '--n_heads',            str(params['n_heads']),
        '--n_layers',           str(params['n_layers']),
        '--ffn_dim',            str(params['ffn_dim']),
        '--max_len',            str(params['max_len']),
    ]
    fgm_args = _FEATURE_MASK_ARGS[params['feature_group_mask']]
    if fgm_args:
        cmd += ['--feature_group_mask'] + fgm_args
    return cmd


def build_cmd_tsjepa(params: dict, gpu: int, profile: dict, epochs: int = 10, eval_interval: int = 10) -> list[str]:
    train_script = str(_CHOROS_ROOT / 'training' / 'train_vr_encoder_tsjepa.py')
    cmd = [
        'conda', 'run', '-n', 'CHOROS',
        'python', train_script,
        '--npy_dir',            str(_DATA_ROOT / 'kinematics' / 'VR_npy_PVAJ'),
        '--out_dir',            str(_CHOROS_ROOT / 'outputs' / 'checkpoints'),
        '--epochs',             str(epochs),
        '--embed_eval_interval',str(eval_interval),
        '--batch_size',         str(profile['batch_size']),
        '--num_workers',        str(profile['num_workers']),
        '--precision',          profile['precision'],
        '--warmup_epochs',      '2',
        '--val_fraction',       '0.1',
        '--min_lr',             '1e-6',
        '--kinematics',         'PVAJ',
        '--samples_per_epoch',  '65536',
        '--seed',               str(_FIXED_SEED),
        '--no_compile',
        '--eval_window_pool',   'mean',
        '--eval_session_pool',  'mean',
        '--eval_split_mode',    'val',
        '--target_ratio',       str(params['target_ratio']),
        '--mask_type',          params['mask_type'],
        '--n_target_blocks',    str(params['n_target_blocks']),
        '--ema_start',          str(params['ema_start']),
        '--pred_layers',        str(params['pred_layers']),
        '--pred_ffn_dim',       str(params['pred_ffn_dim']),
        '--lr',                 str(params['lr']),
        '--sampling_alpha',     str(params['sampling_alpha']),
        '--dropout',            str(params['dropout']),
        '--embed_dim',          str(params['embed_dim']),
        '--n_heads',            str(params['n_heads']),
        '--n_layers',           str(params['n_layers']),
        '--ffn_dim',            str(params['ffn_dim']),
        '--max_len',            str(params['max_len']),
    ]
    fgm_args = _FEATURE_MASK_ARGS[params['feature_group_mask']]
    if fgm_args:
        cmd += ['--feature_group_mask'] + fgm_args
    return cmd


def build_cmd_posejepa(params: dict, gpu: int, profile: dict, epochs: int = 100,
                       eval_interval: int = 20, kinematics: str = 'P') -> list[str]:
    train_script = str(_CHOROS_ROOT / 'training' / 'train_vr_encoder_pose_jepa.py')
    fh_min, fh_max = _FUTURE_HORIZON_MAP[params['future_horizon']]
    return [
        'conda', 'run', '-n', 'CHOROS',
        'python', train_script,
        '--npy_dir',            str(_DATA_ROOT / 'kinematics' / 'VR_npy_PVAJ'),
        '--out_dir',            str(_CHOROS_ROOT / 'outputs' / 'checkpoints'),
        '--epochs',             str(epochs),
        '--embed_eval_interval',str(eval_interval),
        '--batch_size',         str(profile['batch_size']),
        '--num_workers',        str(profile['num_workers']),
        '--precision',          profile['precision'],
        '--warmup_epochs',      str(_warmup_epochs_posejepa(epochs)),
        '--val_fraction',       '0.15',
        '--min_lr',             str(params['min_lr']),
        '--kinematics',         kinematics,
        '--seed',               str(_FIXED_SEED),
        '--no_compile',
        '--eval_session_pool',  'mean',
        '--eval_split_mode',    'val',
        '--eval_metrics',       'portScore', 'bot_dist_mean_s3', 'firing_accuracy_AOBJ_s3',
        '--samples_per_epoch',  str(params['samples_per_epoch']),
        '--eval_window_pool',   params['eval_window_pool'],
        '--stride_factor',      '2',
        '--patch_size',         str(params['patch_size']),
        '--target_ratio',       str(params['target_ratio']),
        '--target_mode',        params['target_mode'],
        '--n_target_blocks',    str(params['n_target_blocks']),
        '--future_min_gap',     str(params['future_min_gap']),
        '--future_horizon_min', str(fh_min),
        '--future_horizon_max', str(fh_max),
        '--ema_start',          str(params['ema_start']),
        '--pred_layers',        str(params['pred_layers']),
        '--pred_ffn_dim',       str(params['pred_ffn_dim']),
        '--latent_loss',        params['latent_loss'],
        '--embed_pool',         'mean',
        '--lr',                 str(params['lr']),
        '--weight_decay',       str(params['weight_decay']),
        '--sampling_alpha',     str(params['sampling_alpha']),
        '--dropout',            str(params['dropout']),
        '--embed_dim',          str(params['embed_dim']),
        '--n_heads',            str(params['n_heads']),
        '--n_layers',           str(params['n_layers']),
        '--ffn_dim',            str(params['ffn_dim']),
        '--max_len',            str(params['max_len']),
    ]


# ---------------------------------------------------------------------------
# VRAM monitoring
# ---------------------------------------------------------------------------

def _vram_monitor(gpu_id: int, peak_mb: list, stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        try:
            result = subprocess.run(
                ['nvidia-smi', f'--id={gpu_id}',
                 '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5,
            )
            mb = int(result.stdout.strip())
            peak_mb[0] = max(peak_mb[0], mb)
        except Exception:
            pass
        stop_event.wait(5.0)


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

_PROBE_SUMMARY_RE = re.compile(
    r'\[Probe\] objective=(\w+)\s+target=(\S+)\s+split=(\w+)'
    r'\s+Balanced Acc\.\:\s*([\d.]+)'
    r'(?:\s+Acc\.\:\s*[\d.]+)?'
    r'\s+MCC\:\s*(-?[\d.]+)'
    r'\s+F1\(macro\)\:\s*([\d.]+)'
    r'\s+ROC-AUC\:\s*([\d.]+)'
)

_TARGET_KEYS = frozenset([
    'portScore',
    'bot_dist_mean_s3',
    'firing_accuracy_AOBJ_s3',
])


def parse_eval_output(text: str) -> dict[str, dict]:
    """
    Parse _periodic_eval() stdout.  Returns {target: {'bacc', 'mcc', 'f1', 'auc'}}
    for each of the 3 downstream objectives, reading the compact [Probe] summary lines
    with split=val.  flag_incidents_s3 is excluded due to severe val-set imbalance.
    """
    results: dict[str, dict] = {}
    for line in text.splitlines():
        m = _PROBE_SUMMARY_RE.search(line)
        if m and m.group(3) == 'val':
            target = m.group(2)
            if target in _TARGET_KEYS:
                results[target] = {
                    'bacc': float(m.group(4)),
                    'mcc':  float(m.group(5)),
                    'f1':   float(m.group(6)),
                    'auc':  float(m.group(7)),
                }
    return results


def aggregate_metric(
    metrics: dict[str, dict],
    auc_weight: float = 0.5,
    mcc_weight: float = 0.5,
    weakest_target_weight: float = 0.25,
) -> float:
    keys = ['portScore', 'bot_dist_mean_s3', 'firing_accuracy_AOBJ_s3']
    weight_sum = auc_weight + mcc_weight
    if weight_sum <= 0:
        raise ValueError('auc_weight + mcc_weight must be positive')
    if not 0.0 <= weakest_target_weight <= 1.0:
        raise ValueError('weakest_target_weight must be in [0, 1]')
    task_scores = []
    for k in keys:
        auc = metrics[k]['auc']
        mcc_rescaled = (metrics[k]['mcc'] + 1.0) / 2.0
        task_scores.append(
            (auc_weight * auc + mcc_weight * mcc_rescaled) / weight_sum
        )
    hmean = harmonic_mean(task_scores)
    weakest = min(task_scores)
    return (1.0 - weakest_target_weight) * hmean + weakest_target_weight * weakest


def _canonical_params(params: dict) -> str:
    """Stable representation for exact duplicate-trial checks."""
    return json.dumps(params, sort_keys=True, separators=(',', ':'), default=str)


def _is_duplicate_trial(trial: optuna.Trial) -> bool:
    current = _canonical_params(trial.params)
    for other in trial.study.get_trials(deepcopy=False):
        if other.number == trial.number:
            continue
        if other.state not in (
            optuna.trial.TrialState.COMPLETE,
            optuna.trial.TrialState.RUNNING,
            optuna.trial.TrialState.WAITING,
        ):
            continue
        if _canonical_params(other.params) == current:
            trial.set_user_attr('duplicate_of', other.number)
            return True
    return False


def aggregate_score_history(
    scores: list[float],
    mode: str = 'stable_tail',
    tail_k: int = 3,
    volatility_penalty: float = 0.5,
    trend_weight: float = 0.1,
) -> float:
    """
    Convert per-eval aggregate probe scores into the Optuna trial value.

    The default stable_tail mode rewards sustained recent performance rather
    than a single lucky final checkpoint:
      mean(last K scores) - penalty * std(last K) + trend_weight * (last - first)
    """
    if not scores:
        return float('nan')
    if mode == 'last':
        return scores[-1]
    tail = scores[-max(1, tail_k):]
    tail_mean = float(sum(tail) / len(tail))
    if mode == 'mean_tail':
        return tail_mean
    if mode != 'stable_tail':
        raise ValueError(f'unknown score history mode: {mode}')
    tail_std = float(np.std(tail)) if len(tail) > 1 else 0.0
    trend = float(scores[-1] - scores[0]) if len(scores) > 1 else 0.0
    return tail_mean - volatility_penalty * tail_std + trend_weight * trend


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _log_trial_summary(
    trial: optuna.Trial,
    params: dict,
    score: float,
    metrics: dict[str, dict],
    elapsed: float,
    peak_vram_gb: float,
) -> None:
    fgm  = params['feature_group_mask']
    vram = f'{peak_vram_gb:.1f}GB' if peak_vram_gb > 0 else 'N/A'
    def _a(key): return metrics.get(key, {}).get('auc', float('nan'))
    print(
        f"[T {trial.number:03d}] score={score:.4f} | "
        f"fab={_a('portScore'):.4f} "
        f"bot={_a('bot_dist_mean_s3'):.4f} "
        f"firing={_a('firing_accuracy_AOBJ_s3'):.4f} | "
        f"vram={vram} | "
        f"mask_ratio={params['mask_ratio']:.3f} "
        f"mask_type={params['mask_type']} "
        f"lr={params['lr']:.2e} "
        f"embed_dim={params['embed_dim']} "
        f"n_heads={params['n_heads']} "
        f"n_layers={params['n_layers']} "
        f"ffn_dim={params['ffn_dim']} "
        f"dropout={params['dropout']:.3f} "
        f"max_len={params['max_len']} "
        f"sa={params['sampling_alpha']:.3f} "
        f"fgmask={fgm} | "
        f"{elapsed:.0f}s",
        flush=True,
    )


def _log_trial_failure(trial_number: int, params: dict, reason: str, elapsed: float) -> None:
    fail_dir = _CHOROS_ROOT / 'outputs' / 'hparam_search'
    fail_dir.mkdir(parents=True, exist_ok=True)
    with open(fail_dir / 'failed_trials.log', 'a') as f:
        f.write(f'\n[T {trial_number:03d}] FAILED ({elapsed:.0f}s) — {reason[:200]}\n')
        f.write(f'  params: {params}\n')


def _progress_callback(study: optuna.Study, trial: optuna.Trial) -> None:
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        return
    best = study.best_trial
    print(
        f'  → running best: [T {best.number:03d}] score={best.value:.4f}  '
        f'({len(completed)} complete)',
        flush=True,
    )


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

_DEFAULT_EPOCHS: dict[str, int] = {'mae': 10, 'tsjepa': 10, 'posejepa': 100}
_DEFAULT_EVAL_INTERVAL: dict[str, int] = {'mae': 10, 'tsjepa': 10, 'posejepa': 20}


def run_trial(trial: optuna.Trial, gpu: int, profile: dict, objective: str = 'mae',
              trial_timeout: int = 1500, trial_epochs: int | None = None,
              eval_interval: int | None = None, kinematics: str = 'P',
              auc_weight: float = 0.5, mcc_weight: float = 0.5,
              weakest_target_weight: float = 0.25,
              score_mode: str = 'stable_tail', score_tail_k: int = 3,
              volatility_penalty: float = 0.5,
              trend_weight: float = 0.1) -> float:
    epochs   = trial_epochs  if trial_epochs  is not None else _DEFAULT_EPOCHS[objective]
    interval = eval_interval if eval_interval is not None else _DEFAULT_EVAL_INTERVAL[objective]

    if objective == 'tsjepa':
        params = sample_hyperparams_tsjepa(trial)
        cmd    = build_cmd_tsjepa(params, gpu, profile, epochs, interval)
    elif objective == 'posejepa':
        params = sample_hyperparams_posejepa(trial)
        cmd    = build_cmd_posejepa(params, gpu, profile, epochs, interval, kinematics)
    else:
        params = sample_hyperparams(trial)
        cmd    = build_cmd(params, gpu, profile, epochs, interval)
    if _is_duplicate_trial(trial):
        dup = trial.user_attrs.get('duplicate_of')
        print(f'[T {trial.number:03d}] duplicate of trial {dup} — pruning before training',
              flush=True)
        raise optuna.exceptions.TrialPruned()
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)}

    peak_mb    = [0]
    stop_event = threading.Event()
    monitor    = threading.Thread(
        target=_vram_monitor, args=(gpu, peak_mb, stop_event), daemon=True
    )
    monitor.start()

    t0   = time.time()
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
    )

    stdout_lines:  list[str] = []
    stderr_lines:  list[str] = []
    current_round: dict      = {}
    score_history: list[float] = []
    metrics_history: list[dict[str, dict]] = []
    step_counter = [0]
    pruned_flag  = [False]

    def _read_stdout() -> None:
        for line in proc.stdout:
            stdout_lines.append(line)
            m = _PROBE_SUMMARY_RE.search(line)
            if m and m.group(3) == 'val':
                target = m.group(2)
                if target in _TARGET_KEYS:
                    current_round[target] = {
                        'bacc': float(m.group(4)),
                        'mcc':  float(m.group(5)),
                        'f1':   float(m.group(6)),
                        'auc':  float(m.group(7)),
                    }
                    if _TARGET_KEYS <= current_round.keys():
                        round_metrics = {k: dict(v) for k, v in current_round.items()}
                        score = aggregate_metric(
                            round_metrics,
                            auc_weight=auc_weight,
                            mcc_weight=mcc_weight,
                            weakest_target_weight=weakest_target_weight,
                        )
                        score_history.append(score)
                        metrics_history.append(round_metrics)
                        reported_score = aggregate_score_history(
                            score_history,
                            mode=score_mode,
                            tail_k=score_tail_k,
                            volatility_penalty=volatility_penalty,
                            trend_weight=trend_weight,
                        )
                        trial.report(reported_score, step_counter[0])
                        step_counter[0] += 1
                        current_round.clear()
                        if trial.should_prune():
                            print(
                                f'[T {trial.number:03d}] pruned at step {step_counter[0] - 1} '
                                f'(score={reported_score:.4f}, instant={score:.4f})',
                                flush=True,
                            )
                            proc.kill()
                            pruned_flag[0] = True
                            return

    def _read_stderr() -> None:
        for line in proc.stderr:
            stderr_lines.append(line)

    reader     = threading.Thread(target=_read_stdout, daemon=True)
    err_reader = threading.Thread(target=_read_stderr, daemon=True)
    reader.start()
    err_reader.start()

    reader.join(timeout=trial_timeout)
    timed_out = reader.is_alive()
    if timed_out:
        proc.kill()
    err_reader.join(timeout=5)
    proc.wait()
    stop_event.set()

    elapsed      = time.time() - t0
    peak_vram_gb = peak_mb[0] / 1024.0

    if timed_out:
        _log_trial_failure(trial.number, params, f'timeout ({trial_timeout}s)', elapsed)
        raise optuna.exceptions.TrialPruned()

    if pruned_flag[0]:
        raise optuna.exceptions.TrialPruned()

    if proc.returncode != 0:
        reason = (''.join(stderr_lines) or ''.join(stdout_lines))[-500:]
        _log_trial_failure(trial.number, params, reason, elapsed)
        raise optuna.exceptions.TrialPruned()

    stdout  = ''.join(stdout_lines)
    metrics = metrics_history[-1] if metrics_history else parse_eval_output(stdout)

    missing = _TARGET_KEYS - metrics.keys()
    if missing:
        _log_trial_failure(
            trial.number, params,
            f'missing metrics: {missing}',
            elapsed,
        )
        raise optuna.exceptions.TrialPruned()

    if not score_history:
        score_history = [aggregate_metric(
            metrics,
            auc_weight=auc_weight,
            mcc_weight=mcc_weight,
            weakest_target_weight=weakest_target_weight,
        )]
    score = aggregate_score_history(
        score_history,
        mode=score_mode,
        tail_k=score_tail_k,
        volatility_penalty=volatility_penalty,
        trend_weight=trend_weight,
    )
    instant_final_score = score_history[-1]
    tail = score_history[-max(1, score_tail_k):]

    for short, key in [('fab',    'portScore'),
                       ('bot',    'bot_dist_mean_s3'),
                       ('firing', 'firing_accuracy_AOBJ_s3')]:
        md = metrics[key]
        trial.set_user_attr(f'{short}_val_bacc', md['bacc'])
        trial.set_user_attr(f'{short}_val_mcc',  md['mcc'])
        trial.set_user_attr(f'{short}_val_f1',   md['f1'])
        trial.set_user_attr(f'{short}_val_auc',  md['auc'])
    trial.set_user_attr('score_mode', score_mode)
    trial.set_user_attr('auc_weight', auc_weight)
    trial.set_user_attr('mcc_weight', mcc_weight)
    trial.set_user_attr('weakest_target_weight', weakest_target_weight)
    trial.set_user_attr('score_history', [round(s, 6) for s in score_history])
    trial.set_user_attr('final_instant_score', round(instant_final_score, 6))
    trial.set_user_attr('tail_mean_score', round(float(sum(tail) / len(tail)), 6))
    trial.set_user_attr('tail_std_score', round(float(np.std(tail)) if len(tail) > 1 else 0.0, 6))
    trial.set_user_attr('score_trend', round(float(score_history[-1] - score_history[0]) if len(score_history) > 1 else 0.0, 6))
    trial.set_user_attr('elapsed_s',    round(elapsed))
    trial.set_user_attr('peak_vram_gb', round(peak_vram_gb, 2))

    if objective == 'tsjepa':
        _log_trial_summary_tsjepa(trial, params, score, metrics, elapsed, peak_vram_gb)
    elif objective == 'posejepa':
        _log_trial_summary_posejepa(trial, params, score, metrics, elapsed, peak_vram_gb)
    else:
        _log_trial_summary(trial, params, score, metrics, elapsed, peak_vram_gb)
    return score


def _log_trial_summary_tsjepa(
    trial: optuna.Trial,
    params: dict,
    score: float,
    metrics: dict[str, dict],
    elapsed: float,
    peak_vram_gb: float,
) -> None:
    fgm  = params['feature_group_mask']
    vram = f'{peak_vram_gb:.1f}GB' if peak_vram_gb > 0 else 'N/A'
    def _a(key): return metrics.get(key, {}).get('auc', float('nan'))
    print(
        f"[T {trial.number:03d}] score={score:.4f} | "
        f"fab={_a('portScore'):.4f} "
        f"bot={_a('bot_dist_mean_s3'):.4f} "
        f"firing={_a('firing_accuracy_AOBJ_s3'):.4f} | "
        f"vram={vram} | "
        f"target_ratio={params['target_ratio']:.3f} "
        f"mask_type={params['mask_type']} "
        f"lr={params['lr']:.2e} "
        f"embed_dim={params['embed_dim']} "
        f"n_heads={params['n_heads']} "
        f"n_layers={params['n_layers']} "
        f"ffn_dim={params['ffn_dim']} "
        f"dropout={params['dropout']:.3f} "
        f"max_len={params['max_len']} "
        f"sa={params['sampling_alpha']:.3f} "
        f"ema_start={params['ema_start']:.4f} "
        f"pred_layers={params['pred_layers']} "
        f"pred_ffn_dim={params['pred_ffn_dim']} "
        f"fgmask={fgm} | "
        f"{elapsed:.0f}s",
        flush=True,
    )


def _log_trial_summary_posejepa(
    trial: optuna.Trial,
    params: dict,
    score: float,
    metrics: dict[str, dict],
    elapsed: float,
    peak_vram_gb: float,
) -> None:
    vram = f'{peak_vram_gb:.1f}GB' if peak_vram_gb > 0 else 'N/A'
    def _a(key): return metrics.get(key, {}).get('auc', float('nan'))
    tmode = params['target_mode']
    future_info = (
        f"fmg={params['future_min_gap']} fh={params['future_horizon']} "
        if tmode in ('future', 'mixed') else ''
    )
    print(
        f"[T {trial.number:03d}] score={score:.4f} | "
        f"fab={_a('portScore'):.4f} "
        f"bot={_a('bot_dist_mean_s3'):.4f} "
        f"firing={_a('firing_accuracy_AOBJ_s3'):.4f} | "
        f"vram={vram} | "
        f"ps={params['patch_size']} "
        f"tmode={tmode} "
        f"tr={params['target_ratio']:.3f} "
        f"{future_info}"
        f"shape={params['model_shape']} "
        f"loss={params['latent_loss']} "
        f"lr={params['lr']:.2e} "
        f"minlr={params['min_lr']:.1e} "
        f"wd={params['weight_decay']:.1e} "
        f"ml={params['max_len']} "
        f"spe={params['samples_per_epoch']} "
        f"wp={params['eval_window_pool']} "
        f"sa={params['sampling_alpha']:.2f} "
        f"ema={params['ema_start']} "
        f"pred={params['pred_layers']}x{params['pred_ffn_dim']} | "
        f"{elapsed:.0f}s",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Study creation
# ---------------------------------------------------------------------------

def create_study(study_name: str, worker_offset: int = 0,
                 storage_url: str | None = None,
                 pruner: optuna.pruners.BasePruner | None = None,
                 sampler_startup: int = 40,
                 n_ei_candidates: int = 64,
                 multivariate_tpe: bool = True) -> optuna.Study:
    if storage_url is None:
        db_path = _CHOROS_ROOT / 'outputs' / 'hparam_search' / 'optuna.db'
        db_path.parent.mkdir(parents=True, exist_ok=True)
        storage_url = f'sqlite:///{db_path}'

    sampler_kwargs = {
        'n_startup_trials': sampler_startup,
        'n_ei_candidates': n_ei_candidates,
        'seed': 42 + worker_offset,
    }
    if multivariate_tpe:
        sampler_kwargs.update({'multivariate': True, 'group': True})
    sampler = optuna.samplers.TPESampler(**sampler_kwargs)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        sampler=sampler,
        pruner=pruner,
        direction='maximize',
        load_if_exists=True,
    )
    return study


# ---------------------------------------------------------------------------
# Results export
# ---------------------------------------------------------------------------

def write_tsv_results(study: optuna.Study, out_path: Path) -> None:
    """Write top trials to a TSV compatible with runs/results.tsv schema."""
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        return

    completed.sort(key=lambda t: t.value or 0.0, reverse=True)
    rows = []
    for t in completed:
        ver  = f'hps_{study.study_name}_t{t.number:03d}'
        agg  = t.value or 0.0
        vram = t.user_attrs.get('peak_vram_gb', 'N/A')
        desc = (
            f"TPE trial {t.number}, aggregate={agg:.4f}, "
            f"score_mode={t.user_attrs.get('score_mode', 'unknown')}, "
            f"auc_weight={t.user_attrs.get('auc_weight', 'unknown')}, "
            f"mcc_weight={t.user_attrs.get('mcc_weight', 'unknown')}, "
            f"weakest_target_weight={t.user_attrs.get('weakest_target_weight', 'unknown')}, "
            f"score_history={t.user_attrs.get('score_history', [])}, "
            + ', '.join(f'{k}={v}' for k, v in sorted(t.params.items()))
        )
        for target, col, short in [
            ('FAB',       'portScore',               'fab'),
            ('DEVCOM_D3', 'bot_dist_mean_s3',         'bot'),
            ('DEVCOM_D3', 'firing_accuracy_AOBJ_s3',  'firing'),
        ]:
            bacc = t.user_attrs.get(f'{short}_val_bacc', float('nan'))
            mcc  = t.user_attrs.get(f'{short}_val_mcc',  float('nan'))
            f1   = t.user_attrs.get(f'{short}_val_f1',   float('nan'))
            auc  = t.user_attrs.get(f'{short}_val_auc',  float('nan'))
            perf = f'val_bacc={bacc:.4f} val_mcc={mcc:.4f} val_f1={f1:.4f} val_auc={auc:.4f}'
            rows.append('\t'.join([ver, target, col, perf, 'hps', str(vram), desc]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        f.write('version\ttarget\tobjective\tperformance\tstatus\tvram_usage\tdescription\n')
        f.write('\n'.join(rows) + '\n')
    print(f'\nResults written to {out_path}')


def _print_best_trial(study: optuna.Study) -> None:
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned    = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]

    print(f'\n{"="*72}')
    print(f'Hyperparameter Search Complete')
    print(f'Study: {study.study_name}   '
          f'Trials: {len(completed)} complete, {len(pruned)} pruned')

    if not completed:
        print('No completed trials.')
        return

    best = study.best_trial
    print(f'\nBest trial: #{best.number}   score={best.value:.4f}')
    print(f'  Score mode/history: {best.user_attrs.get("score_mode", "unknown")}  '
          f'auc_weight={best.user_attrs.get("auc_weight", float("nan"))}  '
          f'mcc_weight={best.user_attrs.get("mcc_weight", float("nan"))}  '
          f'weakest_target_weight={best.user_attrs.get("weakest_target_weight", float("nan"))}  '
          f'history={best.user_attrs.get("score_history", [])}  '
          f'tail_mean={best.user_attrs.get("tail_mean_score", float("nan")):.4f}  '
          f'tail_std={best.user_attrs.get("tail_std_score", float("nan")):.4f}  '
          f'trend={best.user_attrs.get("score_trend", float("nan")):.4f}')
    print(f'  FAB portScore:           auc={best.user_attrs.get("fab_val_auc", float("nan")):.4f}  '
          f'mcc={best.user_attrs.get("fab_val_mcc", float("nan")):.4f}')
    print(f'  DEVCOM bot_dist:         auc={best.user_attrs.get("bot_val_auc", float("nan")):.4f}  '
          f'mcc={best.user_attrs.get("bot_val_mcc", float("nan")):.4f}')
    print(f'  DEVCOM firing_accuracy:  auc={best.user_attrs.get("firing_val_auc", float("nan")):.4f}  '
          f'mcc={best.user_attrs.get("firing_val_mcc", float("nan")):.4f}')
    print(f'  Peak VRAM: {best.user_attrs.get("peak_vram_gb", "N/A")} GB')
    print('  Params:')
    for k, v in sorted(best.params.items()):
        print(f'    {k:25s} = {v}')

    print(f'\nTop 5 by aggregate score (harmonic mean val AUC):')
    print(f'  {"Rank":4s}  {"Trial":5s}  {"score":6s}  {"fab_auc":8s}  '
          f'{"bot_auc":8s}  {"firing_auc":10s}')
    for rank, t in enumerate(completed[:5], 1):
        print(
            f'  {rank:4d}  {t.number:5d}  {(t.value or 0):.4f}  '
            f'{t.user_attrs.get("fab_val_auc", float("nan")):.4f}      '
            f'{t.user_attrs.get("bot_val_auc", float("nan")):.4f}      '
            f'{t.user_attrs.get("firing_val_auc", float("nan")):.4f}'
        )
    print('='*72)


# ---------------------------------------------------------------------------
# Phase 2 + 3  (final model training and test evaluation)
# ---------------------------------------------------------------------------

def _run_streaming(cmd: list[str], env: dict) -> tuple[int, str]:
    """Run subprocess with live stdout, return (returncode, full_stdout)."""
    lines: list[str] = []
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
    )
    for line in proc.stdout:
        print(line, end='', flush=True)
        lines.append(line)
    proc.wait()
    return proc.returncode, ''.join(lines)


def build_final_cmd(params: dict, gpu: int, profile: dict, final_epochs: int) -> list[str]:
    """Build train_vr_encoder.py (MAE) command for the final model (eval_split_mode=test)."""
    train_script = str(_CHOROS_ROOT / 'training' / 'train_vr_encoder.py')
    cmd = [
        'conda', 'run', '-n', 'CHOROS',
        'python', train_script,
        '--npy_dir',            str(_DATA_ROOT / 'kinematics' / 'VR_npy_PVAJ'),
        '--out_dir',            str(_CHOROS_ROOT / 'outputs' / 'checkpoints'),
        '--epochs',             str(final_epochs),
        '--embed_eval_interval','5',
        '--eval_split_mode',    'test',
        '--batch_size',         str(profile['batch_size']),
        '--num_workers',        str(profile['num_workers']),
        '--precision',          profile['precision'],
        '--warmup_epochs',      '2',
        '--val_fraction',       '0.1',
        '--min_lr',             '1e-6',
        '--kinematics',         'PVAJ',
        '--samples_per_epoch',  '65536',
        '--seed',               str(_FIXED_SEED),
        '--no_compile',
        '--eval_window_pool',   'mean',
        '--eval_session_pool',  'mean',
        '--mask_ratio',         str(params['mask_ratio']),
        '--mask_type',          params['mask_type'],
        '--n_span_blocks',      str(params.get('n_span_blocks', 4)),
        '--lr',                 str(params['lr']),
        '--sampling_alpha',     str(params['sampling_alpha']),
        '--dropout',            str(params['dropout']),
        '--embed_dim',          str(params['embed_dim']),
        '--n_heads',            str(params['n_heads']),
        '--n_layers',           str(params['n_layers']),
        '--ffn_dim',            str(params['ffn_dim']),
        '--max_len',            str(params['max_len']),
    ]
    fgm_args = _FEATURE_MASK_ARGS[params['feature_group_mask']]
    if fgm_args:
        cmd += ['--feature_group_mask'] + fgm_args
    return cmd


def build_final_cmd_tsjepa(params: dict, gpu: int, profile: dict, final_epochs: int) -> list[str]:
    """Build train_vr_encoder_tsjepa.py command for the final model (eval_split_mode=test)."""
    train_script = str(_CHOROS_ROOT / 'training' / 'train_vr_encoder_tsjepa.py')
    cmd = [
        'conda', 'run', '-n', 'CHOROS',
        'python', train_script,
        '--npy_dir',            str(_DATA_ROOT / 'kinematics' / 'VR_npy_PVAJ'),
        '--out_dir',            str(_CHOROS_ROOT / 'outputs' / 'checkpoints'),
        '--epochs',             str(final_epochs),
        '--embed_eval_interval','5',
        '--eval_split_mode',    'test',
        '--batch_size',         str(profile['batch_size']),
        '--num_workers',        str(profile['num_workers']),
        '--precision',          profile['precision'],
        '--warmup_epochs',      '2',
        '--val_fraction',       '0.1',
        '--min_lr',             '1e-6',
        '--kinematics',         'PVAJ',
        '--samples_per_epoch',  '65536',
        '--seed',               str(_FIXED_SEED),
        '--no_compile',
        '--eval_window_pool',   'mean',
        '--eval_session_pool',  'mean',
        '--target_ratio',       str(params['target_ratio']),
        '--mask_type',          params['mask_type'],
        '--n_target_blocks',    str(params.get('n_target_blocks', 2)),
        '--ema_start',          str(params.get('ema_start', 0.996)),
        '--pred_layers',        str(params.get('pred_layers', 2)),
        '--pred_ffn_dim',       str(params.get('pred_ffn_dim', 256)),
        '--lr',                 str(params['lr']),
        '--sampling_alpha',     str(params['sampling_alpha']),
        '--dropout',            str(params['dropout']),
        '--embed_dim',          str(params['embed_dim']),
        '--n_heads',            str(params['n_heads']),
        '--n_layers',           str(params['n_layers']),
        '--ffn_dim',            str(params['ffn_dim']),
        '--max_len',            str(params['max_len']),
    ]
    fgm_args = _FEATURE_MASK_ARGS[params['feature_group_mask']]
    if fgm_args:
        cmd += ['--feature_group_mask'] + fgm_args
    return cmd


def build_final_cmd_posejepa(params: dict, gpu: int, profile: dict, final_epochs: int,
                             kinematics: str = 'P') -> list[str]:
    """Build train_vr_encoder_pose_jepa.py command for the final model (eval_split_mode=test).

    params is raw Optuna best.params. If it contains 'model_shape', expand the
    architecture preset; otherwise fall back to individual keys for legacy studies.
    """
    train_script = str(_CHOROS_ROOT / 'training' / 'train_vr_encoder_pose_jepa.py')
    if 'model_shape' in params:
        shape        = _MODEL_SHAPES_POSEJEPA[params['model_shape']]
        embed_dim    = shape['embed_dim']
        ffn_dim      = shape['ffn_dim']
        pred_ffn_dim = shape['pred_ffn_dim']
        n_heads      = shape['n_heads']
        n_layers     = shape['n_layers']
    else:
        embed_dim    = params['embed_dim']
        ffn_dim      = params.get('ffn_dim',      embed_dim * params.get('ffn_mult', 4))
        pred_ffn_dim = params.get('pred_ffn_dim', embed_dim * params.get('pred_ffn_mult', 2))
        n_heads      = params['n_heads']
        n_layers     = params['n_layers']
    fh_key        = params.get('future_horizon', 'medium')
    fh_min, fh_max = _FUTURE_HORIZON_MAP[fh_key]
    stride_factor = params.get('stride_factor', 2)
    return [
        'conda', 'run', '-n', 'CHOROS',
        'python', train_script,
        '--npy_dir',            str(_DATA_ROOT / 'kinematics' / 'VR_npy_PVAJ'),
        '--out_dir',            str(_CHOROS_ROOT / 'outputs' / 'checkpoints'),
        '--epochs',             str(final_epochs),
        '--embed_eval_interval','10',
        '--eval_split_mode',    'test',
        '--batch_size',         str(profile['batch_size']),
        '--num_workers',        str(profile['num_workers']),
        '--precision',          profile['precision'],
        '--warmup_epochs',      str(_warmup_epochs_posejepa(final_epochs)),
        '--val_fraction',       '0.15',
        '--min_lr',             str(params.get('min_lr', 1e-6)),
        '--kinematics',         kinematics,
        '--samples_per_epoch',  str(params.get('samples_per_epoch', 0)),
        '--seed',               str(_FIXED_SEED),
        '--no_compile',
        '--eval_window_pool',   params.get('eval_window_pool', 'stat9'),
        '--eval_session_pool',  'mean',
        '--eval_metrics',       'portScore', 'bot_dist_mean_s3', 'firing_accuracy_AOBJ_s3',
        '--stride_factor',      str(stride_factor),
        '--patch_size',         str(params['patch_size']),
        '--target_ratio',       str(params['target_ratio']),
        '--target_mode',        params['target_mode'],
        '--n_target_blocks',    str(params.get('n_target_blocks', 2)),
        '--future_min_gap',     str(params.get('future_min_gap', 2)),
        '--future_horizon_min', str(fh_min),
        '--future_horizon_max', str(fh_max),
        '--ema_start',          str(params.get('ema_start', 0.996)),
        '--pred_layers',        str(params.get('pred_layers', 2)),
        '--pred_ffn_dim',       str(pred_ffn_dim),
        '--latent_loss',        params.get('latent_loss', 'smooth_l1'),
        '--embed_pool',         'mean',
        '--lr',                 str(_param_alias(params, 'lr', 'lr_cont')),
        '--weight_decay',       str(_param_alias(params, 'weight_decay', 'weight_decay_cont', 1e-4)),
        '--sampling_alpha',     str(params['sampling_alpha']),
        '--dropout',            str(params.get('dropout', 0.0)),
        '--embed_dim',          str(embed_dim),
        '--n_heads',            str(n_heads),
        '--n_layers',           str(n_layers),
        '--ffn_dim',            str(ffn_dim),
        '--max_len',            str(params['max_len']),
    ]


def run_final_phase(study: optuna.Study, gpu: int, profile: dict, final_epochs: int,
                    objective: str = 'mae', kinematics: str = 'P') -> None:
    """
    Phase 2: train a fresh model for final_epochs using the best hyperparameters,
    evaluating on the test split every 5 epochs.
    Phase 3: embed train+val+test with checkpoint_best and evaluate train+val→test.
    """
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print('\n[Final] No completed trials — skipping final evaluation.')
        return

    best   = study.best_trial
    params = best.params
    env    = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)}

    print(f'\n{"="*72}')
    print(f'Phase 2: Final model ({objective}) — {final_epochs} epochs  '
          f'(best trial #{best.number}, val score={best.value:.4f})')
    print(f'{"="*72}\n')

    if objective == 'tsjepa':
        cmd = build_final_cmd_tsjepa(params, gpu, profile, final_epochs)
    elif objective == 'posejepa':
        cmd = build_final_cmd_posejepa(params, gpu, profile, final_epochs, kinematics)
    else:
        cmd = build_final_cmd(params, gpu, profile, final_epochs)
    rc, stdout = _run_streaming(cmd, env)
    if rc != 0:
        print(f'\n[Final] Phase 2 training failed (exit code {rc})')
        return

    run_dir = None
    for line in stdout.splitlines():
        if line.startswith('RUN_DIR:'):
            run_dir = Path(line.split(':', 1)[1].strip())
            break
    if run_dir is None:
        print('\n[Final] Could not parse RUN_DIR from training output — skipping Phase 3.')
        return

    ckpt = run_dir / 'checkpoint_best.pt'
    if not ckpt.exists():
        print(f'\n[Final] checkpoint_best.pt not found at {ckpt} — skipping Phase 3.')
        return

    print(f'\n{"="*72}')
    print(f'Phase 3: Final test evaluation')
    print(f'Checkpoint: {ckpt}')
    print(f'{"="*72}\n')

    # Compute embedding stride to match the final model's training eval stride.
    # For posejepa: stride = max_len // stride_factor; others: use max_len // 2.
    _max_len      = params.get('max_len', 128)
    _stride_factor = params.get('stride_factor', 2)
    _p3_stride    = _max_len // _stride_factor

    embed_script  = str(_CHOROS_ROOT / 'pipeline' / 'embed_target_data.py')
    train_obj     = objective  # save before loop shadows the name
    p3_window_pool = (
        params.get('eval_window_pool', 'stat9') if train_obj == 'posejepa' else 'mean'
    )
    eval_targets = [
        (str(_DATA_ROOT / 'aligned' / 'target_FAB'),       'FAB', []),
        (str(_DATA_ROOT / 'aligned' / 'target_DEVCOM_s2'), 'D3',
         ['bot_dist_mean_s3', 'firing_accuracy_AOBJ_s3']),
    ]
    for data_dir, embed_obj, target_cols in eval_targets:
        print(f'\n[Phase 3] {Path(data_dir).name}  objective={embed_obj}')
        p3_cmd = [
            'conda', 'run', '-n', 'CHOROS',
            'python', embed_script,
            '--ckpt',         str(ckpt),
            '--data_dir',     data_dir,
            '--objective',    embed_obj,
            '--stride',       str(_p3_stride),
            '--window_pool',  p3_window_pool,
            '--session_pool', 'mean',
            '--split_keys',   'train,val,test',
            '--train_split',  'train+val',
            '--eval_split',   'test',
        ]
        if target_cols:
            p3_cmd += ['--target_cols'] + target_cols
        rc, _ = _run_streaming(p3_cmd, env)
        if rc != 0:
            print(f'\n[Phase 3] Failed for {embed_obj} (exit code {rc})')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Bayesian hyperparameter search for VR encoder')
    p.add_argument('--n_trials',       type=int, default=40,
                   help='Number of trials to run (default: 40)')
    p.add_argument('--gpu',            type=int, default=0,
                   help='CUDA device index for training subprocess (default: 0)')
    p.add_argument('--study_name',     type=str, default='baseline_v1',
                   help='Optuna study name (default: baseline_v1)')
    p.add_argument('--final_epochs',   type=int, default=50,
                   help='Epochs for Phase 2 final model training (default: 50)')
    p.add_argument('--skip_final',     action='store_true',
                   help='Skip Phase 2+3 after search (useful for parallel workers where '
                        'only one should run the final model)')
    p.add_argument('--worker_offset',  type=int, default=0,
                   help='Added to the TPE sampler seed (42 + offset) so parallel workers '
                        'explore different regions of the search space. Use a unique value '
                        'per machine. Default 0 preserves existing behaviour.')
    p.add_argument('--storage',        type=str, default=None,
                   help='SQLAlchemy storage URL for the Optuna study. Omit to use a local '
                        'SQLite file. For distributed search across machines use a shared '
                        'PostgreSQL URL, e.g. '
                        'postgresql://optuna:choroshps@app.arcadea.us/optuna_choros')
    p.add_argument('--objective',      type=str, default='mae',
                   choices=['mae', 'tsjepa', 'posejepa'],
                   help='Training objective: mae (default), tsjepa, or posejepa')
    p.add_argument('--batch_size',     type=int, default=None,
                   help='Override GPU-profile batch size for all trial subprocesses')
    p.add_argument('--num_workers',    type=int, default=None,
                   help='Override GPU-profile num_workers for all trial subprocesses')
    p.add_argument('--compile',        dest='compile', action='store_true', default=None,
                   help='Enable torch.compile in trial subprocesses (overrides GPU profile)')
    p.add_argument('--no_compile',     dest='compile', action='store_false',
                   help='Disable torch.compile in trial subprocesses (overrides GPU profile)')
    p.add_argument('--trial_epochs',   type=int, default=None,
                   help='Epochs per trial subprocess. Defaults: mae/tsjepa=10, posejepa=100.')
    p.add_argument('--eval_interval',  type=int, default=None,
                   help='embed_eval_interval per trial subprocess. Defaults: mae/tsjepa=10, posejepa=20.')
    p.add_argument('--kinematics',     type=str, default='P',
                   help='Kinematic channels passed to posejepa trials (default: P). '
                        'Examples: P, PVAJ. Has no effect for mae/tsjepa objectives.')
    p.add_argument('--score_mode',     type=str, default='stable_tail',
                   choices=['stable_tail', 'mean_tail', 'last'],
                   help='How per-eval aggregate probe scores become the Optuna trial value. '
                        'stable_tail rewards sustained recent performance; last preserves '
                        'the old final-checkpoint behavior. Default: stable_tail.')
    p.add_argument('--auc_weight',     type=float, default=0.5,
                   help='Weight for AUROC in each per-target probe score (default: 0.5).')
    p.add_argument('--mcc_weight',     type=float, default=0.5,
                   help='Weight for rescaled MCC in each per-target probe score '
                        '(default: 0.5).')
    p.add_argument('--weakest_target_weight', type=float, default=0.25,
                   help='Blend weight for the weakest per-target score after harmonic '
                        'mean aggregation. 0 disables; 0.25 is the default.')
    p.add_argument('--score_tail_k',   type=int, default=3,
                   help='Number of recent probe evals used by stable_tail/mean_tail '
                        '(default: 3).')
    p.add_argument('--volatility_penalty', type=float, default=0.5,
                   help='Penalty multiplier for std(last K scores) in stable_tail '
                        '(default: 0.5).')
    p.add_argument('--trend_weight',   type=float, default=0.1,
                   help='Weight for final-minus-first probe score in stable_tail '
                        '(default: 0.1).')
    p.add_argument('--trial_timeout',  type=int, default=1500,
                   help='Per-trial subprocess timeout in seconds (default: 1500). '
                        'Increase for slower GPUs running long-epoch objectives '
                        'e.g. --trial_timeout 7200 for posejepa on an RTX 4070.')
    p.add_argument('--no_early_stopping', action='store_true',
                   help='Disable Optuna MedianPruner early stopping. By default, trials '
                        'whose intermediate scores fall below the median of prior trials '
                        'are pruned (most impactful for posejepa with 5 eval checkpoints).')
    p.add_argument('--pruner_startup', type=int, default=20,
                   help='Number of completed trials before the pruner starts pruning '
                        '(default: 20). Passed as n_startup_trials to MedianPruner.')
    p.add_argument('--sampler_startup', type=int, default=40,
                   help='Number of random startup trials for TPE before model-based '
                        'sampling begins (default: 40).')
    p.add_argument('--n_ei_candidates', type=int, default=64,
                   help='Number of expected-improvement candidates sampled by TPE '
                        '(default: 64).')
    p.add_argument('--no_multivariate_tpe', action='store_true',
                   help='Disable multivariate/group TPE. Enabled by default to model '
                        'parameter interactions and reduce brittle one-parameter collapse.')
    return p.parse_args()


def main():
    args    = parse_args()
    profile = _profile_for_gpu(args.gpu)
    if args.batch_size is not None:
        profile['batch_size'] = args.batch_size
    if args.num_workers is not None:
        profile['num_workers'] = args.num_workers
    if args.compile is not None:
        profile['compile'] = args.compile
    print_gpu_profile(profile)

    pruner = None
    if not args.no_early_stopping:
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=args.pruner_startup,
            n_warmup_steps=1,
            interval_steps=1,
        )

    study = create_study(
        args.study_name,
        args.worker_offset,
        args.storage,
        pruner=pruner,
        sampler_startup=args.sampler_startup,
        n_ei_candidates=args.n_ei_candidates,
        multivariate_tpe=not args.no_multivariate_tpe,
    )

    existing = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    print(f'Study: {args.study_name}')
    print(f'Objective: {args.objective}')
    print(f'Probe score weights: auc={args.auc_weight}, mcc={args.mcc_weight}, '
          f'weakest_target={args.weakest_target_weight}')
    print(f'Score mode: {args.score_mode} '
          f'(tail_k={args.score_tail_k}, volatility_penalty={args.volatility_penalty}, '
          f'trend_weight={args.trend_weight})')
    print(f'TPE sampler: startup={args.sampler_startup}, '
          f'n_ei_candidates={args.n_ei_candidates}, '
          f'multivariate_group={not args.no_multivariate_tpe}')
    print(f'Early stopping: {"disabled" if args.no_early_stopping else f"MedianPruner (startup={args.pruner_startup}, warmup=1)"}')
    print(f'Existing completed trials: {existing}')
    print(f'GPU: {args.gpu}   Trials to run: {args.n_trials}')

    import functools
    objective_fn = functools.partial(run_trial, gpu=args.gpu, profile=profile,
                                     objective=args.objective,
                                     trial_timeout=args.trial_timeout,
                                     trial_epochs=args.trial_epochs,
                                     eval_interval=args.eval_interval,
                                     kinematics=args.kinematics,
                                     auc_weight=args.auc_weight,
                                     mcc_weight=args.mcc_weight,
                                     weakest_target_weight=args.weakest_target_weight,
                                     score_mode=args.score_mode,
                                     score_tail_k=args.score_tail_k,
                                     volatility_penalty=args.volatility_penalty,
                                     trend_weight=args.trend_weight)

    t_start = time.time()
    try:
        study.optimize(
            objective_fn,
            n_trials=args.n_trials,
            callbacks=[_progress_callback],
            catch=(Exception,),
        )
    except KeyboardInterrupt:
        print('\nInterrupted. Saving results...')

    total_s = time.time() - t_start
    h, m, s = int(total_s // 3600), int((total_s % 3600) // 60), int(total_s % 60)
    print(f'\nTotal runtime: {h:02d}:{m:02d}:{s:02d}')

    out_path = _CHOROS_ROOT / 'outputs' / 'hparam_search' / f'{args.study_name}_results.tsv'
    write_tsv_results(study, out_path)
    _print_best_trial(study)

    if not args.skip_final:
        run_final_phase(study, args.gpu, profile, args.final_epochs, args.objective, args.kinematics)


if __name__ == '__main__':
    main()
