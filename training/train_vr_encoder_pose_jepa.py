"""
Train the VR motion encoder with a Pose-JEPA self-supervised objective.

Pose-JEPA extends TS-JEPA with patch-level tokenisation, per-sample target
masks, future-prediction mode, context/target feature separation, collapse
diagnostics, and configurable latent loss and inference pooling.

Key differences vs. train_vr_encoder_tsjepa.py
-----------------------------------------------
  • --patch_size       Frames per patch token (default: 8 ≈ 267 ms at 30 Hz).
  • --target_mode      masked_span | future | mixed (default: mixed).
  • --future_min_gap   Min gap (patches) between context end and future target.
  • --future_horizon   Range [min, max] of target patches for future mode.
  • --latent_loss      smooth_l1 | cosine (default: smooth_l1).
  • --embed_pool       cls | mean | mean_std | last (default: mean).
  • --context_device_drop  Devices zeroed in context ONLY (cross-device task).
  • --context_group_mask   Kinematic groups zeroed in context ONLY.
  • --diag_interval    Log collapse diagnostics every N epochs (default: 1).

Checkpoint files
----------------
  checkpoint_latest.pt              — saved every epoch
  checkpoint_best.pt                — best pretraining val loss (or train loss)
  checkpoint_best_probe_val.pt      — best 3-eval smoothed aggregate val MCC from periodic eval
                                      (only updated when embed_eval_interval > 0)
  checkpoint_best_probe_<metric>_wp_<window_pool>_sp_<session_pool>.pt
                                    — best val MCC for each individual metric
                                      (e.g. firing, port_score, bot_dist)

Usage
-----
  conda run -n CHOROS python train_vr_encoder_pose_jepa.py \\
      --npy_dir /srv/CHOROS/data/kinematics/VR_npy_PVAJ \\
      --kinematics PVAJ --patch_size 8 --target_mode mixed \\
      --epochs 50 --embed_eval_interval 5 --eval_split_mode val
"""

import argparse
import os
import re
import random
import math
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

_CHOROS_ROOT = Path(__file__).parent.parent
_DATA_ROOT   = Path(os.environ.get('CHOROS_DATA_ROOT', '/srv/CHOROS/data'))
sys.path.insert(0, str(_CHOROS_ROOT / 'src'))
sys.path.insert(0, str(_CHOROS_ROOT / 'training'))
from gpu_profiles import get_gpu_profile, print_gpu_profile

from features import build_feature_cols
from masking import device_col_indices, group_col_indices, feat_col_indices
from vr_encoder_pose_jepa import (
    PoseJEPA, KINEMATICS, latent_diagnostics,
    MAX_LEN, PATCH_SIZE, EMBED_DIM, N_HEADS, N_LAYERS, FFN_DIM, DROPOUT,
    PRED_LAYERS, PRED_FFN_DIM,
    TARGET_RATIO, N_TARGET_BLOCKS, EMA_DECAY_START, EMA_DECAY_END,
    TARGET_MODE, FUTURE_MIN_GAP, FUTURE_HORIZON, EMBED_POOL, LATENT_LOSS,
)


def npy_feature_indices(feature_cols: list[str]) -> np.ndarray:
    """Column indices for selecting feature_cols from PVAJ-format .npy arrays."""
    pvaj_cols = build_feature_cols("PVAJ")
    col_to_idx = {c: i for i, c in enumerate(pvaj_cols)}
    return np.array([col_to_idx[c] for c in feature_cols], dtype=np.int64)


# ---------------------------------------------------------------------------
# Dataset  (identical to train_vr_encoder_tsjepa.py)
# ---------------------------------------------------------------------------

def _available_ram_bytes() -> int:
    try:
        with open('/proc/meminfo') as _f:
            for _line in _f:
                if _line.startswith('MemAvailable:'):
                    return int(_line.split()[1]) * 1024
    except OSError:
        pass
    return 0


class VRDataset(Dataset):
    def __init__(
        self,
        npy_dir:           str | Path,
        feature_cols:      list[str],
        max_len:           int   = MAX_LEN,
        feat_mean:         np.ndarray | None = None,
        feat_std:          np.ndarray | None = None,
        samples_per_epoch: int   = 0,
        sampling_alpha:    float = 1.0,
        exclude_datasets:  list[str] | None = None,
        file_allowlist:    list | None = None,
        verbose:           bool = True,
    ):
        npy_path = Path(npy_dir)
        all_npy  = sorted(npy_path.glob('*.npy'))
        if file_allowlist is not None:
            allowset = set(file_allowlist)
            all_npy  = [f for f in all_npy if f in allowset]
        self.files        = all_npy
        self._npy_col_idx = npy_feature_indices(feature_cols)

        _packed_path = npy_path.parent / f'{npy_path.name}_packed.npy'
        _index_path  = npy_path.parent / f'{npy_path.name}_index.npz'
        _use_packed  = False
        if _packed_path.exists() and _index_path.exists():
            _packed_size = _packed_path.stat().st_size
            _avail_ram   = _available_ram_bytes()
            if _avail_ram > _packed_size:
                _use_packed = True
                print(f'  Using packed mmap: {_packed_path.name}'
                      f'  ({_packed_size/1e9:.1f} GB fits in {_avail_ram/1e9:.1f} GB available RAM)', flush=True)
            else:
                print(f'  Packed file ({_packed_size/1e9:.1f} GB) exceeds available RAM'
                      f' ({_avail_ram/1e9:.1f} GB) — using per-file reads', flush=True)
        if _use_packed:
            _idx          = np.load(_index_path)
            _name_to_pos  = {n: i for i, n in enumerate(_idx['names'].tolist())}
            self._packed      = np.load(str(_packed_path), mmap_mode='r')
            self._packed_off  = np.array([_idx['offsets'][_name_to_pos[f.name]] for f in self.files], dtype=np.int64)
            self._file_n_rows = [int(_idx['n_rows'][_name_to_pos[f.name]]) for f in self.files]
            self._headers     = None
        else:
            self._packed     = None
            self._packed_off = None
            print(f'Reading npy headers for {len(self.files):,} files …', flush=True)
            self._headers     = [self._npy_header(f) for f in self.files]
            self._file_n_rows = [hdr[1] for hdr in self._headers]

        if exclude_datasets:
            _excl = set(exclude_datasets)
            keep  = [i for i, f in enumerate(self.files)
                     if f.stem.split('_')[0] not in _excl]
            self.files        = [self.files[i]        for i in keep]
            self._file_n_rows = [self._file_n_rows[i] for i in keep]
            if self._headers is not None:
                self._headers = [self._headers[i] for i in keep]
            if self._packed_off is not None:
                self._packed_off = self._packed_off[keep]
            print(f'Excluded datasets: {sorted(_excl)}  ({len(self.files):,} files remain)')

        self.feature_cols   = feature_cols
        self.n_features     = len(feature_cols)
        self.max_len        = max_len
        self.feat_mean      = feat_mean
        self.feat_std       = feat_std
        self.sampling_alpha = sampling_alpha

        groups: dict[str, list[int]] = {}
        for i, f in enumerate(self.files):
            groups.setdefault(f.stem.split('_')[0], []).append(i)
        ds_names = sorted(groups.keys())

        ds_total_wins: dict[str, int]       = {}
        ds_file_cum:   dict[str, list[int]] = {}
        for ds, idxs in groups.items():
            t, cum = 0, []
            for i in idxs:
                t += max(1, self._file_n_rows[i] - max_len + 1)
                cum.append(t)
            ds_total_wins[ds] = t
            ds_file_cum[ds]   = cum

        ds_level_cum: list[float] = []
        acc = 0.0
        for ds in ds_names:
            acc += ds_total_wins[ds] ** sampling_alpha
            ds_level_cum.append(acc)

        z = ds_level_cum[-1]
        self.sampling_shares = {
            ds: (ds_total_wins[ds] ** sampling_alpha) / z for ds in ds_names
        }
        self._ds_names      = ds_names
        self._ds_level_cum  = ds_level_cum
        self._ds_groups     = groups
        self._ds_file_cum   = ds_file_cum
        self._ds_total_wins = ds_total_wins

        total_wins = sum(ds_total_wins.values())
        self._n = samples_per_epoch if samples_per_epoch > 0 else total_wins
        if verbose:
            print(f'\nDataset temperature sampling  alpha={sampling_alpha:.2f}  '
                  f'({len(ds_names)} datasets  {total_wins:,} total windows'
                  f'  window_len={max_len})', flush=True)
            print(f"  {'Dataset':24s}  {'Windows':>12s}  {'Share':>7s}  {'Samples/ep':>11s}")
            print(f"  {'-'*62}")
            for ds in sorted(ds_names, key=lambda k: -self.sampling_shares[k]):
                share = self.sampling_shares[ds]
                print(f"  {ds:24s}  {ds_total_wins[ds]:>12,d}  "
                      f"{100*share:>6.2f}%  {share * self._n:>11,.0f}")
            print()
            print(f"  File-length distribution (frames) per sampled dataset:")
            print(f"  {'Dataset':24s}  {'Files':>6s}  {'Mean':>8s}  "
                  f"{'Q1':>8s}  {'Median':>8s}  {'Q3':>8s}")
            print(f"  {'-'*72}")
            _all_lens: list[int] = []
            for ds in sorted(ds_names, key=lambda k: -self.sampling_shares[k]):
                _lens = [self._file_n_rows[i] for i in groups[ds]]
                _all_lens.extend(_lens)
                _arr = np.array(_lens, dtype=np.float64)
                _q1, _med, _q3 = np.percentile(_arr, [25, 50, 75])
                print(f"  {ds:24s}  {len(_lens):>6,d}  {_arr.mean():>8,.1f}  "
                      f"{_q1:>8,.1f}  {_med:>8,.1f}  {_q3:>8,.1f}")
            _all_arr = np.array(_all_lens, dtype=np.float64)
            _aq1, _amed, _aq3 = np.percentile(_all_arr, [25, 50, 75])
            print(f"  {'ALL':24s}  {len(_all_lens):>6,d}  {_all_arr.mean():>8,.1f}  "
                  f"{_aq1:>8,.1f}  {_amed:>8,.1f}  {_aq3:>8,.1f}")
            print(flush=True)
        else:
            print(f'Val pretraining set: {len(self.files):,} files  '
                  f'{len(ds_names)} datasets  {total_wins:,} total windows'
                  f'  window_len={max_len}', flush=True)

    @staticmethod
    def _npy_header(path: Path) -> tuple[int, int, int, bool]:
        import struct, ast
        with open(path, 'rb') as f:
            f.read(6)
            major = struct.unpack('B', f.read(1))[0]
            f.read(1)
            hlen  = struct.unpack('<H' if major == 1 else '<I',
                                  f.read(2 if major == 1 else 4))[0]
            hdr   = ast.literal_eval(f.read(hlen).decode('latin1').strip().rstrip(','))
            offset = f.tell()
        n_rows, n_cols = hdr['shape']
        return offset, n_rows, n_cols, hdr.get('fortran_order', False)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int):
        ds       = random.choices(self._ds_names, cum_weights=self._ds_level_cum, k=1)[0]
        file_idx = random.choices(self._ds_groups[ds], cum_weights=self._ds_file_cum[ds], k=1)[0]

        if self._packed is not None:
            n_rows = self._file_n_rows[file_idx]
            base   = int(self._packed_off[file_idx])
            n_cols = self._packed.shape[1]
            if n_rows >= self.max_len:
                start  = random.randint(0, n_rows - self.max_len)
                x      = np.array(self._packed[base + start : base + start + self.max_len], dtype=np.float32)
                length = self.max_len
            else:
                x      = np.array(self._packed[base : base + n_rows], dtype=np.float32)
                length = n_rows
                x      = np.concatenate([x, np.zeros((self.max_len - n_rows, n_cols), np.float32)])
        else:
            offset, n_rows, n_cols, fortran_order = self._headers[file_idx]
            if fortran_order:
                mmap = np.load(self.files[file_idx], mmap_mode='r')
                if n_rows >= self.max_len:
                    start  = random.randint(0, n_rows - self.max_len)
                    x      = np.array(mmap[start:start + self.max_len], dtype=np.float32)
                    length = self.max_len
                else:
                    x      = np.array(mmap, dtype=np.float32)
                    length = n_rows
                    x      = np.concatenate([x, np.zeros((self.max_len - n_rows, n_cols), np.float32)])
                del mmap
            else:
                if n_rows >= self.max_len:
                    start    = random.randint(0, n_rows - self.max_len)
                    byte_off = offset + start * n_cols * 4
                else:
                    byte_off = offset
                with open(self.files[file_idx], 'rb') as f:
                    f.seek(byte_off)
                    x = np.fromfile(f, dtype=np.float32,
                                    count=min(n_rows, self.max_len) * n_cols).reshape(-1, n_cols)
                if n_rows >= self.max_len:
                    length = self.max_len
                else:
                    length = n_rows
                    x      = np.concatenate([x, np.zeros((self.max_len - n_rows, n_cols), np.float32)])

        x = x[:, self._npy_col_idx]

        if self.feat_mean is not None:
            x = (x - self.feat_mean) / (self.feat_std + 1e-8)
            x = np.clip(x, -10.0, 10.0)

        return torch.from_numpy(x), torch.tensor(length, dtype=torch.long)


# ---------------------------------------------------------------------------
# Normalisation stats
# ---------------------------------------------------------------------------

def compute_norm_stats(
    npy_dir:          Path,
    feature_cols:     list[str],
    n_sample:         int   = 1000,
    seed:             int   = 42,
    sampling_alpha:   float = 1.0,
    max_len:          int   = MAX_LEN,
    exclude_datasets: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    import random as _rnd
    _rnd.seed(seed)
    rng = np.random.default_rng(seed)

    all_files   = sorted(Path(npy_dir).glob('*.npy'))
    npy_col_idx = npy_feature_indices(feature_cols)
    print(f'  Reading npy headers for norm stats ({len(all_files):,} files) …', flush=True)
    file_n_rows = [VRDataset._npy_header(f)[1] for f in all_files]

    if exclude_datasets:
        _excl = set(exclude_datasets)
        keep        = [i for i, f in enumerate(all_files) if f.stem.split('_')[0] not in _excl]
        all_files   = [all_files[i]   for i in keep]
        file_n_rows = [file_n_rows[i] for i in keep]

    groups: dict[str, list[int]] = {}
    for i, f in enumerate(all_files):
        groups.setdefault(f.stem.split('_')[0], []).append(i)
    ds_names = sorted(groups.keys())

    ds_total_wins: dict[str, int]       = {}
    ds_file_cum:   dict[str, list[int]] = {}
    for ds, idxs in groups.items():
        t, cum = 0, []
        for i in idxs:
            t += max(1, file_n_rows[i] - max_len + 1)
            cum.append(t)
        ds_total_wins[ds] = t
        ds_file_cum[ds]   = cum

    ds_level_cum: list[float] = []
    acc = 0.0
    for ds in ds_names:
        acc += ds_total_wins[ds] ** sampling_alpha
        ds_level_cum.append(acc)

    rows = []
    for _ in range(n_sample):
        ds       = _rnd.choices(ds_names, cum_weights=ds_level_cum, k=1)[0]
        file_idx = _rnd.choices(groups[ds], cum_weights=ds_file_cum[ds], k=1)[0]
        f        = all_files[file_idx]
        n_rows   = file_n_rows[file_idx]
        start    = int(rng.integers(0, max(1, n_rows - max_len + 1)))
        mmap = np.load(f, mmap_mode='r')
        rows.append(np.array(mmap[start : start + max_len, npy_col_idx], dtype=np.float32))
        del mmap

    data = np.concatenate(rows, axis=0)
    mean = data.mean(axis=0).astype(np.float32)
    std  = data.std(axis=0).astype(np.float32)
    std[std < 1e-8] = 1.0
    return mean, std


# ---------------------------------------------------------------------------
# LR / EMA schedules
# ---------------------------------------------------------------------------

def cosine_schedule_with_warmup(
    optimizer:    torch.optim.Optimizer,
    total_steps:  int,
    warmup_steps: int,
    min_lr_ratio: float = 1e-3,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def ema_decay_schedule(
    step:        int,
    total_steps: int,
    start:       float = EMA_DECAY_START,
    end:         float = EMA_DECAY_END,
) -> float:
    return end - (end - start) * (math.cos(math.pi * step / max(1, total_steps)) + 1) / 2


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

class Tee:
    def __init__(self, file):
        self.file   = file
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def _short_gpu_name(gpu_name: str) -> str:
    """Extract compact GPU identifier, e.g. '3090', 'A100' from full device name."""
    m = re.search(r'\b(\d{4}[A-Z]?|[A-Z]\d{2,3}[A-Z]?)\b', gpu_name)
    if m:
        return m.group(1)
    return 'cpu' if 'cpu' in gpu_name.lower() else gpu_name.split()[-1]


def run_stem(args) -> str:
    ts      = datetime.now().strftime('%Y%m%d_%H%M%S')
    dataset = Path(args.npy_dir).name
    gpu     = _short_gpu_name(args._gpu_profile.get('gpu_name', 'cpu'))
    stem = (
        f"{ts}"
        f"_{gpu}"
        f"_{dataset}"
        f"_posejepa"
        f"_e{args.epochs}"
        f"_bs{args.batch_size}"
        f"_lr{args.lr}"
        f"_dim{args.embed_dim}"
        f"_l{args.n_layers}"
        f"_ml{args.max_len}"
        f"_ps{args.patch_size}"
        f"_tr{args.target_ratio}"
        f"_tm{args.target_mode}"
        f"_ll{args.latent_loss}"
        f"_pool{args.embed_pool}"
        f"_wu{args.warmup_epochs}"
        f"_minlr{args.min_lr}"
        f"_kin{args.kinematics.upper()}"
    )
    if args.context_device_drop:
        stem += "_cdev" + "".join(d[0] for d in sorted(args.context_device_drop))
    if args.context_group_mask:
        stem += "_cgrp" + "".join(sorted(g.upper() for g in args.context_group_mask))
    if getattr(args, 'context_group_mask_schedule', None):
        stem += "_cgrpsched"
    stem += f"_sa{args.sampling_alpha}"
    stem += f"_sf{args.stride_factor}"
    if getattr(args, 'eval_window_pool', None):
        stem += f"_ewp{args.eval_window_pool}"
    if getattr(args, 'eval_session_pool', None):
        stem += f"_esp{args.eval_session_pool}"
    return stem


def make_run_dir(base_dir: Path, args) -> Path:
    run_dir  = base_dir / run_stem(args)
    run_dir.mkdir(parents=True, exist_ok=True)

    log_file   = open(run_dir / 'train.log', 'w', buffering=1)
    sys.stdout = Tee(log_file)

    print_gpu_profile(args._gpu_profile)
    print(f"RUN_DIR: {run_dir}")
    print("=" * 72)
    print(f"VREncoder Pose-JEPA Training Run")
    print(f"Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Run dir : {run_dir}")
    print("-" * 72)
    print("Model")
    print(f"  embed_dim      : {args.embed_dim}")
    print(f"  n_layers       : {args.n_layers}")
    print(f"  n_heads        : {args.n_heads}")
    print(f"  ffn_dim        : {args.ffn_dim}")
    print(f"  dropout        : {args.dropout}")
    print(f"  pred_layers    : {args.pred_layers}")
    print(f"  pred_ffn_dim   : {args.pred_ffn_dim}")
    print(f"  patch_size     : {args.patch_size} frames")
    print(f"  embed_pool     : {args.embed_pool}")
    print("Training")
    print(f"  npy_dir        : {args.npy_dir}")
    print(f"  epochs         : {args.epochs}")
    print(f"  batch_size     : {args.batch_size}")
    print(f"  lr             : {args.lr}")
    print(f"  min_lr         : {args.min_lr}")
    print(f"  warmup_epochs  : {args.warmup_epochs}")
    print(f"  max_len        : {args.max_len}")
    print(f"  stride_factor  : {args.stride_factor}")
    print(f"  target_ratio   : {args.target_ratio}")
    print(f"  n_target_blocks: {args.n_target_blocks}")
    print(f"  target_mode    : {args.target_mode}")
    print(f"  future_min_gap : {args.future_min_gap}")
    print(f"  future_horizon : {args.future_horizon_min}–{args.future_horizon_max}")
    print(f"  latent_loss    : {args.latent_loss}")
    print(f"  ctx_dev_drop   : {args.context_device_drop or '(none)'}")
    print(f"  ctx_grp_mask   : {args.context_group_mask or '(none)'}")
    print(f"  ctx_grp_sched  : {args.context_group_mask_schedule or '(none)'}")
    print(f"  ema_start      : {args.ema_start}")
    print(f"  kinematics     : {args.kinematics.upper()}")
    print(f"  val_fraction   : {args.val_fraction}")
    print(f"  samples/ep     : {args.samples_per_epoch or '(= n_files)'}")
    print(f"  sampling_α     : {args.sampling_alpha}")
    print(f"  compile        : {args.compile}")
    print(f"  seed           : {args.seed}")
    print(f"  num_workers    : {args.num_workers}")
    print(f"  eval_interval  : {args.embed_eval_interval} epochs")
    print(f"  diag_interval  : {args.diag_interval} epochs")
    print("=" * 72)
    print()

    return run_dir


# ---------------------------------------------------------------------------
# Validation loss
# ---------------------------------------------------------------------------

@torch.no_grad()
def _compute_val_loss(model, val_loader, device, amp_dtype, args, ctx_mask_cols) -> float:
    model.eval()
    total, n = 0.0, 0
    ctx_dev  = _context_device_col_indices(args)
    for x, lengths in val_loader:
        x       = x.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=device.type == 'cuda'):
            loss, _ = model.jepa_loss(
                x, lengths,
                target_ratio=args.target_ratio,
                n_target_blocks=args.n_target_blocks,
                target_mode=args.target_mode,
                future_min_gap=args.future_min_gap,
                future_horizon=(args.future_horizon_min, args.future_horizon_max),
                feat_mask_cols=ctx_mask_cols,
                context_device_cols=ctx_dev,
                latent_loss=args.latent_loss,
            )
        total += loss.item()
        n += 1
    model.train()
    return total / n if n > 0 else float('inf')


def _context_device_col_indices(args) -> list[int] | None:
    """Compute device column indices to zero in context only (cross-device task).
    Group-mask columns are handled separately via feat_mask_cols."""
    if not args.context_device_drop:
        return None
    from masking import device_col_indices as _dci
    cols = _dci(args.kinematics, list(args.context_device_drop))
    return sorted(set(cols)) if cols else None


def _parse_group_mask_schedule(
    schedule: list[str], kinematics: str,
) -> list[tuple[float, list[int] | None]]:
    """Parse --context_group_mask_schedule entries into (probability, col_indices) pairs.

    Entry format: 'weight:GROUPS' where GROUPS is concatenated group letters (P/V/A/J).
    Empty GROUPS means no masking for that outcome.  Weights are normalised to sum to 1.

    Example: ['65:', '12.5:V', '12.5:AJ', '10:VAJ']
    """
    parsed = []
    for entry in schedule:
        if ':' not in entry:
            raise ValueError(
                f"Invalid schedule entry '{entry}'. Expected 'weight:GROUPS', "
                f"e.g. '65:' (no mask) or '12.5:VAJ'."
            )
        w_str, grp_str = entry.split(':', 1)
        weight = float(w_str)
        if weight < 0:
            raise ValueError(f"Negative weight in schedule entry '{entry}'.")
        if grp_str:
            groups = list(grp_str.upper())
            cols = sorted(set(group_col_indices(kinematics, groups)))
        else:
            cols = None
        parsed.append((weight, cols))

    total = sum(w for w, _ in parsed)
    if total <= 0:
        raise ValueError("--context_group_mask_schedule weights must sum to a positive number.")
    return [(w / total, cols) for w, cols in parsed]


# ---------------------------------------------------------------------------
# Periodic embed + probe (returns aggregate val MCC or None)
# ---------------------------------------------------------------------------

_PROBE_SUMMARY_RE = re.compile(
    r'\[Probe\] objective=(\w+)\s+target=(\S+)\s+split=(\w+)'
    r'\s+Balanced Acc\.\:\s*([\d.]+)'
    r'(?:\s+Acc\.\:\s*[\d.]+)?'
    r'\s+MCC\:\s*(-?[\d.]+)'
    r'\s+F1\(macro\)\:\s*([\d.]+)'
    r'\s+ROC-AUC\:\s*([\d.]+)'
)
_PROBE_TARGET_KEYS = frozenset([
    'portScore', 'bot_dist_mean_s3', 'firing_accuracy_AOBJ_s3',
])
_PROBE_TARGET_SHORT = {
    'portScore':               'port_score',
    'bot_dist_mean_s3':        'bot_dist',
    'firing_accuracy_AOBJ_s3': 'firing',
}


def _active_metrics(args) -> frozenset:
    """Return the set of probe metrics to run, respecting --eval_metrics."""
    m = getattr(args, 'eval_metrics', None)
    return frozenset(m) if m else _PROBE_TARGET_KEYS


def _parse_probe_results(
    text: str, target_keys: frozenset,
) -> tuple[float | None, dict[str, float], float | None, dict[str, float]]:
    """Return (mean_val_mcc, per_val_mccs, mean_train_mcc, per_train_mccs).

    mean_val/train_mcc is None if any requested target is missing.
    """
    val_mccs:   dict[str, float] = {}
    train_mccs: dict[str, float] = {}
    for line in text.splitlines():
        m = _PROBE_SUMMARY_RE.search(line)
        if not m:
            continue
        split  = m.group(3)
        target = m.group(2)
        if target not in target_keys:
            continue
        mcc = float(m.group(5))
        if split == 'val':
            val_mccs[target]   = mcc
        elif split == 'train':
            train_mccs[target] = mcc
    val_avg   = sum(val_mccs.values())   / len(val_mccs)   if target_keys <= val_mccs.keys()   else None
    train_avg = sum(train_mccs.values()) / len(train_mccs) if target_keys <= train_mccs.keys() else None
    return val_avg, val_mccs, train_avg, train_mccs


def _probe_ckpt_suffix(args) -> str:
    """Build filename suffix encoding window/session pool args, e.g. '_wp_mean_std_max_sp_mean'."""
    parts = []
    wp = getattr(args, 'eval_window_pool', None)
    sp = getattr(args, 'eval_session_pool', None)
    if wp:
        parts.append(f'wp_{wp}')
    if sp:
        parts.append(f'sp_{sp}')
    return ('_' + '_'.join(parts)) if parts else ''


def _periodic_eval(
    run_dir: Path,
    epoch:   int,
    args,
    use_best_ckpt: bool = False,
) -> tuple[float | None, dict[str, float], float | None, dict[str, float]]:
    """
    Embed FAB and DEVCOM_s2 targets then run linear probes.
    Returns (aggregate_val_mcc, per_val_mccs, aggregate_train_mcc, per_train_mccs).
    aggregate_val_mcc is None if any of the 3 required targets is missing.
    """
    ckpt_name = 'checkpoint_best.pt' if use_best_ckpt else 'checkpoint_latest.pt'
    ckpt = run_dir / ckpt_name
    if not ckpt.exists():
        ckpt = run_dir / 'checkpoint_latest.pt'
    if not ckpt.exists():
        print(f'[Eval] checkpoint not found — skipping epoch {epoch} eval')
        return None, {}, None, {}

    embed_script = Path(__file__).parent.parent / 'pipeline' / 'embed_target_data.py'

    eval_metrics = _active_metrics(args)

    eval_targets = []
    if args.fab_eval_dir and Path(args.fab_eval_dir).exists():
        cols = [c for c in ['portScore'] if c in eval_metrics]
        if cols:
            eval_targets.append((args.fab_eval_dir, 'FAB', cols))
    if args.devcom_eval_dir and Path(args.devcom_eval_dir).exists():
        cols = [c for c in ['bot_dist_mean_s3', 'firing_accuracy_AOBJ_s3'] if c in eval_metrics]
        if cols:
            eval_targets.append((args.devcom_eval_dir, 'D3', cols))
    if not eval_targets:
        print(f'[Eval] no valid eval dirs — skipping epoch {epoch} eval')
        return None, {}, None, {}

    print(f'\n{"="*72}')
    print(f'Periodic eval — epoch {epoch}  ckpt: {ckpt}')
    print(f'{"="*72}')

    eval_split_mode = getattr(args, 'eval_split_mode', None)
    if eval_split_mode == 'val':
        extra_embed = ['--split_keys', 'train,val']
        extra_probe = ['--train_split', 'train', '--eval_split', 'val']
    elif eval_split_mode == 'test':
        extra_embed = ['--split_keys', 'train,val,test']
        extra_probe = ['--train_split', 'train+val', '--eval_split', 'test']
    else:
        extra_embed = []
        extra_probe = []

    all_stdout = []
    for data_dir, objective, target_cols in eval_targets:
        print(f'\n[Eval] {Path(data_dir).name}  objective={objective}')
        cmd = [sys.executable, str(embed_script),
               '--ckpt',      str(ckpt),
               '--data_dir',  data_dir,
               '--stride',    str(args.max_len // args.stride_factor),
               '--objective', objective,
               '--target_cols'] + target_cols
        if getattr(args, 'eval_window_pool', None):
            cmd += ['--window_pool', args.eval_window_pool]
        if getattr(args, 'eval_session_pool', None):
            cmd += ['--session_pool', args.eval_session_pool]
        cmd += extra_embed + extra_probe + ['--classification_only']
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(f'[Eval ERROR]\n{result.stderr}')
        all_stdout.append(result.stdout)

    print(f'{"="*72}\n')
    sys.stdout.flush()

    return _parse_probe_results('\n'.join(all_stdout), eval_metrics)


# ---------------------------------------------------------------------------
# Diagnostic logging helper
# ---------------------------------------------------------------------------

def _log_diag(epoch: int, diag_acc: dict, n_diag: int) -> None:
    if n_diag == 0:
        return
    print(f'  [Collapse diag epoch {epoch}]')
    for branch in ('ctx', 'tgt', 'pred'):
        d = {k: v / n_diag for k, v in diag_acc[branch].items()}
        print(
            f'    {branch:4s}  std_mean={d["std_mean"]:.4f}  '
            f'std_min={d["std_min"]:.4f}  '
            f'collapse_frac={d["collapse_frac"]:.4f}  '
            f'norm_mean={d["norm_mean"]:.4f}'
        )
    print(f'    future_frac={diag_acc["future_frac"] / n_diag:.4f}')


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npy_dir = Path(args.npy_dir)

    if args.max_len % args.patch_size != 0:
        raise ValueError(
            f'--max_len {args.max_len} must be divisible by --patch_size {args.patch_size}'
        )

    feature_cols = build_feature_cols(args.kinematics)
    n_features   = len(feature_cols)

    # Context-only group masking (feat_mask_cols) and device masking (ctx_dev_cols)
    # are kept separate to avoid double-zeroing the same columns.
    _fgm_cols = group_col_indices(args.kinematics, list(args.context_group_mask)) \
                if args.context_group_mask else []
    feat_mask_cols = sorted(set(_fgm_cols)) or None

    cgm_schedule = (
        _parse_group_mask_schedule(args.context_group_mask_schedule, args.kinematics)
        if args.context_group_mask_schedule else None
    )
    if cgm_schedule and feat_mask_cols:
        raise ValueError(
            "--context_group_mask and --context_group_mask_schedule are mutually exclusive."
        )

    ctx_dev_cols = _context_device_col_indices(args)

    run_dir = make_run_dir(out_dir, args)
    print(f'Device: {device}')
    if device.type == 'cuda':
        print(f'GPU   : {torch.cuda.get_device_name(0)}')
    print(f'Features : {n_features} ({args.kinematics.upper()})')
    n_patches = args.max_len // args.patch_size
    print(f'Patches  : {n_patches} × {args.patch_size} frames = {args.max_len} frame window')

    # ------------------------------------------------------------------ stats
    _excl_tag = ('_excl_' + '_'.join(sorted(args.exclude_datasets))
                 if args.exclude_datasets else '')
    norm_n_sample = 1000
    stats_path = (out_dir /
                  f'norm_stats_{npy_dir.name}'
                  f'_{args.kinematics.upper()}'
                  f'_a{args.sampling_alpha}'
                  f'_ml{args.max_len}'
                  f'_seed{args.seed}'
                  f'_ns{norm_n_sample}'
                  f'{_excl_tag}.npz')
    if stats_path.exists():
        print(f'Loading cached norm stats from {stats_path}')
        npz       = np.load(stats_path)
        mean, std = npz['mean'], npz['std']
    else:
        print(f'Computing normalisation stats from {npy_dir} …')
        mean, std = compute_norm_stats(
            npy_dir, feature_cols, n_sample=norm_n_sample, seed=args.seed,
            sampling_alpha=args.sampling_alpha, max_len=args.max_len,
            exclude_datasets=args.exclude_datasets,
        )
        np.savez(stats_path, mean=mean, std=std,
                 max_len=args.max_len, seed=args.seed, n_sample=norm_n_sample,
                 sampling_alpha=args.sampling_alpha, kinematics=args.kinematics.upper())
        print(f'  saved to {stats_path}')

    # ----------------------------------------------------------------- dataset
    val_loader       = None
    _train_allowlist = None
    _N_VAL_BATCHES   = 64

    if args.val_fraction > 0:
        all_npy = sorted(Path(npy_dir).glob('*.npy'))
        _ds_groups: dict[str, list[Path]] = {}
        for f in all_npy:
            _ds_groups.setdefault(f.stem.split('_')[0], []).append(f)
        _train_files, _val_files = [], []
        for ds in sorted(_ds_groups):
            files = _ds_groups[ds]
            n_val = max(1, round(len(files) * args.val_fraction))
            _val_files.extend(files[-n_val:])
            _train_files.extend(files[:-n_val])
        _train_allowlist = set(_train_files)
        print(f'Train/val split: {len(_train_files):,} train / {len(_val_files):,} val files '
              f'({args.val_fraction:.0%} per dataset)', flush=True)
        _val_ds = VRDataset(
            npy_dir, feature_cols,
            max_len=args.max_len, feat_mean=mean, feat_std=std,
            samples_per_epoch=_N_VAL_BATCHES * args.batch_size,
            sampling_alpha=args.sampling_alpha,
            file_allowlist=_val_files,
            verbose=False,
        )
        val_loader = DataLoader(
            _val_ds, batch_size=args.batch_size,
            shuffle=True, num_workers=2,
            pin_memory=device.type == 'cuda',
            drop_last=True,
        )

    dataset = VRDataset(
        npy_dir, feature_cols,
        max_len=args.max_len, feat_mean=mean, feat_std=std,
        samples_per_epoch=args.samples_per_epoch,
        sampling_alpha=args.sampling_alpha,
        exclude_datasets=args.exclude_datasets,
        file_allowlist=_train_allowlist,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda',
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    print(f'Dataset: {len(dataset):,} sequences | {len(loader):,} batches/epoch')

    # ------------------------------------------------------------------- model
    model = PoseJEPA(
        n_features=n_features,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        max_len=args.max_len,
        pred_layers=args.pred_layers,
        pred_ffn_dim=args.pred_ffn_dim,
        embed_pool=args.embed_pool,
    ).to(device)

    trainable_params   = list(model.context_encoder.parameters()) + \
                         list(model.predictor.parameters())
    n_params_total     = sum(p.numel() for p in model.parameters())
    n_params_trainable = sum(p.numel() for p in trainable_params)
    print(f'Model  : {n_params_total:,} total params  '
          f'({n_params_trainable:,} trainable  '
          f'{n_params_total - n_params_trainable:,} EMA target encoder)')

    pose_jepa_model = model
    if args.compile and device.type == 'cuda':
        print('Compiling model with torch.compile ...')
        model = torch.compile(model)

    _PRECISION_MAP = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}
    amp_dtype = _PRECISION_MAP[args.precision]
    if amp_dtype == torch.bfloat16 and device.type == 'cuda' and not torch.cuda.is_bf16_supported():
        print('WARNING: bf16 not supported on this GPU; falling back to fp16')
        amp_dtype = torch.float16
    use_scaler = (device.type == 'cuda') and (amp_dtype == torch.float16)
    scaler     = torch.amp.GradScaler('cuda', enabled=use_scaler)
    print(f'AMP dtype: {amp_dtype}')

    optimizer    = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    total_steps  = len(loader) * args.epochs
    warmup_steps = len(loader) * args.warmup_epochs
    min_lr_ratio = args.min_lr / args.lr
    scheduler    = cosine_schedule_with_warmup(
        optimizer, total_steps, warmup_steps, min_lr_ratio=min_lr_ratio
    )

    # --------------------------------------------------------------- resume
    best_jepa_loss               = float('inf')
    best_jepa_epoch              = 0
    best_smoothed_probe_mcc      = float('-inf')
    best_smoothed_probe_mcc_epoch = 0
    probe_mcc_history: list[float] = []
    best_probe_metric_mccs: dict[str, float] = {k: float('-inf') for k in _active_metrics(args)}
    global_step          = 0
    start_epoch          = 1
    if args.resume:
        print(f'Resuming from: {args.resume}')
        ckpt_r = torch.load(args.resume, map_location=device, weights_only=False)
        pose_jepa_model.load_state_dict(ckpt_r['model_state'])
        optimizer.load_state_dict(ckpt_r['optimizer_state'])
        scheduler.load_state_dict(ckpt_r['scheduler_state'])
        scaler.load_state_dict(ckpt_r['scaler_state'])
        best_jepa_loss                = ckpt_r.get('best_loss', float('inf'))
        best_jepa_epoch               = ckpt_r.get('best_jepa_epoch', 0)
        best_smoothed_probe_mcc       = ckpt_r.get('best_smoothed_probe_mcc', float('-inf'))
        best_smoothed_probe_mcc_epoch = ckpt_r.get('best_smoothed_probe_mcc_epoch', 0)
        probe_mcc_history             = ckpt_r.get('probe_mcc_history', [])
        best_probe_metric_mccs        = ckpt_r.get(
            'best_probe_metric_mccs', {k: float('-inf') for k in _active_metrics(args)}
        )
        global_step    = ckpt_r.get('global_step', 0)
        start_epoch    = ckpt_r['epoch'] + 1
        print(f'  -> epoch {ckpt_r["epoch"]} restored; resuming at epoch {start_epoch}')

    # ---------------------------------------------------------------- training
    future_horizon = (args.future_horizon_min, args.future_horizon_max)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches  = 0
        t0         = time.perf_counter()

        # Diagnostic accumulators (reset each epoch)
        diag_acc = {
            branch: {'std_mean': 0.0, 'std_min': 0.0,
                     'collapse_frac': 0.0, 'norm_mean': 0.0}
            for branch in ('ctx', 'tgt', 'pred')
        }
        diag_acc['future_frac'] = 0.0
        n_diag = 0

        for x, lengths in loader:
            x       = x.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)

            batch_feat_mask = (
                random.choices(
                    [cols for _, cols in cgm_schedule],
                    weights=[w for w, _ in cgm_schedule],
                )[0]
                if cgm_schedule else feat_mask_cols
            )

            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=device.type == 'cuda'):
                loss, diag = model.jepa_loss(
                    x, lengths,
                    target_ratio=args.target_ratio,
                    n_target_blocks=args.n_target_blocks,
                    target_mode=args.target_mode,
                    future_min_gap=args.future_min_gap,
                    future_horizon=future_horizon,
                    feat_mask_cols=batch_feat_mask,
                    context_device_cols=ctx_dev_cols,
                    latent_loss=args.latent_loss,
                )

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(trainable_params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            ema_decay = ema_decay_schedule(global_step, total_steps, start=args.ema_start)
            pose_jepa_model.update_target_encoder(ema_decay)

            total_loss  += loss.item()
            n_batches   += 1
            global_step += 1

            # Accumulate diagnostics
            if diag:
                for branch in ('ctx', 'tgt', 'pred'):
                    if branch in diag:
                        for k, v in diag[branch].items():
                            diag_acc[branch][k] += v
                diag_acc['future_frac'] += diag.get('future_frac', 0.0)
                n_diag += 1

        avg_loss   = total_loss / n_batches
        lr_now     = scheduler.get_last_lr()[0]
        ema_now    = ema_decay_schedule(global_step, total_steps, start=args.ema_start)
        epoch_secs = time.perf_counter() - t0

        if val_loader is not None:
            val_loss = _compute_val_loss(
                model, val_loader, device, amp_dtype, args,
                None if cgm_schedule else feat_mask_cols,
            )
            print(f'Epoch {epoch:3d}/{args.epochs}  '
                  f'train={avg_loss:.6f}  val={val_loss:.6f}  '
                  f'lr={lr_now:.2e}  ema={ema_now:.5f}  '
                  f'time={epoch_secs:.1f}s')
            checkpoint_score = val_loss
        else:
            print(f'Epoch {epoch:3d}/{args.epochs}  '
                  f'loss={avg_loss:.6f}  '
                  f'lr={lr_now:.2e}  ema={ema_now:.5f}  '
                  f'time={epoch_secs:.1f}s')
            checkpoint_score = avg_loss

        # Collapse diagnostics
        if args.diag_interval > 0 and epoch % args.diag_interval == 0:
            _log_diag(epoch, diag_acc, n_diag)

        # Checkpoint payload
        ckpt = {
            'epoch':                   epoch,
            'best_loss':               best_jepa_loss,
            'best_jepa_epoch':         best_jepa_epoch,
            'best_smoothed_probe_mcc':       best_smoothed_probe_mcc,
            'best_smoothed_probe_mcc_epoch': best_smoothed_probe_mcc_epoch,
            'probe_mcc_history':             probe_mcc_history,
            'best_probe_metric_mccs':        best_probe_metric_mccs,
            'global_step':             global_step,
            'model_state':     pose_jepa_model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'scaler_state':    scaler.state_dict(),
            'norm_mean':       mean,
            'norm_std':        std,
            'args': {
                'kinematics':          args.kinematics.upper(),
                'n_features':          n_features,
                'patch_size':          args.patch_size,
                'embed_dim':           args.embed_dim,
                'n_heads':             args.n_heads,
                'n_layers':            args.n_layers,
                'ffn_dim':             args.ffn_dim,
                'dropout':             args.dropout,
                'max_len':             args.max_len,
                'pred_layers':         args.pred_layers,
                'pred_ffn_dim':        args.pred_ffn_dim,
                'embed_pool':          args.embed_pool,
                'embedding_out_dim':   args.embed_dim * 2 if args.embed_pool == 'mean_std' else args.embed_dim,
                'target_ratio':        args.target_ratio,
                'n_target_blocks':     args.n_target_blocks,
                'target_mode':         args.target_mode,
                'future_min_gap':      args.future_min_gap,
                'future_horizon':      future_horizon,
                'latent_loss':         args.latent_loss,
                'context_device_drop':          args.context_device_drop,
                'context_group_mask':           args.context_group_mask,
                'context_group_mask_schedule':  args.context_group_mask_schedule,
                'feat_mask_cols':               feat_mask_cols,
                'cgm_schedule':                 cgm_schedule,
                'ctx_dev_cols':        ctx_dev_cols,
                'sampling_alpha':      args.sampling_alpha,
                'samples_per_epoch':   dataset._n,
                'dataset_sampling_shares': dataset.sampling_shares,
            },
        }
        torch.save(ckpt, run_dir / 'checkpoint_latest.pt')

        if checkpoint_score < best_jepa_loss:
            best_jepa_loss  = checkpoint_score
            best_jepa_epoch = epoch
            torch.save(ckpt, run_dir / 'checkpoint_best.pt')
            metric_label = 'val' if val_loader is not None else 'loss'
            print(f'  -> new best checkpoint saved ({metric_label}={best_jepa_loss:.6f})')

        # Periodic probe eval
        if args.embed_eval_interval > 0 and epoch % args.embed_eval_interval == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            probe_mcc, per_metric_mccs, train_probe_mcc, _ = _periodic_eval(
                run_dir, epoch, args,
                use_best_ckpt=getattr(args, 'eval_use_best_ckpt', False),
            )
            pool_suffix = _probe_ckpt_suffix(args)
            if probe_mcc is not None:
                probe_mcc_history.append(probe_mcc)
                probe_mcc_history = probe_mcc_history[-3:]
                smoothed_mcc = float(np.mean(probe_mcc_history))
                train_mcc_str = f'  probe_train_mcc={train_probe_mcc:.4f}' if train_probe_mcc is not None else ''
                print(f'  probe_val_mcc={probe_mcc:.4f}{train_mcc_str}  smoothed({len(probe_mcc_history)})={smoothed_mcc:.4f}')
                if smoothed_mcc > best_smoothed_probe_mcc:
                    best_smoothed_probe_mcc       = smoothed_mcc
                    best_smoothed_probe_mcc_epoch = epoch
                    ckpt['best_smoothed_probe_mcc']       = best_smoothed_probe_mcc
                    ckpt['best_smoothed_probe_mcc_epoch'] = best_smoothed_probe_mcc_epoch
                    ckpt['probe_mcc_history']             = probe_mcc_history
                    torch.save(ckpt, run_dir / f'checkpoint_best_probe_val{pool_suffix}.pt')
                    print(f'  -> new best probe checkpoint saved '
                          f'(smoothed_mcc={best_smoothed_probe_mcc:.4f})')
            for target_key, mcc in per_metric_mccs.items():
                if mcc > best_probe_metric_mccs.get(target_key, float('-inf')):
                    best_probe_metric_mccs[target_key] = mcc
                    ckpt['best_probe_metric_mccs'] = best_probe_metric_mccs
                    short = _PROBE_TARGET_SHORT.get(target_key, target_key)
                    fname = f'checkpoint_best_probe_{short}{pool_suffix}.pt'
                    torch.save(ckpt, run_dir / fname)
                    print(f'  -> new best {short} checkpoint saved '
                          f'(mcc={mcc:.4f}) -> {fname}')

    _metric_label = 'val' if val_loader is not None else 'loss'
    print(f'\nTraining complete. Best {_metric_label}: {best_jepa_loss:.6f} (epoch {best_jepa_epoch})')
    if best_smoothed_probe_mcc > float('-inf'):
        print(f'Best smoothed probe val MCC: {best_smoothed_probe_mcc:.4f} (epoch {best_smoothed_probe_mcc_epoch})')
    print(f'Run dir  : {run_dir}')
    print(f'Finished : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    _profile = get_gpu_profile()
    p = argparse.ArgumentParser(description='Train VR motion encoder (Pose-JEPA)')

    # Data / output
    p.add_argument('--npy_dir',           type=str, required=True,
                   help='Directory of .npy files from preprocess_to_npy.py')
    p.add_argument('--out_dir',           default=str(_CHOROS_ROOT / 'outputs' / 'checkpoints'))
    p.add_argument('--samples_per_epoch', type=int,   default=0)
    p.add_argument('--sampling_alpha',    type=float, default=0.5)
    p.add_argument('--exclude_datasets',  nargs='*',  metavar='DATASET', default=None)
    p.add_argument('--compile',           action='store_true', default=_profile['compile'])
    p.add_argument('--no_compile',        dest='compile', action='store_false')
    p.add_argument('--precision',         type=str,   default=_profile['precision'],
                   choices=['bf16', 'fp16', 'fp32'])
    p.add_argument('--resume',            type=str,   default=None)

    # Training
    p.add_argument('--epochs',        type=int,   default=50)
    p.add_argument('--batch_size',    type=int,   default=_profile['batch_size'])
    p.add_argument('--lr',            type=float, default=1e-3)
    p.add_argument('--min_lr',        type=float, default=1e-6)
    p.add_argument('--warmup_epochs', type=int,   default=2)
    p.add_argument('--max_len',       type=int,   default=MAX_LEN)
    p.add_argument('--stride_factor', type=int,   default=2)
    p.add_argument('--num_workers',   type=int,   default=_profile['num_workers'])
    p.add_argument('--seed',          type=int,   default=42)
    p.add_argument('--kinematics',    type=str,   default=KINEMATICS)
    p.add_argument('--val_fraction',  type=float, default=0.1,
                   help='Fraction of files per dataset held out for val checkpoint selection '
                        '(0 disables, default: 0.1)')

    # Pose-JEPA model
    p.add_argument('--patch_size',    type=int,   default=PATCH_SIZE,
                   help='Frames per patch token (default: 8 ≈ 267 ms at 30 Hz)')
    p.add_argument('--embed_dim',     type=int,   default=EMBED_DIM)
    p.add_argument('--n_heads',       type=int,   default=N_HEADS)
    p.add_argument('--n_layers',      type=int,   default=N_LAYERS)
    p.add_argument('--ffn_dim',       type=int,   default=FFN_DIM)
    p.add_argument('--dropout',       type=float, default=DROPOUT)
    p.add_argument('--pred_layers',   type=int,   default=PRED_LAYERS)
    p.add_argument('--pred_ffn_dim',  type=int,   default=PRED_FFN_DIM)
    p.add_argument('--embed_pool',    type=str,   default=EMBED_POOL,
                   choices=['cls', 'mean', 'mean_std', 'last'],
                   help='Inference pooling strategy (default: mean)')

    # Pose-JEPA objective
    p.add_argument('--target_ratio',       type=float, default=TARGET_RATIO)
    p.add_argument('--n_target_blocks',    type=int,   default=N_TARGET_BLOCKS)
    p.add_argument('--target_mode',        type=str,   default=TARGET_MODE,
                   choices=['masked_span', 'future', 'mixed'],
                   help='Target sampling mode (default: mixed)')
    p.add_argument('--future_min_gap',     type=int,   default=FUTURE_MIN_GAP,
                   help='Minimum patch gap between context end and future target (default: 4)')
    p.add_argument('--future_horizon_min', type=int,   default=FUTURE_HORIZON[0],
                   help='Minimum future target window in patches (default: 2)')
    p.add_argument('--future_horizon_max', type=int,   default=FUTURE_HORIZON[1],
                   help='Maximum future target window in patches (default: 8)')
    p.add_argument('--latent_loss',        type=str,   default=LATENT_LOSS,
                   choices=['smooth_l1', 'cosine'],
                   help='Latent-space loss (default: smooth_l1)')
    p.add_argument('--ema_start',          type=float, default=EMA_DECAY_START)

    # Context-only masking (cross-device / cross-kinematic task)
    p.add_argument('--context_device_drop', nargs='*', metavar='DEVICE',
                   help='Zero device columns in context ONLY (cross-device prediction). '
                        'Valid: head left right.')
    p.add_argument('--context_group_mask', nargs='*', metavar='GROUP',
                   help='Zero kinematic group columns in context ONLY. Valid: P V A J.')
    p.add_argument('--context_group_mask_schedule', nargs='*', metavar='ENTRY',
                   help='Stochastic context group masking: sample a mask per batch. '
                        'Each entry: "weight:GROUPS" where GROUPS are concatenated letters '
                        'P/V/A/J (empty = no mask). Weights are normalised. '
                        'Mutually exclusive with --context_group_mask. '
                        'Example: "65:" "12.5:V" "12.5:AJ" "10:VAJ"')

    # EMA
    p.add_argument('--diag_interval', type=int, default=1,
                   help='Log collapse diagnostics every N epochs (0 = disabled, default: 1)')

    # Periodic eval
    p.add_argument('--embed_eval_interval', type=int, default=5)
    p.add_argument('--fab_eval_dir',    type=str,
                   default=str(_DATA_ROOT / 'aligned' / 'target_FAB'))
    p.add_argument('--devcom_eval_dir', type=str,
                   default=str(_DATA_ROOT / 'aligned' / 'target_DEVCOM_s2'))
    p.add_argument('--eval_window_pool',  type=str, default=None,
                   choices=['mean', 'mean_std', 'mean_std_max', 'layer_avg', 'cls', 'mean_all', 'last', 'stat9'])
    p.add_argument('--eval_session_pool', type=str, default=None,
                   choices=['mean', 'stat4'])
    p.add_argument('--eval_split_mode',   type=str, default=None,
                   choices=['val', 'test'])
    p.add_argument('--eval_use_best_ckpt', action='store_true', default=False)
    p.add_argument('--eval_metrics', nargs='+', default=None, metavar='METRIC',
                   choices=['portScore', 'bot_dist_mean_s3', 'firing_accuracy_AOBJ_s3'],
                   help='subset of probe metrics to embed and evaluate '
                        '(default: all three). '
                        'Choices: portScore bot_dist_mean_s3 firing_accuracy_AOBJ_s3')

    args = p.parse_args()
    args._gpu_profile = _profile
    return args


if __name__ == '__main__':
    train(parse_args())
