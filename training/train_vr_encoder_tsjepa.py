"""
Train the VR motion encoder with a TS-JEPA self-supervised objective.

TS-JEPA (Time-Series Joint-Embedding Predictive Architecture) predicts latent
representations of *target* time segments from *context* segments, entirely in
embedding space — unlike MAE there is no pixel-space reconstruction.

Architecture recap
------------------
  context_encoder  — Transformer (student).  Sees only context positions.
  target_encoder   — EMA copy of context encoder.  Sees only target positions,
                     produces ground-truth latents (no gradient path).
  predictor        — Small Transformer.  Maps context latents → predicted target
                     latents.  Stop-gradient on targets prevents collapse.

Key differences vs. train_vr_encoder.py (MAE)
----------------------------------------------
  • Loss is Smooth-L1 in latent space, not MSE in input space.
  • After every optimizer step the target encoder is updated via EMA:
      θ_t ← τ·θ_t + (1−τ)·θ_s
    with τ following a cosine schedule from EMA_DECAY_START → 1.0.
  • --mask_ratio replaced by --target_ratio / --n_target_blocks.

Usage
-----
  conda run -n CHOROS python train_vr_encoder_tsjepa.py [options]

Key options
-----------
  --npy_dir          Path to pre-built .npy files from preprocess_to_npy.py
  --out_dir          Checkpoint / log output  (default: outputs/checkpoints)
  --epochs           Training epochs          (default: 50)
  --batch_size       Batch size               (default: 256)
  --lr               Peak learning rate       (default: 1e-3)
  --max_len          Window length            (default: 128)
  --target_ratio     Fraction of valid timesteps used as prediction targets
                                               (default: 0.25)
  --n_target_blocks  Number of contiguous target blocks per sample (default: 2)
  --ema_start        Initial EMA decay for target encoder (default: 0.996)
  --embed_eval_interval  Embed+probe FAB/DEVCOM_s2 every N epochs (default: 5)
  --num_workers      DataLoader workers       (default: 4)
  --seed             Random seed              (default: 42)
"""

import argparse
import os
import random
import math
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

_CHOROS_ROOT = Path(__file__).parent.parent
_DATA_ROOT   = Path(os.environ.get('CHOROS_DATA_ROOT', '/srv/CHOROS/data'))
sys.path.insert(0, str(_CHOROS_ROOT / 'src'))
sys.path.insert(0, str(_CHOROS_ROOT / 'training'))
from gpu_profiles import get_gpu_profile, print_gpu_profile

from features import build_feature_cols
from masking import feat_col_indices
from vr_encoder_tsjepa import (
    TSJEPA, KINEMATICS,
    MAX_LEN, EMBED_DIM, N_HEADS, N_LAYERS, FFN_DIM, DROPOUT,
    PRED_LAYERS, PRED_FFN_DIM,
    TARGET_RATIO, N_TARGET_BLOCKS, EMA_DECAY_START, EMA_DECAY_END,
)


def npy_feature_indices(feature_cols: list[str]) -> np.ndarray:
    """Column indices for selecting feature_cols from PVAJ-format .npy arrays."""
    pvaj_cols = build_feature_cols("PVAJ")
    col_to_idx = {c: i for i, c in enumerate(pvaj_cols)}
    return np.array([col_to_idx[c] for c in feature_cols], dtype=np.int64)


# ---------------------------------------------------------------------------
# Dataset
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
    """
    Loads ego-centric VR motion sequences for TS-JEPA pre-training.

    Fast path: reads .npy files from npy_dir using direct byte seeks —
    no persistent file handles, no mmap exhaustion on large datasets.

    Sampling uses a two-level temperature scheme:
      1. Choose dataset d with probability ∝ total_windows(d)^sampling_alpha.
      2. Choose file within d weighted by its number of drawable windows.
      3. Choose a random crop start within that file.
    sampling_alpha=1.0 → global window-proportional (each crop equally likely).
    sampling_alpha=0.5 → sqrt-balanced (smaller datasets boosted, default).
    sampling_alpha=0.0 → equal dataset probability regardless of size.
    """

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
        all_npy = sorted(npy_path.glob('*.npy'))
        if file_allowlist is not None:
            allowset = set(file_allowlist)
            all_npy = [f for f in all_npy if f in allowset]
        self.files   = all_npy
        self.use_npy = True
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
            keep = [i for i, f in enumerate(self.files) if f.stem.split('_')[0] not in _excl]
            self.files        = [self.files[i] for i in keep]
            self._file_n_rows = [self._file_n_rows[i] for i in keep]
            if self._headers is not None:
                self._headers = [self._headers[i] for i in keep]
            if self._packed_off is not None:
                self._packed_off = self._packed_off[keep]
            print(f'Excluded datasets: {sorted(_excl)}  ({len(self.files):,} files remain)', flush=True)

        self.feature_cols   = feature_cols
        self.n_features     = len(feature_cols)
        self.max_len        = max_len
        self.feat_mean      = feat_mean
        self.feat_std       = feat_std
        self.sampling_alpha = sampling_alpha

        # Build temperature-weighted two-level sampler.
        groups: dict[str, list[int]] = {}
        for i, f in enumerate(self.files):
            groups.setdefault(f.stem.split('_')[0], []).append(i)
        ds_names = sorted(groups.keys())

        ds_total_wins: dict[str, int]        = {}
        ds_file_cum:   dict[str, list[int]]  = {}
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
        self.sampling_shares: dict[str, float] = {
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
            print(
                f'\nDataset temperature sampling  alpha={sampling_alpha:.2f}  '
                f'({len(ds_names)} datasets  {total_wins:,} total windows)',
                flush=True,
            )
            print(f"  {'Dataset':24s}  {'Windows':>12s}  {'Share':>7s}  {'Samples/ep':>11s}")
            print(f"  {'-'*62}")
            for ds in sorted(ds_names, key=lambda k: -self.sampling_shares[k]):
                share = self.sampling_shares[ds]
                print(
                    f"  {ds:24s}  {ds_total_wins[ds]:>12,d}  "
                    f"{100*share:>6.2f}%  {share * self._n:>11,.0f}"
                )
            print(flush=True)
        else:
            print(
                f'Val pretraining set: {len(self.files):,} files  '
                f'{len(ds_names)} datasets  {total_wins:,} total windows',
                flush=True,
            )

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
        fortran_order  = hdr.get('fortran_order', False)
        return offset, n_rows, n_cols, fortran_order

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
                    pad    = np.zeros((self.max_len - n_rows, n_cols), dtype=np.float32)
                    x      = np.concatenate([x, pad], axis=0)
                del mmap
            else:
                if n_rows >= self.max_len:
                    start    = random.randint(0, n_rows - self.max_len)
                    byte_off = offset + start * n_cols * 4
                else:
                    byte_off = offset
                with open(self.files[file_idx], 'rb') as f:
                    f.seek(byte_off)
                    count = min(n_rows, self.max_len) * n_cols
                    x = np.fromfile(f, dtype=np.float32, count=count).reshape(-1, n_cols)
                if n_rows >= self.max_len:
                    length = self.max_len
                else:
                    length = n_rows
                    pad = np.zeros((self.max_len - n_rows, n_cols), dtype=np.float32)
                    x   = np.concatenate([x, pad], axis=0)

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

    all_files = sorted(Path(npy_dir).glob('*.npy'))
    npy_col_idx = npy_feature_indices(feature_cols)
    print(f'  Reading npy headers for norm stats ({len(all_files):,} files) …', flush=True)
    file_n_rows = [VRDataset._npy_header(f)[1] for f in all_files]

    if exclude_datasets:
        _excl = set(exclude_datasets)
        keep = [i for i, f in enumerate(all_files) if f.stem.split('_')[0] not in _excl]
        all_files   = [all_files[i] for i in keep]
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
# LR schedule
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
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# EMA decay schedule
# ---------------------------------------------------------------------------

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


def run_stem(args) -> str:
    ts      = datetime.now().strftime('%Y%m%d_%H%M%S')
    dataset = Path(args.npy_dir).name
    stem = (
        f"{ts}"
        f"_{dataset}"
        f"_tsjepa"
        f"_e{args.epochs}"
        f"_bs{args.batch_size}"
        f"_lr{args.lr}"
        f"_dim{args.embed_dim}"
        f"_l{args.n_layers}"
        f"_ml{args.max_len}"
        f"_tr{args.target_ratio}"
        f"_nb{args.n_target_blocks}"
        f"_mt{args.mask_type}"
        f"_wu{args.warmup_epochs}"
        f"_minlr{args.min_lr}"
        f"_kin{args.kinematics.upper()}"
    )
    if args.device_mask:
        stem += "_dev" + "".join(d[0] for d in sorted(args.device_mask))
    if args.feature_group_mask:
        stem += "_grp" + "".join(sorted(g.upper() for g in args.feature_group_mask))
    stem += f"_sa{args.sampling_alpha}"
    return stem


def make_run_dir(base_dir: Path, args) -> Path:
    run_dir  = base_dir / run_stem(args)
    run_dir.mkdir(parents=True, exist_ok=True)

    log_file   = open(run_dir / 'train.log', 'w', buffering=1)
    sys.stdout = Tee(log_file)

    print_gpu_profile(args._gpu_profile)
    print(f"RUN_DIR: {run_dir}")
    print("=" * 72)
    print(f"VREncoder TS-JEPA Training Run")
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
    print("Training")
    print(f"  npy_dir        : {args.npy_dir}")
    print(f"  epochs         : {args.epochs}")
    print(f"  batch_size     : {args.batch_size}")
    print(f"  lr             : {args.lr}")
    print(f"  min_lr         : {args.min_lr}")
    print(f"  warmup_epochs  : {args.warmup_epochs}")
    print(f"  max_len        : {args.max_len}")
    print(f"  target_ratio   : {args.target_ratio}")
    print(f"  n_target_blocks: {args.n_target_blocks}")
    print(f"  mask_type      : {args.mask_type}")
    print(f"  device_mask    : {args.device_mask or '(none)'}")
    print(f"  feat_grp_msk   : {args.feature_group_mask or '(none)'}")
    print(f"  ema_start      : {args.ema_start}")
    print(f"  kinematics     : {args.kinematics.upper()}")
    print(f"  val_fraction   : {args.val_fraction}")
    print(f"  samples/ep     : {args.samples_per_epoch or '(= n_files)'}")
    print(f"  sampling_α     : {args.sampling_alpha}")
    print(f"  compile        : {args.compile}")
    print(f"  seed           : {args.seed}")
    print(f"  num_workers    : {args.num_workers}")
    print(f"  eval_interval  : {args.embed_eval_interval} epochs")
    print("=" * 72)
    print()

    return run_dir


# ---------------------------------------------------------------------------
# Validation loss
# ---------------------------------------------------------------------------

@torch.no_grad()
def _compute_val_loss(model, val_loader, device, amp_dtype, args, feat_mask_cols) -> float:
    model.eval()
    total, n = 0.0, 0
    for x, lengths in val_loader:
        x       = x.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=device.type == 'cuda'):
            loss = model.jepa_loss(
                x, lengths,
                target_ratio=args.target_ratio,
                n_target_blocks=args.n_target_blocks,
                mask_type=args.mask_type,
                feat_mask_cols=feat_mask_cols,
            )
        total += loss.item()
        n += 1
    model.train()
    return total / n if n > 0 else float('inf')


# ---------------------------------------------------------------------------
# Periodic embed + probe
# ---------------------------------------------------------------------------

def _periodic_eval(run_dir: Path, epoch: int, args, use_best_ckpt: bool = False):
    """
    Embed FAB and DEVCOM_s2 targets then run linear probes for the relevant
    objective metrics.  Output is captured and printed through the active Tee logger.
    """
    ckpt_name = 'checkpoint_best.pt' if use_best_ckpt else 'checkpoint_latest.pt'
    ckpt = run_dir / ckpt_name
    if not ckpt.exists():
        ckpt = run_dir / 'checkpoint_latest.pt'
    if not ckpt.exists():
        print(f'[Eval] checkpoint not found — skipping epoch {epoch} eval')
        return

    embed_script = Path(__file__).parent.parent / 'pipeline' / 'embed_target_data.py'

    eval_targets = []
    if args.fab_eval_dir and Path(args.fab_eval_dir).exists():
        eval_targets.append((args.fab_eval_dir, 'FAB', ['portScore']))
    if args.devcom_eval_dir and Path(args.devcom_eval_dir).exists():
        eval_targets.append((args.devcom_eval_dir, 'D3',
                             ['bot_dist_mean_s3', 'firing_accuracy_AOBJ_s3']))

    if not eval_targets:
        print(f'[Eval] no valid eval dirs found — skipping epoch {epoch} eval')
        return

    print(f'\n{"="*72}')
    print(f'Periodic eval — epoch {epoch}  ckpt: {ckpt}')
    print(f'{"="*72}')

    eval_split_mode = getattr(args, 'eval_split_mode', None)
    if eval_split_mode == 'val':
        extra_embed_args = ['--split_keys', 'train,val']
        extra_probe_args = ['--train_split', 'train', '--eval_split', 'val']
    elif eval_split_mode == 'test':
        extra_embed_args = ['--split_keys', 'train,val,test']
        extra_probe_args = ['--train_split', 'train+val', '--eval_split', 'test']
    else:
        extra_embed_args = []
        extra_probe_args = []

    for data_dir, objective, target_cols in eval_targets:
        print(f'\n[Eval] {Path(data_dir).name}  objective={objective}')
        cmd = [sys.executable, str(embed_script),
               '--ckpt',        str(ckpt),
               '--data_dir',    data_dir,
               '--stride',      str(args.batch_size // 2),
               '--objective',   objective,
               '--target_cols'] + target_cols
        if getattr(args, 'eval_window_pool', None):
            cmd += ['--window_pool', args.eval_window_pool]
        if getattr(args, 'eval_session_pool', None):
            cmd += ['--session_pool', args.eval_session_pool]
        cmd += extra_embed_args + extra_probe_args
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(f'[Eval ERROR]\n{result.stderr}')

    print(f'{"="*72}\n')
    sys.stdout.flush()


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

    feature_cols = build_feature_cols(args.kinematics)
    n_features   = len(feature_cols)

    _fcols = feat_col_indices(args.kinematics, args.device_mask, args.feature_group_mask)
    feat_mask_cols = _fcols if _fcols else None

    run_dir = make_run_dir(out_dir, args)
    print(f'Device: {device}')
    if device.type == 'cuda':
        print(f'GPU   : {torch.cuda.get_device_name(0)}')
    print(f'Features: {n_features} ({args.kinematics.upper()})')

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
        np.savez(
            stats_path,
            mean=mean,
            std=std,
            max_len=args.max_len,
            seed=args.seed,
            n_sample=norm_n_sample,
            sampling_alpha=args.sampling_alpha,
            kinematics=args.kinematics.upper(),
        )
        print(f'  saved to {stats_path}')

    # ----------------------------------------------------------------- dataset
    val_loader       = None
    _train_allowlist = None
    _N_VAL_BATCHES   = 64

    if args.val_fraction > 0:
        all_npy = sorted(Path(npy_dir).glob('*.npy'))
        _ds_file_groups: dict[str, list[Path]] = {}
        for f in all_npy:
            _ds_file_groups.setdefault(f.stem.split('_')[0], []).append(f)
        _train_files, _val_files = [], []
        for ds in sorted(_ds_file_groups):
            files = _ds_file_groups[ds]
            n_val = max(1, round(len(files) * args.val_fraction))
            _val_files.extend(files[-n_val:])
            _train_files.extend(files[:-n_val])
        _train_allowlist = set(_train_files)
        print(
            f'Train/val split: {len(_train_files):,} train / {len(_val_files):,} val files '
            f'({args.val_fraction:.0%} per dataset)',
            flush=True,
        )
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
    loader  = DataLoader(
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
    model = TSJEPA(
        n_features=n_features,
        embed_dim=args.embed_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        max_len=args.max_len,
        pred_layers=args.pred_layers,
        pred_ffn_dim=args.pred_ffn_dim,
    ).to(device)

    trainable_params = list(model.context_encoder.parameters()) + \
                       list(model.predictor.parameters())
    n_params_total     = sum(p.numel() for p in model.parameters())
    n_params_trainable = sum(p.numel() for p in trainable_params)
    print(f'Model  : {n_params_total:,} total params  '
          f'({n_params_trainable:,} trainable  '
          f'{n_params_total - n_params_trainable:,} EMA target encoder)')

    jepa_model = model
    if args.compile and device.type == 'cuda':
        print('Compiling model with torch.compile ...')
        model = torch.compile(model)

    _PRECISION_MAP = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}
    amp_dtype = _PRECISION_MAP[args.precision]
    if amp_dtype == torch.bfloat16 and device.type == 'cuda' and not torch.cuda.is_bf16_supported():
        print('WARNING: bf16 requested but not supported on this GPU; falling back to fp16')
        amp_dtype = torch.float16
    use_scaler = (device.type == 'cuda') and (amp_dtype == torch.float16)
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)
    print(f'AMP dtype: {amp_dtype}')

    optimizer    = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    total_steps  = len(loader) * args.epochs
    warmup_steps = len(loader) * args.warmup_epochs
    min_lr_ratio = args.min_lr / args.lr
    scheduler    = cosine_schedule_with_warmup(
        optimizer, total_steps, warmup_steps, min_lr_ratio=min_lr_ratio
    )

    # --------------------------------------------------------------- resume
    best_loss   = float('inf')
    global_step = 0
    start_epoch = 1
    if args.resume:
        print(f'Resuming from: {args.resume}')
        ckpt_r = torch.load(args.resume, map_location=device, weights_only=False)
        jepa_model.load_state_dict(ckpt_r['model_state'])
        optimizer.load_state_dict(ckpt_r['optimizer_state'])
        scheduler.load_state_dict(ckpt_r['scheduler_state'])
        scaler.load_state_dict(ckpt_r['scaler_state'])
        best_loss   = ckpt_r.get('best_loss', float('inf'))
        global_step = ckpt_r.get('global_step', 0)
        start_epoch = ckpt_r['epoch'] + 1
        print(f'  -> epoch {ckpt_r["epoch"]} restored; resuming at epoch {start_epoch} '
              f'(global_step={global_step})')

    # ---------------------------------------------------------------- training
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches  = 0
        t0         = time.perf_counter()

        for x, lengths in loader:
            x       = x.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=device.type == 'cuda'):
                loss = model.jepa_loss(
                    x, lengths,
                    target_ratio=args.target_ratio,
                    n_target_blocks=args.n_target_blocks,
                    mask_type=args.mask_type,
                    feat_mask_cols=feat_mask_cols,
                )

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(trainable_params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            ema_decay = ema_decay_schedule(global_step, total_steps, start=args.ema_start)
            jepa_model.update_target_encoder(ema_decay)

            total_loss  += loss.item()
            n_batches   += 1
            global_step += 1

        avg_loss   = total_loss / n_batches
        lr_now     = scheduler.get_last_lr()[0]
        ema_now    = ema_decay_schedule(global_step, total_steps, start=args.ema_start)
        epoch_secs = time.perf_counter() - t0

        if val_loader is not None:
            val_loss = _compute_val_loss(model, val_loader, device, amp_dtype, args, feat_mask_cols)
            print(
                f'Epoch {epoch:3d}/{args.epochs}  '
                f'train={avg_loss:.6f}  val={val_loss:.6f}  '
                f'lr={lr_now:.2e}  ema={ema_now:.5f}  '
                f'time={epoch_secs:.1f}s'
            )
            checkpoint_score = val_loss
        else:
            print(
                f'Epoch {epoch:3d}/{args.epochs}  '
                f'loss={avg_loss:.6f}  '
                f'lr={lr_now:.2e}  '
                f'ema={ema_now:.5f}  '
                f'time={epoch_secs:.1f}s'
            )
            checkpoint_score = avg_loss

        ckpt = {
            'epoch':           epoch,
            'best_loss':       best_loss,
            'global_step':     global_step,
            'model_state':     jepa_model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'scaler_state':    scaler.state_dict(),
            'norm_mean':       mean,
            'norm_std':        std,
            'args': {
                'kinematics':        args.kinematics.upper(),
                'n_features':        n_features,
                'embed_dim':         args.embed_dim,
                'n_heads':           args.n_heads,
                'n_layers':          args.n_layers,
                'ffn_dim':           args.ffn_dim,
                'dropout':           args.dropout,
                'max_len':           args.max_len,
                'pred_layers':       args.pred_layers,
                'pred_ffn_dim':      args.pred_ffn_dim,
                'target_ratio':      args.target_ratio,
                'n_target_blocks':   args.n_target_blocks,
                'mask_type':         args.mask_type,
                'device_mask':       args.device_mask,
                'feature_group_mask': args.feature_group_mask,
                'feat_mask_cols':          feat_mask_cols,
                'sampling_alpha':          args.sampling_alpha,
                'samples_per_epoch':       dataset._n,
                'dataset_sampling_shares': dataset.sampling_shares,
            },
        }
        torch.save(ckpt, run_dir / 'checkpoint_latest.pt')

        if checkpoint_score < best_loss:
            best_loss = checkpoint_score
            torch.save(ckpt, run_dir / 'checkpoint_best.pt')
            metric_label = 'val' if val_loader is not None else 'loss'
            print(f'  -> new best checkpoint saved ({metric_label}={best_loss:.6f})')

        if args.embed_eval_interval > 0 and epoch % args.embed_eval_interval == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _periodic_eval(run_dir, epoch, args,
                           use_best_ckpt=getattr(args, 'eval_use_best_ckpt', False))

    _metric_label = 'val' if val_loader is not None else 'loss'
    print(f'\nTraining complete. Best {_metric_label}: {best_loss:.6f}')
    print(f'Run dir  : {run_dir}')
    print(f'Finished : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    _profile = get_gpu_profile()
    p = argparse.ArgumentParser(description='Train VR motion encoder (TS-JEPA)')

    # Data / output
    p.add_argument('--npy_dir',           type=str,   required=True,
                   help='Directory of .npy files from preprocess_to_npy.py')
    p.add_argument('--out_dir',           default=str(_CHOROS_ROOT / 'outputs' / 'checkpoints'))
    p.add_argument('--samples_per_epoch', type=int,   default=0,
                   help='Crops drawn per epoch; 0 = one per file (default)')
    p.add_argument('--sampling_alpha',    type=float, default=0.5,
                   help='Dataset-level temperature exponent for the two-level sampler. '
                        '1.0 = window-proportional. 0.5 = sqrt-balanced (default). '
                        '0.0 = equal dataset probability.')
    p.add_argument('--exclude_datasets', nargs='*', metavar='DATASET', default=None,
                   help='Dataset names to exclude from training. '
                        'E.g. --exclude_datasets who Wu17')
    p.add_argument('--compile',           action='store_true', default=_profile['compile'],
                   help='torch.compile the model (default: auto per GPU profile)')
    p.add_argument('--no_compile',        dest='compile', action='store_false')
    p.add_argument('--precision',         type=str,   default=_profile['precision'],
                   choices=['bf16', 'fp16', 'fp32'],
                   help='AMP precision (default: auto per GPU — bf16 if supported, else fp16)')
    p.add_argument('--resume',            type=str,   default=None,
                   help='Path to checkpoint_latest.pt to resume training from')

    # Training
    p.add_argument('--epochs',         type=int,   default=50)
    p.add_argument('--batch_size',     type=int,   default=_profile['batch_size'])
    p.add_argument('--lr',             type=float, default=1e-3)
    p.add_argument('--min_lr',         type=float, default=1e-6,
                   help='LR floor at end of cosine decay (default: 1e-6)')
    p.add_argument('--warmup_epochs',  type=int,   default=2,
                   help='Linear warmup epochs before cosine decay (default: 2)')
    p.add_argument('--max_len',        type=int,   default=MAX_LEN)
    p.add_argument('--num_workers',    type=int,   default=_profile['num_workers'])
    p.add_argument('--seed',           type=int,   default=42)
    p.add_argument('--kinematics',     type=str,   default=KINEMATICS,
                   help='Kinematic orders to include as features: any combination of '
                        'P (position/orientation), V (velocity), A (acceleration), '
                        'J (jerk). E.g. "PVAJ" for all, "AJ" for accel+jerk. '
                        f'(default: {KINEMATICS})')
    p.add_argument('--val_fraction', type=float, default=0.1,
                   help='Fraction of files per dataset held out as a pretraining validation set '
                        'for checkpoint_best.pt selection. 0 disables (default: 0.1)')

    # TS-JEPA specific
    p.add_argument('--target_ratio',    type=float, default=TARGET_RATIO,
                   help='Fraction of valid timesteps used as prediction targets')
    p.add_argument('--n_target_blocks', type=int,   default=N_TARGET_BLOCKS,
                   help='Number of contiguous target blocks sampled per sequence '
                        '(only used when mask_type=span)')
    p.add_argument('--mask_type',       type=str,   default='span',
                   choices=['span', 'random'],
                   help='Target masking strategy: contiguous span blocks (default) '
                        'or uniform random timestep selection.')
    p.add_argument('--device_mask', nargs='*', metavar='DEVICE',
                   help='Zero out feature columns for these devices globally. '
                        'E.g. --device_mask left right.  Valid: head left right.')
    p.add_argument('--feature_group_mask', nargs='*', metavar='GROUP',
                   help='Zero out feature columns for these kinematic groups globally. '
                        'E.g. --feature_group_mask V A.  Valid: P V A J.')
    p.add_argument('--ema_start',       type=float, default=EMA_DECAY_START,
                   help='Initial EMA decay for the target encoder (cosine-annealed to 1.0)')

    # Encoder architecture
    p.add_argument('--embed_dim',   type=int,   default=EMBED_DIM)
    p.add_argument('--n_heads',     type=int,   default=N_HEADS)
    p.add_argument('--n_layers',    type=int,   default=N_LAYERS)
    p.add_argument('--ffn_dim',     type=int,   default=FFN_DIM)
    p.add_argument('--dropout',     type=float, default=DROPOUT)

    # Predictor architecture
    p.add_argument('--pred_layers',  type=int, default=PRED_LAYERS,
                   help='Number of Transformer layers in the predictor')
    p.add_argument('--pred_ffn_dim', type=int, default=PRED_FFN_DIM,
                   help='FFN hidden dim in the predictor')

    # Periodic eval
    p.add_argument('--embed_eval_interval', type=int, default=5,
                   help='Run embed+probe eval every N epochs using checkpoint_latest '
                        '(0 = disabled, default: 5)')
    p.add_argument('--fab_eval_dir',    type=str,
                   default=str(_DATA_ROOT / 'aligned' / 'target_FAB'),
                   help='FAB target directory for periodic eval embeddings')
    p.add_argument('--devcom_eval_dir', type=str,
                   default=str(_DATA_ROOT / 'aligned' / 'target_DEVCOM_s2'),
                   help='DEVCOM_s2 target directory for periodic eval embeddings')
    p.add_argument('--eval_window_pool', type=str, default=None,
                   choices=['mean', 'mean_std_max', 'layer_avg', 'cls', 'mean_all', 'last'],
                   help='When set, periodic eval runs only this window pool instead of all four')
    p.add_argument('--eval_session_pool', type=str, default=None,
                   choices=['mean', 'stat4'],
                   help='When set, periodic eval runs only this session pool instead of all four')
    p.add_argument('--eval_split_mode', type=str, default=None,
                   choices=['val', 'test'],
                   help="Activates split-aware periodic eval. "
                        "'val': embed train+val, probe train→val (Phase 1 / hparam search). "
                        "'test': embed all splits, probe train+val→test (Phase 2/3).")
    p.add_argument('--eval_use_best_ckpt', action='store_true', default=False,
                   help='Use checkpoint_best.pt instead of checkpoint_latest.pt for periodic '
                        'downstream evals (use in hparam search; default: False)')

    args = p.parse_args()
    args._gpu_profile = _profile
    return args


if __name__ == '__main__':
    train(parse_args())
