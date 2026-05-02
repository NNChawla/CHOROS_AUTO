"""
Generate embeddings for every parquet file in aligned_target_data using a
trained VREncoder checkpoint.

Each invocation runs all four pooling configurations and, when --objective is
given (or auto-detected from the data directory name), runs the linear probes
for the relevant target columns after each embedding directory is produced.

Pool configurations
-------------------
  1. window=mean        session=mean
  2. window=mean_std_max session=mean
  3. window=layer_avg   session=mean
  4. window=mean        session=stat4

For long sequences (longer than the model's max_len window), a sliding window
is used and per-window embeddings are produced then aggregated into a single
session-level embedding.  Per-window embeddings are also saved for downstream
temporal analysis.

Usage
-----
  conda run -n CHOROS python embed_target_data.py [options]

Key options
-----------
  --ckpt          Path to checkpoint file (default: most recent checkpoint_best.pt)
  --data_dir      aligned target data directory
  --out_dir       Output directory for embeddings (default: outputs/embeddings)
  --stride        Sliding window stride in timesteps (default: 64)
  --batch_size    Number of windows processed simultaneously on GPU (default: 256)
  --layer_avg_n   Number of last transformer layers to average for layer_avg pool (default: 4)
  --objective     Probe objective: FAB, D2, or D3 (auto-detected from data_dir if omitted)
  --target_cols   Target columns to probe (space-separated; defaults per objective if omitted)

Output
------
  Four subdirectories under <out_dir>, one per pool configuration:
    <out_dir>/<timestamp>_<dataset>_..._wp<WP>_sp<SP>/
      per_file/<filename_stem>.npy   — shape (n_windows, window_embed_dim), float32
      sequence_embeddings.npy        — shape (n_files, session_embed_dim), float32
      sequence_embeddings.csv        — filename, embedding columns e0..e(D-1)

  Each directory is followed by linear-probe results if --objective is provided.
  Machine-parseable output: lines starting with "OUT_DIR:" give each output path.
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

_CHOROS_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_CHOROS_ROOT / 'src'))

from vr_encoder import VREncoder, FEATURE_COLS, N_FEATURES
from features import build_feature_cols
from vr_encoder_tsjepa import TSJEPA

# Four pool configurations run on every invocation.
POOL_CONFIGS = [
    ('mean',         'mean'),
    ('mean_std_max', 'mean'),
    ('layer_avg',    'mean'),
    ('mean',         'stat4'),
]

# Default target columns per objective (used when --target_cols is not given).
_DEFAULT_TARGET_COLS = {
    'FAB': ['portScore'],
    'D2':  [],   # all columns – probed with target_col=None
    'D3':  ['bot_dist_mean_s3', 'firing_accuracy_AOBJ_s3'],
}


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_checkpoint(ckpt_path: Path, device: torch.device) -> tuple[nn.Module, dict]:
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg   = ckpt['args']
    state = ckpt['model_state']

    if any(k.startswith('context_encoder.') for k in state):
        model = TSJEPA(
            n_features=cfg['n_features'],
            embed_dim=cfg['embed_dim'],
            n_heads=cfg['n_heads'],
            n_layers=cfg['n_layers'],
            ffn_dim=cfg['ffn_dim'],
            dropout=cfg.get('dropout', 0.0),
            max_len=cfg['max_len'],
            pred_layers=cfg.get('pred_layers', 2),
            pred_ffn_dim=cfg.get('pred_ffn_dim', 256),
        ).to(device)
    else:
        model = VREncoder(
            n_features=cfg['n_features'],
            embed_dim=cfg['embed_dim'],
            n_heads=cfg['n_heads'],
            n_layers=cfg['n_layers'],
            ffn_dim=cfg['ffn_dim'],
            dropout=cfg.get('dropout', 0.0),
            max_len=cfg['max_len'],
        ).to(device)

    model.load_state_dict(state)
    model.eval()
    return model, ckpt


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

def sequence_to_windows(
    x:       np.ndarray,
    max_len: int,
    stride:  int,
) -> tuple[np.ndarray, np.ndarray]:
    T = len(x)
    starts = list(range(0, max(1, T - max_len + 1), stride))
    if T > max_len and (T - max_len) not in starts:
        starts.append(T - max_len)

    windows = []
    lengths = []
    for s in starts:
        end   = s + max_len
        chunk = x[s:end]
        L     = len(chunk)
        if L < max_len:
            pad   = np.zeros((max_len - L, x.shape[1]), dtype=np.float32)
            chunk = np.concatenate([chunk, pad], axis=0)
        windows.append(chunk)
        lengths.append(L)

    return np.stack(windows, axis=0).astype(np.float32), np.array(lengths, dtype=np.int64)


# ---------------------------------------------------------------------------
# Hidden-state extraction (model-agnostic)
# ---------------------------------------------------------------------------

def _model_hidden_states(
    model:        nn.Module,
    x_b:          torch.Tensor,
    padding_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(model, TSJEPA):
        T = x_b.shape[1]
        all_positions = torch.arange(T, device=x_b.device)
        cls_emb, token_hidden = model.context_encoder(x_b, all_positions, padding_mask)
        return cls_emb, token_hidden

    captured: dict[str, torch.Tensor] = {}

    def _hook(module, inp, out):
        captured['out'] = out

    h = model.norm.register_forward_hook(_hook)
    model(x_b, padding_mask)
    h.remove()

    out = captured['out']
    return out[:, 0], out[:, 1:]


def _model_layer_avg_states(
    model:        nn.Module,
    x_b:          torch.Tensor,
    padding_mask: torch.Tensor | None,
    n_layers_avg: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(model, TSJEPA):
        layers = model.context_encoder.transformer.layers
        norm   = model.context_encoder.norm
    else:
        layers = model.transformer.layers
        norm   = model.norm

    selected = list(layers)[-min(n_layers_avg, len(layers)):]
    B, T = x_b.shape[:2]

    captured: list[torch.Tensor] = []
    hooks = []
    for layer in selected:
        def _hook(module, inp, out, _c=captured):
            _c.append(out)
        hooks.append(layer.register_forward_hook(_hook))

    if isinstance(model, TSJEPA):
        T = x_b.shape[1]
        all_positions = torch.arange(T, device=x_b.device)
        model.context_encoder(x_b, all_positions, padding_mask)
    else:
        model(x_b, padding_mask)

    for h in hooks:
        h.remove()

    avg_out = torch.stack(captured, dim=0).mean(0)

    if avg_out.dim() == 2:
        # Flash transformer produces packed (total_valid, D); unpack to (B, 1+T, D).
        # ContextEncoder prepends a CLS token (always valid), so the transformer sees
        # (B, 1+T) tokens — the packed count is B + sum(~padding_mask), not sum(~padding_mask).
        cls_col = x_b.new_zeros(B, 1, dtype=torch.bool)  # CLS is never masked
        if padding_mask is not None:
            full_mask = torch.cat([cls_col, padding_mask], dim=1)  # (B, 1+T)
        else:
            full_mask = x_b.new_zeros(B, 1 + T, dtype=torch.bool)
        valid = ~full_mask
        out_3d = x_b.new_zeros(B, 1 + T, avg_out.shape[-1], dtype=avg_out.dtype)
        out_3d[valid] = avg_out
        avg_out = out_3d

    avg_out = norm(avg_out)
    return avg_out[:, 0], avg_out[:, 1:]


# ---------------------------------------------------------------------------
# Window-level pooling
# ---------------------------------------------------------------------------

def _apply_window_pool(
    strategy:     str,
    cls_emb:      torch.Tensor,
    token_hidden: torch.Tensor,
    valid_mask:   torch.Tensor,
) -> torch.Tensor:
    if strategy == 'cls':
        return cls_emb

    mask_f  = valid_mask.float().unsqueeze(-1)
    n_valid = mask_f.sum(1).clamp(min=1)

    if strategy in ('mean', 'layer_avg'):
        return (token_hidden * mask_f).sum(1) / n_valid

    if strategy == 'mean_all':
        return token_hidden.mean(dim=1)

    if strategy == 'last':
        lengths = valid_mask.long().sum(dim=1).clamp(min=1)
        idx     = lengths - 1
        return token_hidden[torch.arange(len(idx), device=token_hidden.device), idx]

    if strategy == 'mean_std_max':
        mean_e = (token_hidden * mask_f).sum(1) / n_valid
        diff   = (token_hidden - mean_e.unsqueeze(1)) * mask_f
        std_e  = ((diff ** 2).sum(1) / n_valid).sqrt()
        tok_m  = token_hidden.masked_fill(~valid_mask.unsqueeze(-1), float('-inf'))
        max_e  = tok_m.max(dim=1).values
        return torch.cat([mean_e, std_e, max_e], dim=-1)

    raise ValueError(f'Unknown window_pool: {strategy!r}')


# ---------------------------------------------------------------------------
# Session-level pooling
# ---------------------------------------------------------------------------

def _apply_session_pool(strategy: str, window_embs: np.ndarray) -> np.ndarray:
    if strategy == 'mean':
        return window_embs.mean(axis=0)

    if strategy == 'stat4':
        return np.concatenate([
            window_embs.mean(axis=0),
            window_embs.std(axis=0),
            np.percentile(window_embs, 25, axis=0),
            np.percentile(window_embs, 75, axis=0),
        ])

    raise ValueError(f'Unknown session_pool: {strategy!r}')


# ---------------------------------------------------------------------------
# Core embedding loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def embed_windows(
    model:       nn.Module,
    windows:     np.ndarray,
    lengths:     np.ndarray,
    max_len:     int,
    batch_size:  int,
    device:      torch.device,
    window_pool: str = 'mean',
    layer_avg_n: int = 4,
) -> np.ndarray:
    all_embs = []
    n = len(windows)
    for start in range(0, n, batch_size):
        end  = min(start + batch_size, n)
        x_b  = torch.from_numpy(windows[start:end]).to(device)
        L_b  = torch.from_numpy(lengths[start:end]).to(device)

        T            = max_len
        arange       = torch.arange(T, device=device).unsqueeze(0)
        padding_mask = arange >= L_b.unsqueeze(1)
        valid_mask   = ~padding_mask

        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            if window_pool == 'layer_avg':
                cls_emb, token_hidden = _model_layer_avg_states(
                    model, x_b, padding_mask, layer_avg_n)
            else:
                cls_emb, token_hidden = _model_hidden_states(model, x_b, padding_mask)

        emb = _apply_window_pool(window_pool, cls_emb, token_hidden, valid_mask)
        all_embs.append(emb.float().cpu().numpy())

    return np.concatenate(all_embs, axis=0)


# ---------------------------------------------------------------------------
# Run labeling
# ---------------------------------------------------------------------------

def run_stem(ckpt_args: dict, stride: int, dataset: str,
             window_pool: str, session_pool: str) -> str:
    ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
    mask = ckpt_args.get('mask_ratio', ckpt_args.get('target_ratio', 'unk'))
    kin  = ckpt_args.get('kinematics', 'unk')
    return (
        f"{ts}"
        f"_{dataset}"
        f"_dim{ckpt_args['embed_dim']}"
        f"_l{ckpt_args['n_layers']}"
        f"_ml{ckpt_args['max_len']}"
        f"_mask{mask}"
        f"_kin{kin}"
        f"_stride{stride}"
        f"_wp{window_pool}"
        f"_sp{session_pool}"
    )


# ---------------------------------------------------------------------------
# Single-pool embedding
# ---------------------------------------------------------------------------

def embed_all(
    ckpt_path:     Path,
    data_dir:      Path,
    out_dir:       Path,
    stride:        int,
    batch_size:    int,
    window_pool:   str,
    session_pool:  str,
    layer_avg_n:   int = 4,
    model:         nn.Module | None = None,
    ckpt:          dict | None = None,
    include_files: set[Path] | None = None,
) -> Path:
    """
    Embed parquet files in data_dir with the given pooling strategy.
    When include_files is provided, only those files are embedded (split filtering).
    Reuses a pre-loaded model+ckpt if supplied (avoids redundant checkpoint loads).
    Returns the output directory path.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model is None or ckpt is None:
        model, ckpt = load_checkpoint(ckpt_path, device)

    max_len      = ckpt['args']['max_len']
    embed_dim    = ckpt['args']['embed_dim']
    norm_mean    = ckpt['norm_mean'].astype(np.float32)
    norm_std     = ckpt['norm_std'].astype(np.float32)
    feature_cols = build_feature_cols(ckpt['args'].get('kinematics', 'P'))
    dataset      = data_dir.name

    window_embed_dim  = embed_dim * (3 if window_pool == 'mean_std_max' else 1)
    session_embed_dim = window_embed_dim * (4 if session_pool == 'stat4' else 1)

    emb_dir      = out_dir / run_stem(ckpt['args'], stride, dataset, window_pool, session_pool)
    per_file_dir = emb_dir / 'per_file'
    per_file_dir.mkdir(parents=True, exist_ok=True)

    print(f'  Pool        : window={window_pool}  session={session_pool}')
    print(f'  Output dir  : {emb_dir}')
    print(f'  Window emb  : {window_embed_dim}d   Session emb: {session_embed_dim}d')

    all_parquet   = sorted(data_dir.glob('*.parquet'))
    parquet_files = [f for f in all_parquet if include_files is None or f in include_files]
    if include_files is not None:
        print(f'  Split filter: {len(parquet_files)}/{len(all_parquet)} files')
    seq_names      = []
    seq_embeddings = []

    for i, fpath in enumerate(parquet_files):
        df = pd.read_parquet(fpath, columns=feature_cols)
        x  = np.nan_to_num(df.values.astype(np.float32), nan=0.0)
        x  = (x - norm_mean) / (norm_std + 1e-8)
        x  = np.clip(x, -10.0, 10.0)

        windows, lengths = sequence_to_windows(x, max_len=max_len, stride=stride)
        embs = embed_windows(
            model, windows, lengths,
            max_len=max_len, batch_size=batch_size, device=device,
            window_pool=window_pool, layer_avg_n=layer_avg_n,
        )

        np.save(per_file_dir / f'{fpath.stem}.npy', embs)
        seq_embeddings.append(_apply_session_pool(session_pool, embs))
        seq_names.append(fpath.stem)

        if (i + 1) % 50 == 0 or (i + 1) == len(parquet_files):
            print(f'    [{i+1:4d}/{len(parquet_files)}] {fpath.name} '
                  f'→ {len(windows)} windows, shape {embs.shape}')

    seq_arr = np.stack(seq_embeddings, axis=0).astype(np.float32)
    np.save(emb_dir / 'sequence_embeddings.npy', seq_arr)

    col_names = [f'e{i}' for i in range(session_embed_dim)]
    df_out    = pd.DataFrame(seq_arr, columns=col_names)
    df_out.insert(0, 'filename', seq_names)
    df_out.to_csv(emb_dir / 'sequence_embeddings.csv', index=False)

    print(f'  Sequences   : {seq_arr.shape}')
    print(f'OUT_DIR: {emb_dir}')
    return emb_dir


# ---------------------------------------------------------------------------
# Objective helpers
# ---------------------------------------------------------------------------

def infer_objective(data_dir: str) -> str | None:
    name = Path(data_dir).name
    if 'FAB' in name:
        return 'FAB'
    if 'DEVCOM' in name:
        return 'D3'
    return None


# ---------------------------------------------------------------------------
# All-pools embedding + probing
# ---------------------------------------------------------------------------

def embed_and_probe(args) -> list[Path]:
    """
    Embed all files in args.data_dir and run linear probes.

    When args.window_pool and args.session_pool are both set, runs only that
    single pool configuration.  Otherwise runs all four POOL_CONFIGS.
    Returns list of output directory paths (one per pool config run).
    """
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = Path(args.ckpt)
    data_dir  = Path(args.data_dir)
    out_dir   = Path(args.out_dir)
    objective = args.objective
    split_keys_raw = getattr(args, 'split_keys', None)
    probe_train_split = getattr(args, 'train_split', None)
    probe_eval_split  = getattr(args, 'eval_split', None)

    if (probe_train_split is None) != (probe_eval_split is None):
        raise ValueError('--train_split and --eval_split must be provided together')
    if (
        split_keys_raw
        and split_keys_raw.lower() != 'all'
        and objective
        and not (probe_train_split and probe_eval_split)
    ):
        raise ValueError(
            '--split_keys with probing requires --train_split and --eval_split'
        )

    print(f'Device      : {device}')
    print(f'Checkpoint  : {ckpt_path}')
    print(f'Data dir    : {data_dir}')
    print(f'Stride      : {args.stride}')
    print(f'Files       : {len(sorted(data_dir.glob("*.parquet")))}')

    model, ckpt = load_checkpoint(ckpt_path, device)
    print(f'Model       : embed_dim={ckpt["args"]["embed_dim"]}  '
          f'max_len={ckpt["args"]["max_len"]}  '
          f'n_layers={ckpt["args"]["n_layers"]}')

    probe_script = _CHOROS_ROOT / 'training' / 'train_linear_probe.py'
    target_cols  = args.target_cols or (
        _DEFAULT_TARGET_COLS.get(objective, []) if objective else []
    )

    # Determine which files to embed based on --split_keys
    if split_keys_raw and split_keys_raw.lower() != 'all':
        sys.path.insert(0, str(_CHOROS_ROOT / 'training'))
        from splits import filter_devcom_files, filter_fab_files
        keys = [k.strip() for k in split_keys_raw.split(',')]
        all_files = sorted(data_dir.glob('*.parquet'))
        if 'FAB' in data_dir.name:
            include_files: set[Path] | None = set(filter_fab_files(all_files, keys))
        else:
            include_files = set(filter_devcom_files(all_files, keys))
        print(f'Split keys  : {keys}  ({len(include_files)} files selected)')
    else:
        include_files = None

    wp = getattr(args, 'window_pool', None)
    sp = getattr(args, 'session_pool', None)
    pool_configs = [(wp, sp)] if (wp and sp) else POOL_CONFIGS

    out_dirs: list[Path] = []
    for window_pool, session_pool in pool_configs:
        print(f'\n{"─"*60}')
        emb_dir = embed_all(
            ckpt_path, data_dir, out_dir,
            args.stride, args.batch_size,
            window_pool, session_pool, args.layer_avg_n,
            model=model, ckpt=ckpt,
            include_files=include_files,
        )
        out_dirs.append(emb_dir)

        if not objective:
            continue

        cols_to_probe = target_cols if target_cols else [None]
        for tc in cols_to_probe:
            tc_label = tc or '(default)'
            print(f'\n  [Probe] objective={objective}  target={tc_label}')
            cmd = [sys.executable, str(probe_script),
                   '--emb-dir', str(emb_dir),
                   '--objective', objective]
            if tc:
                cmd += ['--target-col', tc]
            if probe_train_split:
                cmd += ['--train_split', probe_train_split]
            if probe_eval_split:
                cmd += ['--eval_split', probe_eval_split]
            result = subprocess.run(cmd, capture_output=True, text=True)
            print(result.stdout)
            if result.returncode != 0:
                print(f'  [Probe ERROR]\n{result.stderr}', file=sys.stderr)

    print(f'\n{"="*60}')
    print(f'All output dirs for {data_dir.name}:')
    for d in out_dirs:
        print(f'  OUT_DIR: {d}')

    return out_dirs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Embed target VR sequences with all four pool configurations')
    _ckpt_runs = sorted((_CHOROS_ROOT / 'outputs' / 'checkpoints').glob('*/checkpoint_best.pt'))
    _ckpt_default = str(_ckpt_runs[-1]) if _ckpt_runs else str(
        _CHOROS_ROOT / 'outputs' / 'checkpoints' / 'checkpoint_best.pt')
    p.add_argument('--ckpt',        default=_ckpt_default)
    p.add_argument('--data_dir',    default=str(_CHOROS_ROOT / 'data' / 'aligned' / 'target'))
    p.add_argument('--out_dir',     default=str(_CHOROS_ROOT / 'outputs' / 'embeddings'))
    p.add_argument('--stride',      type=int, default=64)
    p.add_argument('--batch_size',  type=int, default=256)
    p.add_argument('--layer_avg_n', type=int, default=4,
                   help='Layers to average from the end (for layer_avg pool config)')
    p.add_argument('--objective',   default=None,
                   choices=['FAB', 'D2', 'D3'],
                   help='Probe objective. Auto-detected from data_dir name if omitted.')
    p.add_argument('--target_cols', nargs='*', default=None,
                   help='Columns to probe (space-separated). '
                        'Defaults to the standard columns for the objective.')
    p.add_argument('--window_pool', default=None,
                   choices=['mean', 'mean_std_max', 'layer_avg', 'cls', 'mean_all', 'last'],
                   help='Run only this window pooling strategy instead of all four configs.')
    p.add_argument('--session_pool', default=None,
                   choices=['mean', 'stat4'],
                   help='Run only this session pooling strategy instead of all four configs.')
    p.add_argument('--split_keys', default=None,
                   help="Comma-separated split subsets to embed: 'train,val', 'train,val,test', "
                        "or 'all' (default). Files not in the given splits are skipped.")
    p.add_argument('--train_split', default=None,
                   choices=['train', 'train+val'],
                   help='Fit probe on this split subset (passed through to train_linear_probe).')
    p.add_argument('--eval_split', default=None,
                   choices=['val', 'test'],
                   help='Evaluate probe on this split subset (passed through to train_linear_probe).')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.objective is None:
        args.objective = infer_objective(args.data_dir)
    embed_and_probe(args)
