#!/usr/bin/env python3
"""
auto_tune_jepa.py — VRAM-aware hyperparameter optimizer for PoseJEPA.

Designed for the CHOROS VR motion dataset:
  ~71K files, 26 datasets, ~378 h at 30 Hz, 75 features (PVAJ),
  median session ~60 frames, max_len=128 → ~16 patches per window.

Finds optimal combinations of:
  batch_size, embed_dim, n_heads, n_layers, ffn_dim, pred_ffn_dim, pred_layers

JEPA-specific guidance baked in:
  • Batch size is capped at 2048 by default.  JEPA is non-contrastive, so
    there is no benefit from large batches the way SimCLR needs them.  Larger
    batches reduce steps/epoch and therefore EMA update count, which slows
    convergence.  256–1024 is the practical sweet spot.
  • Width-to-depth ratio matters.  D=768 with L=4 (current default) is
    unusually shallow; the cross-sensor/cross-kinematic masking task benefits
    from hierarchy.  L ≥ embed_dim/128 is a healthy minimum.
  • VRAM fill is not the goal.  For 16-patch sequences the activation memory
    per sample is small; unused VRAM is fine.

Three recommendation tiers:
  [1] Max capacity   — largest model within the params budget
  [2] Depth-balanced — best capacity among architecturally sound configs
                       (n_layers ≥ embed_dim/128, head_dim 64–128)
  [3] Max throughput — fastest samples/sec (useful for rapid ablations)

Usage:
  python training/auto_tune_jepa.py [options]

  --kinematics PVAJ       Feature set (default: PVAJ = 75 features)
  --max_len    128         Frames per sequence window
  --patch_size 8           Frames per patch token
  --target_ratio 0.25      Fraction of patches used as JEPA targets
  --vram_frac  0.85        Fraction of total VRAM to target (default: 0.85)
  --max_params 0           Max params in millions; 0 = auto (5 × n_patches M)
  --max_batch  2048        Cap on batch size searched (default: 2048)
  --max_head_dim 128       Max attention head dimension (default: 128)
  --top_n      10          Configs to benchmark empirically (per tier)
  --no_empirical           Print analytical estimates only (no actual runs)
  --min_batch  64          Minimum acceptable batch size (default: 64)
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import NamedTuple

import torch

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / 'src'))
sys.path.insert(0, str(_ROOT / 'training'))

from features import build_feature_cols


# ---------------------------------------------------------------------------
# Config tuple
# ---------------------------------------------------------------------------

class Config(NamedTuple):
    embed_dim:    int
    n_heads:      int
    n_layers:     int
    ffn_dim:      int
    pred_layers:  int
    pred_ffn_dim: int
    batch_size:   int


# ---------------------------------------------------------------------------
# Analytical estimates
# ---------------------------------------------------------------------------

def count_params(cfg: Config, n_features: int, patch_size: int) -> tuple[int, int]:
    """Return (n_total, n_trainable)."""
    D, F, pF = cfg.embed_dim, cfg.ffn_dim, cfg.pred_ffn_dim
    L, pL, P = cfg.n_layers, cfg.pred_layers, patch_size

    # PyTorch TransformerEncoderLayer per-layer costs
    # Self-attention: in_proj (3D×D + 3D bias) + out_proj (D×D + D)
    attn = 4 * D * D + 4 * D
    # FFN: linear1 (D×F + F) + linear2 (F×D + D)
    ffn = 2 * D * F + F + D
    # Two LayerNorms: 2 × (D weight + D bias)
    lnorms = 4 * D
    per_enc_layer = attn + ffn + lnorms

    # Encoder
    patch_embed = (patch_size * n_features + 1) * D   # linear + bias
    cls_token   = D
    final_ln    = 2 * D
    n_enc = patch_embed + cls_token + L * per_enc_layer + final_ln

    # Predictor
    pred_attn = 4 * D * D + 4 * D
    pred_ffn  = 2 * D * pF + pF + D
    pred_lnorms = 4 * D
    per_pred_layer = pred_attn + pred_ffn + pred_lnorms
    n_pred = D + pL * per_pred_layer + 2 * D   # mask_token + layers + final_ln

    n_total     = 2 * n_enc + n_pred   # context + EMA target + predictor
    n_trainable = n_enc + n_pred       # only context_encoder + predictor are trained
    return n_total, n_trainable


def estimate_vram_gb(
    cfg: Config,
    n_features: int,
    patch_size: int,
    max_len: int,
    target_ratio: float = 0.25,
) -> float:
    """
    Conservative analytical VRAM estimate (GB) for one training step.

    Model params and optimizer states are fp32.  Activations are bf16/fp16 (2 bytes).
    Overhead factor of 1.18 covers CUDA workspace, fragmentation, and cuDNN.
    """
    D, F, pF = cfg.embed_dim, cfg.ffn_dim, cfg.pred_ffn_dim
    L, pL    = cfg.n_layers, cfg.pred_layers
    B, H     = cfg.batch_size, cfg.n_heads
    N        = max_len // patch_size
    K        = max(1, round(N * target_ratio))
    N_ctx    = N - K
    ctx_seq  = N_ctx + 1   # +1 for CLS token

    n_total, n_trainable = count_params(cfg, n_features, patch_size)

    # ---- static memory (fp32) ----
    # Parameters (all, fp32)
    static = n_total * 4
    # Gradients (trainable only, fp32)
    static += n_trainable * 4
    # Adam m + v states (trainable only, fp32)
    static += 2 * n_trainable * 4

    amp = 2   # bf16 / fp16 bytes per element

    # ---- context encoder activations (need backward storage) ----
    # ~10D + F elements per (batch × token) per layer
    act_enc  = B * L * (10 * D + F) * ctx_seq * amp
    # Attention weight matrices: B × H × seq × seq per layer
    attn_enc = B * H * ctx_seq * ctx_seq * amp * L

    # ---- predictor activations (need backward storage) ----
    pred_seq = N_ctx + K   # context latents + target query tokens
    act_pred  = B * pL * (10 * D + pF) * pred_seq * amp
    attn_pred = B * H * pred_seq * pred_seq * amp * pL

    # ---- target encoder (torch.no_grad → transient only) ----
    tgt_seq = K + 1
    act_tgt = 3 * B * tgt_seq * D * amp   # Q/K/V working buffers

    # ---- input tensor ----
    act_input = B * max_len * n_features * amp

    total_act = act_enc + attn_enc + act_pred + attn_pred + act_tgt + act_input
    return (static + total_act) * 1.18 / (1024 ** 3)


def flops_per_forward(
    cfg: Config,
    n_features: int,
    patch_size: int,
    max_len: int,
    target_ratio: float = 0.25,
) -> float:
    """Approximate FLOPs for a single forward pass (training FLOPs ≈ 3×)."""
    D, F, pF = cfg.embed_dim, cfg.ffn_dim, cfg.pred_ffn_dim
    L, pL    = cfg.n_layers, cfg.pred_layers
    N        = max_len // patch_size
    K        = max(1, round(N * target_ratio))
    N_ctx    = N - K

    def transformer_flops(seq, ffn_d, layers):
        per_layer = (
            2 * 3 * D * D * seq        # QKV projections
            + 2 * seq * seq * D        # attention scores + weighted sum
            + 2 * D * D * seq          # output projection
            + 2 * 2 * D * ffn_d * seq  # FFN two linear layers
        )
        return layers * per_layer

    ctx_seq  = N_ctx + 1
    tgt_seq  = K + 1
    pred_seq = N_ctx + K

    fwd = (
        2 * (patch_size * n_features) * D * N_ctx  # patch embedding
        + transformer_flops(ctx_seq,  F,  L)       # context encoder
        + transformer_flops(tgt_seq,  F,  L)       # target encoder (no grad)
        + transformer_flops(pred_seq, pF, pL)      # predictor
    )
    return fwd


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------

def valid_heads(embed_dim: int, max_head_dim: int = 128) -> list[int]:
    """n_heads where embed_dim % n_heads == 0 and head_dim is in [16, max_head_dim].

    FlashAttention hard limit is head_dim <= 256.
    Default max_head_dim=128 matches common practice (64–128 per head).
    Pass max_head_dim=256 to allow large head dims like GPT-style models.
    """
    fa_limit = 256
    limit = min(max_head_dim, fa_limit)
    return [h for h in [2, 4, 6, 8, 12, 16]
            if embed_dim % h == 0 and 16 <= embed_dim // h <= limit]


def build_grid(min_batch: int = 64, max_batch: int = 2048,
               max_head_dim: int = 128) -> list[Config]:
    embed_dims       = [64, 128, 192, 256, 320, 384, 512, 640, 768, 1024]
    n_layers_list    = [2, 4, 6, 8, 10, 12]
    pred_layers_list = [1, 2, 3]
    batch_sizes      = [b for b in [64, 128, 256, 512, 1024, 2048, 4096, 8192]
                        if min_batch <= b <= max_batch]

    grid = []
    for D in embed_dims:
        for H in valid_heads(D, max_head_dim):
            for L in n_layers_list:
                for F in [D * 2, D * 4]:
                    for pL in pred_layers_list:
                        for pF in [D, D * 2]:
                            for B in batch_sizes:
                                grid.append(Config(D, H, L, F, pL, pF, B))
    return grid


# ---------------------------------------------------------------------------
# Pareto frontier helper
# ---------------------------------------------------------------------------

def pareto_front(
    points: list[tuple[float, float]],
) -> list[int]:
    """Return indices of non-dominated points (maximise both axes)."""
    dominated = [False] * len(points)
    for i, (ax, ay) in enumerate(points):
        for j, (bx, by) in enumerate(points):
            if i != j and bx >= ax and by >= ay and (bx > ax or by > ay):
                dominated[i] = True
                break
    return [i for i, d in enumerate(dominated) if not d]


# ---------------------------------------------------------------------------
# Empirical benchmark
# ---------------------------------------------------------------------------

def benchmark(
    cfg:        Config,
    n_features: int,
    patch_size: int,
    max_len:    int,
    device:     torch.device,
    amp_dtype:  torch.dtype,
    n_warmup:   int = 3,
    n_steps:    int = 10,
) -> tuple[float, float]:
    """
    Run n_steps forward+backward passes and return (samples_per_sec, peak_vram_gb).
    Returns (0.0, inf) on OOM.
    """
    from vr_encoder_pose_jepa import PoseJEPA

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    try:
        model = PoseJEPA(
            n_features   = n_features,
            patch_size   = patch_size,
            embed_dim    = cfg.embed_dim,
            n_heads      = cfg.n_heads,
            n_layers     = cfg.n_layers,
            ffn_dim      = cfg.ffn_dim,
            max_len      = max_len,
            pred_layers  = cfg.pred_layers,
            pred_ffn_dim = cfg.pred_ffn_dim,
        ).to(device)

        trainable = (list(model.context_encoder.parameters())
                     + list(model.predictor.parameters()))
        opt = torch.optim.AdamW(trainable, lr=1e-3)

        x       = torch.randn(cfg.batch_size, max_len, n_features, device=device)
        lengths = torch.full((cfg.batch_size,), max_len, dtype=torch.long, device=device)

        def _step():
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=amp_dtype):
                loss, _ = model.jepa_loss(x, lengths)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            model.update_target_encoder(0.996)

        import torch.nn as nn
        for _ in range(n_warmup):
            _step()

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        for _ in range(n_steps):
            _step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        sps     = n_steps * cfg.batch_size / elapsed

        del model, opt, x, lengths
        torch.cuda.empty_cache()
        return sps, peak_gb

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return 0.0, float('inf')


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

_HDR = (f"{'dim':>5} {'heads':>5} {'layers':>6} {'ffn':>6} "
        f"{'p_l':>4} {'p_ffn':>6} {'batch':>6} "
        f"{'params':>9} {'est_GB':>7} {'vram%':>6}")
_SEP = '-' * 76

def _fmt(cfg: Config, n_total: int, est_gb: float, total_vram: float, extra: str = '') -> str:
    pct = 100 * est_gb / total_vram
    return (f"{cfg.embed_dim:>5} {cfg.n_heads:>5} {cfg.n_layers:>6} {cfg.ffn_dim:>6} "
            f"{cfg.pred_layers:>4} {cfg.pred_ffn_dim:>6} {cfg.batch_size:>6} "
            f"{n_total/1e6:>8.2f}M {est_gb:>7.2f} {pct:>5.1f}%{extra}")


def _cli(cfg: Config) -> str:
    return (f"  --batch_size {cfg.batch_size} --embed_dim {cfg.embed_dim} "
            f"--n_heads {cfg.n_heads} --n_layers {cfg.n_layers} "
            f"--ffn_dim {cfg.ffn_dim} --pred_layers {cfg.pred_layers} "
            f"--pred_ffn_dim {cfg.pred_ffn_dim}")


def _section(title: str) -> None:
    print(f'\n{"=" * 68}')
    print(title)
    print('=' * 68)


def _params_per_token_note(n_total: int, n_patches: int,
                            n_features: int = 75, patch_size: int = 8) -> str:
    ppt = n_total / n_patches
    # Raw input dimensionality per patch: n_features × patch_size
    # (e.g. PVAJ: 75 × 8 = 600 floats/patch)
    # Heuristic: flag if params/patch > 10000 × raw_input_dim/patch
    # (i.e. fewer than 1 training example per 10K params, assuming ~35M windows)
    raw_dim = n_features * patch_size
    if ppt > raw_dim * 10000:
        flag = '  ⚠ likely overparameterized'
    elif ppt > raw_dim * 2000:
        flag = '  (large for this feature count)'
    else:
        flag = ''
    return f'{ppt/1e3:.0f}K params/patch{flag}'


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description='VRAM-aware PoseJEPA hyperparameter optimizer')
    ap.add_argument('--kinematics',   default='PVAJ', help='P | PV | PVA | PVAJ  (default: PVAJ)')
    ap.add_argument('--max_len',      type=int,   default=128)
    ap.add_argument('--patch_size',   type=int,   default=8)
    ap.add_argument('--target_ratio', type=float, default=0.25)
    ap.add_argument('--vram_frac',    type=float, default=0.85,
                    help='Fraction of total VRAM to target (default: 0.85)')
    ap.add_argument('--top_n',        type=int,   default=10,
                    help='Configs to empirically benchmark per tier (default: 10)')
    ap.add_argument('--no_empirical', action='store_true',
                    help='Print analytical estimates only (skip actual runs)')
    ap.add_argument('--min_batch',    type=int,   default=64,
                    help='Minimum batch size (default: 64)')
    ap.add_argument('--max_batch',    type=int,   default=2048,
                    help='Maximum batch size searched (default: 2048).  '
                         'JEPA is non-contrastive: large batches reduce steps/epoch '
                         'and EMA update count without adding gradient diversity.')
    ap.add_argument('--max_params',   type=float, default=0.0,
                    help='Max model parameters in millions; 0 = auto (5 × n_patches M)')
    ap.add_argument('--max_head_dim', type=int,   default=128,
                    help='Max attention head dimension; 128 is standard, 256 is FA2 limit '
                         '(default: 128)')
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print('No CUDA GPU detected. This script requires a GPU.')
        sys.exit(1)

    props        = torch.cuda.get_device_properties(0)
    total_vram   = props.total_memory / (1024 ** 3)
    budget_gb    = total_vram * args.vram_frac
    amp_dtype    = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    device       = torch.device('cuda:0')
    feature_cols = build_feature_cols(args.kinematics)
    n_features   = len(feature_cols)
    n_patches    = args.max_len // args.patch_size

    print(f'\nGPU          : {props.name}')
    print(f'Total VRAM   : {total_vram:.1f} GB')
    print(f'Budget       : {budget_gb:.1f} GB  ({args.vram_frac*100:.0f}%)')
    print(f'AMP dtype    : {amp_dtype}')
    print(f'Kinematics   : {args.kinematics.upper()}  ({n_features} features)')
    print(f'max_len      : {args.max_len}   patch_size : {args.patch_size}  '
          f'→ {n_patches} patches')
    print(f'target_ratio : {args.target_ratio}  '
          f'({round(n_patches * args.target_ratio)} target / '
          f'{n_patches - round(n_patches * args.target_ratio)} context patches)')

    # ---------------------------------------------------------------- grid scan
    # Auto max_params: 5 × n_patches million.
    # For 16 patches → 80M, generous enough to cover the productive range
    # without admitting absurdly overparameterized configs.
    max_params_m = args.max_params if args.max_params > 0 else 5 * n_patches
    max_params   = int(max_params_m * 1e6)
    print(f'max_params   : {max_params_m:.0f}M  max_batch : {args.max_batch}')

    print('\nScanning hyperparameter grid …', end='', flush=True)
    grid = build_grid(min_batch=args.min_batch, max_batch=args.max_batch,
                      max_head_dim=args.max_head_dim)

    candidates = []   # (cfg, est_gb, n_total, fwd_flops)
    for cfg in grid:
        est = estimate_vram_gb(cfg, n_features, args.patch_size, args.max_len,
                               args.target_ratio)
        if est > budget_gb:
            continue
        n_total, _ = count_params(cfg, n_features, args.patch_size)
        if n_total > max_params:
            continue
        fwd = flops_per_forward(cfg, n_features, args.patch_size,
                                args.max_len, args.target_ratio)
        candidates.append((cfg, est, n_total, fwd))

    print(f' {len(grid):,} configs scanned → {len(candidates):,} fit '
          f'{budget_gb:.1f} GB and ≤{max_params_m:.0f}M params')

    if not candidates:
        print('No config fits the VRAM budget. Try --vram_frac 0.75 or smaller max_len.')
        sys.exit(1)

    # For display, keep one row per architecture (the batch size that fits and is largest)
    arch_best: dict[tuple, tuple] = {}
    for item in sorted(candidates, key=lambda x: x[0].batch_size, reverse=True):
        key = (item[0].embed_dim, item[0].n_heads, item[0].n_layers,
               item[0].ffn_dim, item[0].pred_layers, item[0].pred_ffn_dim)
        if key not in arch_best:
            arch_best[key] = item

    by_capacity = sorted(arch_best.values(),
                         key=lambda x: (x[2], x[0].batch_size), reverse=True)
    # Throughput proxy: batch_size / (3 × fwd_flops)  — forward+backward ≈ 3×
    by_throughput = sorted(arch_best.values(),
                           key=lambda x: x[0].batch_size / (3 * x[3]) if x[3] else 0,
                           reverse=True)
    # Depth-balanced: reward configs where n_layers ≥ embed_dim/128.
    # A 768-dim transformer "should" have ≥6 layers; 4 is abnormally shallow.
    # Score = n_total_params × depth_ratio², so depth deficits penalise heavily.
    def _depth_score(item: tuple) -> float:
        cfg, est, n_total, fwd = item
        ideal_layers = cfg.embed_dim / 128          # e.g. 6 for D=768, 4 for D=512
        depth_ratio  = min(1.0, cfg.n_layers / ideal_layers)
        return n_total * (depth_ratio ** 2)

    by_depth = sorted(arch_best.values(), key=_depth_score, reverse=True)

    # ------------------------------------------------- analytical tables
    _section('TOP CONFIGS BY MODEL CAPACITY  (largest model within params budget)')
    print(_HDR)
    print(_SEP)
    for item in by_capacity[:15]:
        print(_fmt(item[0], item[2], item[1], total_vram))

    _section('TOP CONFIGS BY DEPTH BALANCE  (n_layers ≥ embed_dim/128 — good for SSL)')
    print(_HDR)
    print(_SEP)
    for item in by_depth[:15]:
        print(_fmt(item[0], item[2], item[1], total_vram))

    _section('TOP CONFIGS BY THROUGHPUT PROXY  (fastest wall-clock epoch)')
    print(_HDR)
    print(_SEP)
    for item in by_throughput[:15]:
        print(_fmt(item[0], item[2], item[1], total_vram))

    if args.no_empirical:
        _section('RECOMMENDED COMMANDS (analytical estimates)')
        cap   = by_capacity[0]
        depth = by_depth[0]
        tp    = by_throughput[0]
        shown_cfgs: set = set()
        idx = 1

        def _show_a(label: str, item: tuple) -> None:
            nonlocal idx
            cfg, est, n_total, _ = item
            key = (cfg.embed_dim, cfg.n_heads, cfg.n_layers, cfg.ffn_dim,
                   cfg.pred_layers, cfg.pred_ffn_dim, cfg.batch_size)
            if key in shown_cfgs:
                return
            shown_cfgs.add(key)
            print(f'[{idx}] {label}')
            print(_cli(cfg))
            print(f'    → {n_total/1e6:.2f}M params  {est:.2f} GB est  '
                  f'({100*est/total_vram:.0f}% VRAM)  '
                  f'({_params_per_token_note(n_total, n_patches, n_features, args.patch_size)})')
            idx += 1

        _show_a('Max capacity:', cap)
        _show_a('Depth-balanced (recommended for representation quality):', depth)
        _show_a('Max throughput (for rapid ablations):', tp)
        return

    # ---------------------------------------------------------------- benchmark
    bench_set:  list[tuple] = []
    seen_keys:  set[tuple]  = set()

    def _add(item):
        cfg = item[0]
        key = (cfg.embed_dim, cfg.n_heads, cfg.n_layers, cfg.ffn_dim,
               cfg.pred_layers, cfg.pred_ffn_dim, cfg.batch_size)
        if key not in seen_keys:
            seen_keys.add(key)
            bench_set.append(item)

    for item in by_capacity[:args.top_n]:
        _add(item)
    for item in by_depth[:args.top_n]:
        _add(item)
    for item in by_throughput[:args.top_n]:
        _add(item)

    _section(f'EMPIRICAL BENCHMARK — {len(bench_set)} configs')
    print(_HDR + f"  {'sps':>8}")
    print(_SEP + '----------')

    results = []   # (cfg, n_total, peak_gb, sps)
    for i, (cfg, est, n_total, fwd) in enumerate(bench_set):
        tag = (f"D={cfg.embed_dim} H={cfg.n_heads} L={cfg.n_layers} "
               f"F={cfg.ffn_dim} pL={cfg.pred_layers} pF={cfg.pred_ffn_dim} "
               f"B={cfg.batch_size}")
        print(f"  [{i+1:2d}/{len(bench_set)}] {tag}  est={est:.2f}GB  ...",
              end='', flush=True)
        sps, peak_gb = benchmark(cfg, n_features, args.patch_size, args.max_len,
                                 device, amp_dtype)
        if sps > 0:
            print(f"\r  [{i+1:2d}/{len(bench_set)}] "
                  + _fmt(cfg, n_total, peak_gb, total_vram, f"  {sps:>8.0f}"))
            results.append((cfg, n_total, peak_gb, sps))
        else:
            print(f"\r  [{i+1:2d}/{len(bench_set)}] {tag}  OOM")

    if not results:
        print('\nAll benchmarked configs ran out of memory. Lower --vram_frac.')
        return

    _section('BENCHMARK RESULTS — sorted by throughput (samples/sec)')
    print(_HDR + f"  {'sps':>8}")
    print(_SEP + '----------')
    for cfg, n_total, peak_gb, sps in sorted(results, key=lambda x: x[3], reverse=True):
        print(_fmt(cfg, n_total, peak_gb, total_vram, f"  {sps:>8.0f}"))

    # Pareto frontier: maximise (n_params, sps)
    pts    = [(float(n), s) for _, n, _, s in results]
    pareto = set(pareto_front(pts))

    best_tp  = max(results, key=lambda x: x[3])
    best_cap = max(results, key=lambda x: x[1])
    # Depth-balanced: same scoring as analytical tier, using actual measured VRAM
    best_depth = max(
        results,
        key=lambda x: x[1] * (min(1.0, x[0].n_layers / max(1, x[0].embed_dim / 128)) ** 2)
    )
    best_balanced = max(
        results,
        key=lambda x: math.sqrt(x[1] / 1e6) * math.sqrt(x[3])
    )

    _section('PARETO-OPTIMAL CONFIGS  (no config improves both params and throughput)')
    print(_HDR + f"  {'sps':>8}")
    print(_SEP + '----------')
    pareto_items = [results[i] for i in pareto]
    for cfg, n_total, peak_gb, sps in sorted(pareto_items, key=lambda x: x[3], reverse=True):
        print(_fmt(cfg, n_total, peak_gb, total_vram, f"  {sps:>8.0f}"))

    _section('RECOMMENDED COMMANDS')
    shown: set[tuple] = set()

    def _show(label: str, item: tuple) -> None:
        cfg, n_total, peak_gb, sps = item
        key = (cfg.embed_dim, cfg.n_heads, cfg.n_layers, cfg.ffn_dim,
               cfg.pred_layers, cfg.pred_ffn_dim, cfg.batch_size)
        if key in shown:
            return
        shown.add(key)
        print(f'{label}')
        print(_cli(cfg))
        print(f'    → {n_total/1e6:.2f}M params  {sps:.0f} samples/s  '
              f'{peak_gb:.2f} GB VRAM ({100*peak_gb/total_vram:.0f}%)  '
              f'({_params_per_token_note(n_total, n_patches, n_features, args.patch_size)})')

    _show('[1] Max capacity (most params):', best_cap)
    _show('\n[2] Depth-balanced (recommended — n_layers proportional to embed_dim):', best_depth)
    _show('\n[3] Balanced (√params × √throughput):', best_balanced)
    _show('\n[4] Max throughput (fastest samples/sec — good for ablations):', best_tp)


if __name__ == '__main__':
    main()
