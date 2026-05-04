"""
Pose-JEPA: Patch-level Joint-Embedding Predictive Architecture for VR motion.

Improvements over generic TS-JEPA (vr_encoder_tsjepa.py)
---------------------------------------------------------
1. Temporal patching: sequences are split into non-overlapping patch blocks
   (default: 8 frames ≈ 267 ms at 30 Hz).  Each token represents a motion
   chunk, not a single frame, forcing the model to encode movement dynamics.

2. Three target modes:
     masked_span — context surrounds missing target spans (BERT-style).
     future      — context is earlier motion, target is later motion.
     mixed       — each sample in the batch independently picks one of the two.

3. Per-sample target masks: each sequence draws its own target patches, giving
   diverse prediction tasks per optimizer step.  Context/target are gathered
   per-sample with torch.gather.

4. Context/target feature separation: feat_mask_cols and context_device_cols
   are zeroed only in the context view.  The target encoder always sees clean
   features, turning feature masking into a true cross-modal prediction task.

5. Explicit symmetric latent normalization: both predicted and target latents
   are layer-normed at loss time before Smooth-L1 or cosine loss.

6. Configurable inference pooling: cls | mean | mean_std | last.

7. Collapse diagnostics: latent_diagnostics() utility function.

Architecture
------------
  context_encoder  Encodes gathered context patches (student, optimised via grad).
  target_encoder   EMA copy of context_encoder (no gradient).
  predictor        Small Transformer: (context latents, target positions) →
                   predicted target latents.

Usage
-----
  model = PoseJEPA(...)
  loss, diag = model.jepa_loss(x, lengths)
  model.update_target_encoder(ema_decay)      # call after each optimizer step
  emb = model.embed(x, padding_mask)          # inference → (B, D) or (B, 2D)
"""

import copy
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from flash_transformer import build_transformer_encoder
from features import build_feature_cols

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

KINEMATICS   = "P"
FEATURE_COLS = build_feature_cols(KINEMATICS)
N_FEATURES   = len(FEATURE_COLS)   # 21 for "P"

MAX_LEN    = 128
PATCH_SIZE = 8                     # frames per patch  (8 × 1/30 s ≈ 267 ms)

EMBED_DIM  = 128
N_HEADS    = 4
N_LAYERS   = 4
FFN_DIM    = 512
DROPOUT    = 0.1

PRED_LAYERS  = 2
PRED_FFN_DIM = 256

TARGET_RATIO    = 0.25
N_TARGET_BLOCKS = 2
EMA_DECAY_START = 0.996
EMA_DECAY_END   = 1.0

TARGET_MODE    = "mixed"           # "masked_span" | "future" | "mixed"
FUTURE_MIN_GAP = 4                 # minimum gap (in patches) between ctx end and tgt start
FUTURE_HORIZON = (2, 8)            # (min, max) target patches for future mode

EMBED_POOL   = "mean"              # "cls" | "mean" | "mean_std" | "last"
LATENT_LOSS  = "smooth_l1"        # "smooth_l1" | "cosine"


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

class SinusoidalPosEnc(nn.Module):
    """Pre-computed sinusoidal PE table; lookup by (possibly 2-D) index tensor."""

    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)   # (max_len, d_model)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        indices: (...,) int64 → (..., d_model)
        Supports both 1-D shared indices and 2-D per-sample indices (B, N).
        """
        return self.pe[indices]


# ---------------------------------------------------------------------------
# Patch embedding
# ---------------------------------------------------------------------------

class PatchEmbedding(nn.Module):
    """Linear projection of a raw patch (patch_size × n_features) → D."""

    def __init__(self, n_features: int, patch_size: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size * n_features, embed_dim)

    def forward(self, raw_patches: torch.Tensor) -> torch.Tensor:
        """raw_patches: (B, N, patch_size * F) → (B, N, D)"""
        return self.proj(raw_patches)


# ---------------------------------------------------------------------------
# Context / target encoder
# ---------------------------------------------------------------------------

class PoseContextEncoder(nn.Module):
    """
    Encodes a per-sample subset of motion patches.

    Input raw_patches (B, N_ctx, patch_size × F) are the already-gathered
    patches for each sample.  patch_idx (B, N_ctx) provides original patch
    positions so that positional encodings reflect true temporal location.

    CLS position is reserved at PE index 0; patch indices are shifted by +1.
    """

    def __init__(
        self,
        n_features:  int,
        patch_size:  int,
        embed_dim:   int,
        n_heads:     int,
        n_layers:    int,
        ffn_dim:     int,
        dropout:     float,
        max_patches: int,
    ):
        super().__init__()
        self.patch_emb = PatchEmbedding(n_features, patch_size, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_enc   = SinusoidalPosEnc(embed_dim, max_patches + 1)  # +1 for CLS slot

        self.transformer = build_transformer_encoder(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=ffn_dim,
            dropout=dropout, num_layers=n_layers,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        raw_patches:  torch.Tensor,              # (B, N_ctx, patch_size × F)
        patch_idx:    torch.Tensor,              # (B, N_ctx) original patch positions
        padding_mask: torch.Tensor | None = None,  # (B, N_ctx) True = padding
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        cls_emb   : (B, D)       — CLS token embedding
        token_out : (B, N_ctx, D) — per-patch latent representations
        """
        B = raw_patches.shape[0]
        tokens = self.patch_emb(raw_patches)                       # (B, N_ctx, D)
        tokens = tokens + self.pos_enc(patch_idx + 1)              # per-sample PE: (B, N_ctx, D)

        cls    = self.cls_token.expand(B, -1, -1)                  # (B, 1, D)
        tokens = torch.cat([cls, tokens], dim=1)                   # (B, 1+N_ctx, D)

        if padding_mask is not None:
            cls_pad   = torch.zeros(B, 1, dtype=torch.bool, device=raw_patches.device)
            full_mask = torch.cat([cls_pad, padding_mask], dim=1)  # (B, 1+N_ctx)
        else:
            full_mask = None

        out = self.transformer(tokens, src_key_padding_mask=full_mask)
        out = self.norm(out)

        cls_emb   = out[:, 0]     # (B, D)
        token_out = out[:, 1:]    # (B, N_ctx, D)
        return cls_emb, token_out


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

class PosePredictor(nn.Module):
    """
    Small Transformer: maps (context latents at known positions, target positions)
    → predicted target latents.  Mirrors the I-JEPA predictor design.
    """

    def __init__(
        self,
        embed_dim:   int,
        n_heads:     int,
        n_layers:    int,
        ffn_dim:     int,
        dropout:     float,
        max_patches: int,
    ):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_enc    = SinusoidalPosEnc(embed_dim, max_patches + 1)
        self.transformer = build_transformer_encoder(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=ffn_dim,
            dropout=dropout, num_layers=n_layers,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        ctx_latents:  torch.Tensor,               # (B, N_ctx, D)
        ctx_idx:      torch.Tensor,               # (B, N_ctx) original patch positions
        tgt_idx:      torch.Tensor,               # (B, K) target patch positions
        ctx_pad_mask: torch.Tensor | None = None, # (B, N_ctx) True = padding
    ) -> torch.Tensor:                             # (B, K, D)
        B, N_ctx, D = ctx_latents.shape
        K = tgt_idx.shape[1]

        # Re-add positional encodings to give predictor explicit position signals
        ctx_tokens = ctx_latents + self.pos_enc(ctx_idx + 1)   # (B, N_ctx, D)

        # Learnable mask tokens at target positions
        tgt_queries = self.mask_token.expand(B, K, -1).clone()
        tgt_queries = tgt_queries + self.pos_enc(tgt_idx + 1)  # (B, K, D)

        combined = torch.cat([ctx_tokens, tgt_queries], dim=1) # (B, N_ctx+K, D)

        if ctx_pad_mask is not None:
            tgt_no_mask = torch.zeros(B, K, dtype=torch.bool, device=ctx_latents.device)
            full_mask   = torch.cat([ctx_pad_mask, tgt_no_mask], dim=1)
        else:
            full_mask = None

        out = self.transformer(combined, src_key_padding_mask=full_mask)
        out = self.norm(out)
        return out[:, N_ctx:]   # (B, K, D)


# ---------------------------------------------------------------------------
# Collapse diagnostics
# ---------------------------------------------------------------------------

@torch.no_grad()
def latent_diagnostics(z: torch.Tensor) -> dict[str, float]:
    """
    Compute quick collapse indicators for a set of latent vectors.
    z: (N, D) — flattened token/sample latents.
    Returns a dict with std_mean, std_min, collapse_frac, norm_mean.
    """
    if z.numel() == 0:
        return {'std_mean': 0.0, 'std_min': 0.0, 'collapse_frac': 1.0, 'norm_mean': 0.0}
    z   = z.reshape(-1, z.shape[-1]).float()
    std = z.std(dim=0)
    return {
        'std_mean':     std.mean().item(),
        'std_min':      std.min().item(),
        'collapse_frac': (std < 1e-4).float().mean().item(),
        'norm_mean':    z.norm(dim=-1).mean().item(),
    }


# ---------------------------------------------------------------------------
# PoseJEPA — full model
# ---------------------------------------------------------------------------

class PoseJEPA(nn.Module):
    """
    Pose-JEPA model bundling context_encoder, target_encoder (EMA), and predictor.

    Training
    --------
      loss, diag = model.jepa_loss(x, lengths, ...)
      optimizer.zero_grad(); loss.backward(); optimizer.step()
      model.update_target_encoder(ema_decay)   # after every optimizer step

    Inference
    ---------
      emb = model.embed(x, padding_mask, pool='mean')   # (B, D) or (B, 2D)
    """

    def __init__(
        self,
        n_features:   int   = N_FEATURES,
        patch_size:   int   = PATCH_SIZE,
        embed_dim:    int   = EMBED_DIM,
        n_heads:      int   = N_HEADS,
        n_layers:     int   = N_LAYERS,
        ffn_dim:      int   = FFN_DIM,
        dropout:      float = DROPOUT,
        max_len:      int   = MAX_LEN,
        pred_layers:  int   = PRED_LAYERS,
        pred_ffn_dim: int   = PRED_FFN_DIM,
        embed_pool:   str   = EMBED_POOL,
    ):
        super().__init__()
        if max_len % patch_size != 0:
            raise ValueError(
                f"max_len ({max_len}) must be divisible by patch_size ({patch_size})"
            )
        self.patch_size  = patch_size
        self.max_patches = max_len // patch_size
        self.embed_pool  = embed_pool

        enc_kw = dict(
            n_features=n_features, patch_size=patch_size, embed_dim=embed_dim,
            n_heads=n_heads, n_layers=n_layers, ffn_dim=ffn_dim,
            dropout=dropout, max_patches=self.max_patches,
        )
        self.context_encoder = PoseContextEncoder(**enc_kw)

        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

        self.predictor = PosePredictor(
            embed_dim=embed_dim, n_heads=n_heads, n_layers=pred_layers,
            ffn_dim=pred_ffn_dim, dropout=dropout, max_patches=self.max_patches,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _patchify_gather(
        self,
        x:         torch.Tensor,   # (B, T, F)
        patch_idx: torch.Tensor,   # (B, K)
    ) -> torch.Tensor:             # (B, K, patch_size * F)
        """Reshape x into patches then gather the requested patch indices."""
        B, T, F = x.shape
        P = self.patch_size
        N = T // P
        x_patches = x[:, :N * P].reshape(B, N, P * F)
        idx_exp   = patch_idx.unsqueeze(-1).expand(-1, -1, P * F)
        return x_patches.gather(1, idx_exp)

    @staticmethod
    def _sample_masks(
        N:               int,
        K:               int,
        B:               int,
        lengths:         torch.Tensor,
        patch_size:      int,
        device:          torch.device,
        target_mode:     str,
        n_target_blocks: int,
        future_min_gap:  int,
        future_horizon:  tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """
        Sample per-sample context / target patch indices.

        Returns
        -------
        tgt_idx  : (B, K)    — target patch indices per sample
        ctx_idx  : (B, N-K)  — context patch indices per sample
        n_future : int        — number of samples that used future mode
                                (diagnostic; actual count may be < B/2 when future
                                falls back to masked_span on short sequences)
        Note: only full patches are considered valid (n_valid = lengths // patch_size).
        """
        K_ctx    = N - K
        n_valid  = (lengths // patch_size).clamp(min=K + 1, max=N).tolist()
        n_future = 0

        tgt_rows: list[torch.Tensor] = []
        ctx_rows: list[torch.Tensor] = []

        for b in range(B):
            nv   = int(n_valid[b])
            mode = target_mode
            if mode == "mixed":
                mode = "masked_span" if random.random() < 0.5 else "future"

            tgt_set: set[int] = set()
            used_future = False

            if mode == "future":
                lo    = future_horizon[0]
                hi    = future_horizon[1]
                max_H = min(hi, K, nv - future_min_gap - 1)
                if max_H >= lo:
                    H       = random.randint(lo, max_H)
                    max_spl = nv - future_min_gap - H
                    if max_spl >= 1:
                        split     = random.randint(1, max_spl)
                        tgt_start = split + future_min_gap
                        tgt_set   = set(range(tgt_start, min(tgt_start + K, nv)))
                        used_future = True

            # Fall through to masked_span when future didn't produce targets
            if not used_future:
                tgt_set   = set()
                block_len = max(1, round(nv * (K / N)) // max(1, n_target_blocks))
                for _ in range(n_target_blocks):
                    max_s = max(0, nv - block_len)
                    start = random.randint(0, max_s)
                    tgt_set.update(range(start, min(start + block_len, nv)))

            if used_future:
                n_future += 1

            # Trim / pad to exactly K (use random.sample to avoid early-index bias)
            tgt_list = sorted(tgt_set)
            if len(tgt_list) > K:
                tgt_list = sorted(random.sample(tgt_list, K))
            else:
                pool = [i for i in range(nv) if i not in set(tgt_list)]
                random.shuffle(pool)
                while len(tgt_list) < K:
                    tgt_list.append(pool.pop() if pool else tgt_list[-1])
                tgt_list = sorted(tgt_list)

            tgt_set_f = set(tgt_list)
            ctx_list  = [i for i in range(N) if i not in tgt_set_f][:K_ctx]
            while len(ctx_list) < K_ctx:
                ctx_list.append(0)   # over-padding guard; masked out below

            tgt_rows.append(torch.tensor(tgt_list, dtype=torch.long))
            ctx_rows.append(torch.tensor(ctx_list, dtype=torch.long))

        tgt_idx = torch.stack(tgt_rows).to(device)   # (B, K)
        ctx_idx = torch.stack(ctx_rows).to(device)   # (B, N-K)
        return tgt_idx, ctx_idx, n_future

    # ------------------------------------------------------------------
    # EMA update
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update_target_encoder(self, ema_decay: float) -> None:
        """θ_target ← ema_decay · θ_target + (1 − ema_decay) · θ_context"""
        for p_s, p_t in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters(),
        ):
            p_t.data.mul_(ema_decay).add_(p_s.data, alpha=1.0 - ema_decay)

    # ------------------------------------------------------------------
    # Training loss
    # ------------------------------------------------------------------

    def jepa_loss(
        self,
        x:               torch.Tensor,               # (B, T, F)  normalised
        lengths:         torch.Tensor,               # (B,)       actual frame lengths
        target_ratio:    float             = TARGET_RATIO,
        n_target_blocks: int               = N_TARGET_BLOCKS,
        target_mode:     str               = TARGET_MODE,
        future_min_gap:  int               = FUTURE_MIN_GAP,
        future_horizon:  tuple[int, int]   = FUTURE_HORIZON,
        feat_mask_cols:  list[int] | None  = None,
        context_device_cols: list[int] | None = None,
        latent_loss:     str               = LATENT_LOSS,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute Pose-JEPA loss.

        feat_mask_cols and context_device_cols are zeroed only in the context
        branch; the target encoder always receives clean features.

        Returns
        -------
        loss : scalar Tensor
        diag : dict of collapse diagnostics for ctx / tgt / pred latents
        """
        B, T, _ = x.shape
        device  = x.device
        P       = self.patch_size
        N       = T // P

        if N < 2:
            return x.new_zeros(()).requires_grad_(True), {}

        K = max(1, min(round(N * target_ratio), N - 1))

        # ---- per-sample masks ----------------------------------------
        tgt_idx, ctx_idx, n_future = self._sample_masks(
            N, K, B, lengths, P, device,
            target_mode, n_target_blocks, future_min_gap, future_horizon,
        )

        # ---- feature separation: context corrupted, target clean ------
        x_context = x
        needs_clone = bool(feat_mask_cols or context_device_cols)
        if needs_clone:
            x_context = x.clone()
            if feat_mask_cols:
                col_t = torch.tensor(feat_mask_cols, dtype=torch.long, device=device)
                x_context[:, :, col_t] = 0.0
            if context_device_cols:
                col_t = torch.tensor(context_device_cols, dtype=torch.long, device=device)
                x_context[:, :, col_t] = 0.0
        x_target = x   # always clean

        # ---- gather context / target patches --------------------------
        ctx_raw = self._patchify_gather(x_context, ctx_idx)   # (B, N-K, P*F)
        tgt_raw = self._patchify_gather(x_target,  tgt_idx)   # (B, K,   P*F)

        # ---- patch-level padding masks --------------------------------
        # Patch p covers frames [p*P, (p+1)*P); valid if p < lengths // P
        n_valid_patches = lengths // P    # (B,)
        ctx_pad_mask = ctx_idx >= n_valid_patches.unsqueeze(1)  # (B, N-K)
        tgt_pad_mask = tgt_idx >= n_valid_patches.unsqueeze(1)  # (B, K)

        # Guard: all samples need ≥1 valid context and ≥1 valid target
        if not (~ctx_pad_mask).any(dim=1).all() or \
           not (~tgt_pad_mask).any(dim=1).all():
            return x.new_zeros(()).requires_grad_(True), {}

        # ---- context encoder ----------------------------------------
        _, ctx_latents = self.context_encoder(ctx_raw, ctx_idx, ctx_pad_mask)
        # (B, N-K, D)

        # ---- target encoder (EMA, no gradient, clean input) ----------
        with torch.no_grad():
            _, tgt_latents = self.target_encoder(tgt_raw, tgt_idx, tgt_pad_mask)
            tgt_latents = F.layer_norm(tgt_latents, [tgt_latents.shape[-1]])

        # ---- predictor -----------------------------------------------
        pred_latents = self.predictor(ctx_latents, ctx_idx, tgt_idx, ctx_pad_mask)
        # (B, K, D)

        # Symmetric normalization at loss time
        pred_latents = F.layer_norm(pred_latents, [pred_latents.shape[-1]])

        # ---- loss on valid target positions --------------------------
        valid_tgt = ~tgt_pad_mask   # (B, K)
        if valid_tgt.sum() == 0:
            return x.new_zeros(()).requires_grad_(True), {}

        pred_v = pred_latents[valid_tgt]
        tgt_v  = tgt_latents[valid_tgt].detach()

        if latent_loss == "cosine":
            loss = (2 - 2 * (F.normalize(pred_v, dim=-1) *
                             F.normalize(tgt_v,  dim=-1)).sum(dim=-1)).mean()
        else:
            loss = F.smooth_l1_loss(pred_v, tgt_v)

        # ---- diagnostics (no grad) -----------------------------------
        with torch.no_grad():
            diag = {
                'ctx':              latent_diagnostics(ctx_latents[~ctx_pad_mask]),
                'tgt':              latent_diagnostics(tgt_v),
                'pred':             latent_diagnostics(pred_v),
                'future_frac':      n_future / B,
            }

        return loss, diag

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def embed(
        self,
        x:            torch.Tensor,                  # (B, T, F)
        padding_mask: torch.Tensor | None = None,    # (B, T) frame-level True=pad
        pool:         str | None          = None,
    ) -> torch.Tensor:
        """
        Encode the full sequence and return a pooled embedding.

        pool choices
        ------------
        cls       — CLS token                        → (B, D)
        mean      — mean of valid patch tokens        → (B, D)
        mean_std  — mean concatenated with std        → (B, 2D)
        last      — last valid patch token            → (B, D)
        """
        pool = pool or self.embed_pool
        B, T, F = x.shape
        P = self.patch_size
        N = T // P

        # Patchify full sequence
        x_patches = x[:, :N * P].reshape(B, N, P * F)  # (B, N, P*F)

        # Patch-level padding mask from frame-level mask
        if padding_mask is not None:
            # Patch p is padding if its first frame is padding
            patch_pad = padding_mask[:, torch.arange(N, device=x.device) * P]  # (B, N)
        else:
            patch_pad = None

        # Shared patch indices (same for all samples at inference)
        all_idx = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)  # (B, N)

        cls_emb, token_out = self.context_encoder(x_patches, all_idx, patch_pad)
        # cls_emb: (B, D), token_out: (B, N, D)

        if pool == "cls":
            return cls_emb

        if patch_pad is not None:
            valid = (~patch_pad).unsqueeze(-1).float()   # (B, N, 1)
            n_val = valid.sum(dim=1).clamp(min=1)        # (B, 1)
        else:
            valid = torch.ones(B, N, 1, device=x.device)
            n_val = torch.full((B, 1), N, device=x.device, dtype=torch.float)

        if pool == "mean":
            return (token_out * valid).sum(dim=1) / n_val

        if pool == "mean_std":
            mean = (token_out * valid).sum(dim=1) / n_val              # (B, D)
            var  = ((token_out - mean.unsqueeze(1)) ** 2 * valid).sum(dim=1) / n_val
            return torch.cat([mean, var.sqrt()], dim=-1)               # (B, 2D)

        if pool == "last":
            if patch_pad is not None:
                valid_b      = ~patch_pad                               # (B, N)
                valid_counts = valid_b.long().sum(dim=1).clamp(min=1)  # (B,)
                last         = valid_counts - 1                         # (B,)
                return token_out[torch.arange(B, device=x.device), last]
            return token_out[:, -1]

        return cls_emb   # fallback
