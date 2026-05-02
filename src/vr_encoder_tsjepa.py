"""
TS-JEPA: Time-Series Joint-Embedding Predictive Architecture for VR motion.

Architecture
------------
  context_encoder  Transformer (student).  Encodes only the context positions.
                   Uses original sinusoidal positions so the predictor knows *where*
                   each token came from.  CLS token → sequence embedding at inference.

  target_encoder   EMA copy of context_encoder (no gradient, not directly trained).
                   Encodes only the target positions → produces latent targets.

  predictor        Small Transformer.  Takes context latents (re-positionally-encoded)
                   plus learnable mask tokens at target positions, and predicts the
                   target encoder's latent representations.

Loss
----
  Smooth-L1 between predictor output and (layer-normed) target encoder output,
  computed only on valid (non-padding) target positions.

Collapse prevention
-------------------
  The EMA stop-gradient on the target encoder, combined with the predictor having
  to infer target representations from a *different* (context) view of the sequence,
  naturally prevents representational collapse — no contrastive or variance terms needed.

Usage
-----
  model = TSJEPA(...)
  loss  = model.jepa_loss(x, lengths)          # training
  emb   = model.embed(x, padding_mask)         # inference → CLS embedding
  model.update_target_encoder(ema_decay)       # call after every optimizer step
"""

import copy
import math
import torch
import torch.nn as nn

from flash_transformer import build_transformer_encoder

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

from features import build_feature_cols
from masking import random_timestep_mask

KINEMATICS   = "P"                   # change here or pass --kinematics to the training script
FEATURE_COLS = build_feature_cols(KINEMATICS)
N_FEATURES   = len(FEATURE_COLS)     # 21 for "P"

# Encoder architecture
MAX_LEN    = 128
EMBED_DIM  = 128
N_HEADS    = 4
N_LAYERS   = 4
FFN_DIM    = 512
DROPOUT    = 0.1

# Predictor architecture (kept intentionally smaller than the encoder)
PRED_LAYERS  = 2
PRED_FFN_DIM = 256

# TS-JEPA training parameters
TARGET_RATIO    = 0.25   # fraction of valid timesteps masked as prediction targets
N_TARGET_BLOCKS = 2      # number of contiguous target blocks per sample
EMA_DECAY_START = 0.996  # initial EMA decay (increases to 1.0 over training)
EMA_DECAY_END   = 1.0


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

class SinusoidalPosEnc(nn.Module):
    """
    Pre-computed sinusoidal positional encoding table.
    Lookup by index: self(indices) → (len(indices), d_model).
    """

    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)          # (max_len, d_model)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """indices: (N,) int64 → (N, d_model)"""
        return self.pe[indices]


# ---------------------------------------------------------------------------
# Context encoder (student)
# ---------------------------------------------------------------------------

class ContextEncoder(nn.Module):
    """
    Encodes a *subset* of time-series tokens (the context positions).

    Each token gets the sinusoidal positional encoding for its *original* position
    in the full sequence, so the predictor can unambiguously map context latents
    to their timestamps.

    Position 0 is reserved for the CLS token; sequence positions are shifted +1.
    """

    def __init__(
        self,
        n_features: int,
        embed_dim:  int,
        n_heads:    int,
        n_layers:   int,
        ffn_dim:    int,
        dropout:    float,
        max_len:    int,
    ):
        super().__init__()
        self.embed_dim  = embed_dim
        self.input_proj = nn.Linear(n_features, embed_dim)
        self.cls_token  = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_enc    = SinusoidalPosEnc(embed_dim, max_len + 1)  # +1 for CLS slot

        self.transformer = build_transformer_encoder(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=ffn_dim,
            dropout=dropout, num_layers=n_layers,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x:            torch.Tensor,              # (B, T', F)
        positions:    torch.Tensor,              # (T',) original indices in [0, max_len)
        padding_mask: torch.Tensor | None = None,  # (B, T') True = padding
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        cls_emb   : (B, D)     — CLS token embedding (for inference)
        token_out : (B, T', D) — per-position latent representations
        """
        B = x.shape[0]
        tokens = self.input_proj(x)                                 # (B, T', D)
        tokens = tokens + self.pos_enc(positions + 1).unsqueeze(0)  # shift +1; CLS at 0

        cls    = self.cls_token.expand(B, -1, -1)                   # (B, 1, D)
        tokens = torch.cat([cls, tokens], dim=1)                    # (B, 1+T', D)

        if padding_mask is not None:
            cls_pad   = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
            full_mask = torch.cat([cls_pad, padding_mask], dim=1)   # (B, 1+T')
        else:
            full_mask = None

        out = self.transformer(tokens, src_key_padding_mask=full_mask)
        out = self.norm(out)

        cls_emb   = out[:, 0]    # (B, D)
        token_out = out[:, 1:]   # (B, T', D)
        return cls_emb, token_out


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

class Predictor(nn.Module):
    """
    Small Transformer predictor.

    Takes context latents + learnable mask tokens at target positions and
    predicts what the (EMA) target encoder would output at those positions.

    To give the predictor explicit positional signals it has its own positional
    encoding table.  Positional encodings are re-added to context latents and
    also added to the learnable mask tokens for target positions.  This mirrors
    the I-JEPA design: the predictor is a *separate* map from (context latents,
    target positions) → predicted target latents.
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads:   int,
        n_layers:  int,
        ffn_dim:   int,
        dropout:   float,
        max_len:   int,
    ):
        super().__init__()
        self.embed_dim  = embed_dim
        # Learnable mask token — common starting point for all target queries
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_enc    = SinusoidalPosEnc(embed_dim, max_len + 1)

        self.transformer = build_transformer_encoder(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=ffn_dim,
            dropout=dropout, num_layers=n_layers,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        ctx_latents:   torch.Tensor,              # (B, T_ctx, D)
        ctx_positions: torch.Tensor,              # (T_ctx,) original indices
        tgt_positions: torch.Tensor,              # (T_tgt,) original indices to predict
        ctx_pad_mask:  torch.Tensor | None = None,  # (B, T_ctx) True = padding
    ) -> torch.Tensor:
        """Returns (B, T_tgt, D) — predicted latent at each target position."""
        B      = ctx_latents.shape[0]
        T_ctx  = ctx_latents.shape[1]
        T_tgt  = len(tgt_positions)

        # Re-add positional encodings to context latents so the predictor has
        # explicit position signals on top of whatever the encoder encoded.
        ctx_tokens = ctx_latents + self.pos_enc(ctx_positions + 1).unsqueeze(0)

        # Target query tokens: learnable mask token + positional encoding
        tgt_queries  = self.mask_token.expand(B, T_tgt, -1).clone()
        tgt_queries  = tgt_queries + self.pos_enc(tgt_positions + 1).unsqueeze(0)

        # Run predictor over [context tokens | target queries].
        # Target queries are never masked; context positions inherit ctx_pad_mask so
        # that padded context tokens cannot corrupt attention via unmasked key access.
        combined = torch.cat([ctx_tokens, tgt_queries], dim=1)  # (B, T_ctx+T_tgt, D)
        if ctx_pad_mask is not None:
            tgt_no_mask = torch.zeros(B, T_tgt, dtype=torch.bool, device=ctx_latents.device)
            full_mask   = torch.cat([ctx_pad_mask, tgt_no_mask], dim=1)  # (B, T_ctx+T_tgt)
        else:
            full_mask = None
        out = self.transformer(combined, src_key_padding_mask=full_mask)
        out = self.norm(out)

        # Extract only the target-query outputs (last T_tgt positions)
        return out[:, T_ctx:, :]    # (B, T_tgt, D)


# ---------------------------------------------------------------------------
# TSJEPA — full model
# ---------------------------------------------------------------------------

class TSJEPA(nn.Module):
    """
    TS-JEPA model bundling context_encoder, target_encoder (EMA), and predictor.

    Training
    --------
      loss = model.jepa_loss(x, lengths)
      optimizer.zero_grad(); loss.backward(); optimizer.step()
      model.update_target_encoder(ema_decay)   # after every optimizer step

    Inference
    ---------
      emb = model.embed(x, padding_mask)   # (B, embed_dim) CLS embedding
    """

    def __init__(
        self,
        n_features:  int   = N_FEATURES,
        embed_dim:   int   = EMBED_DIM,
        n_heads:     int   = N_HEADS,
        n_layers:    int   = N_LAYERS,
        ffn_dim:     int   = FFN_DIM,
        dropout:     float = DROPOUT,
        max_len:     int   = MAX_LEN,
        pred_layers: int   = PRED_LAYERS,
        pred_ffn_dim: int  = PRED_FFN_DIM,
    ):
        super().__init__()
        enc_kw = dict(
            n_features=n_features, embed_dim=embed_dim,
            n_heads=n_heads, n_layers=n_layers,
            ffn_dim=ffn_dim, dropout=dropout, max_len=max_len,
        )
        self.context_encoder = ContextEncoder(**enc_kw)

        # Target encoder: EMA copy — not directly optimised
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

        self.predictor = Predictor(
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_layers=pred_layers,
            ffn_dim=pred_ffn_dim,
            dropout=dropout,
            max_len=max_len,
        )

        self.max_len = max_len

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
        x:               torch.Tensor,             # (B, T, F)  normalised, NaN → 0
        lengths:         torch.Tensor,             # (B,)       actual sequence lengths
        target_ratio:    float          = TARGET_RATIO,
        n_target_blocks: int            = N_TARGET_BLOCKS,
        mask_type:       str            = "span",  # "span" | "random"
        feat_mask_cols:  list[int] | None = None,  # feature columns zeroed globally
    ) -> torch.Tensor:
        """
        1. Sample target positions using mask_type:
             "span"   — N_TARGET_BLOCKS contiguous blocks (batch-shared mask).
             "random" — target_ratio of valid positions uniformly at random
                        (batch-shared mask based on min valid length).
        2. Optionally apply feature masking: zero feat_mask_cols across all timesteps.
        3. Encode context positions with context_encoder.
        4. Encode target positions with target_encoder (no gradient).
        5. Predict target latents from context latents via predictor.
        6. Return Smooth-L1 loss on valid (non-padding) target positions.

        Both mask types produce a batch-shared mask so context/target slicing
        remains a simple index operation over the batch dimension.

        feat_mask_cols
            Optional list of feature column indices to zero out across ALL timesteps
            before passing inputs to the encoders.  Trains cross-sensor and
            cross-kinematic-group prediction in latent space.
            Compute with masking.feat_col_indices().
        """
        B, T, F = x.shape
        device  = x.device

        min_len = max(1, int(lengths.min().item()))

        # ---- sample batch-shared target mask -------------------------
        if mask_type == "random":
            # Uniform random selection from valid positions, batch-shared
            n_tgt = min(max(1, int(min_len * target_ratio)), max(1, min_len - 1))
            perm  = torch.randperm(min_len, device=device)[:n_tgt]
            target_mask = torch.zeros(T, dtype=torch.bool, device=device)
            target_mask[perm] = True
        else:
            # Contiguous span masking (original behaviour)
            # Clamp block_len so n_target_blocks spans never cover all valid positions.
            max_block_len = max(1, (min_len - 1) // n_target_blocks) if min_len > 1 else 1
            block_len     = min(max(1, int(min_len * target_ratio)), max_block_len)
            target_mask   = torch.zeros(T, dtype=torch.bool, device=device)
            for _ in range(n_target_blocks):
                max_start = max(0, min_len - block_len)
                start = torch.randint(0, max_start + 1, (1,), device=device).item()
                target_mask[start : start + block_len] = True

        context_mask = ~target_mask

        tgt_idx = target_mask.nonzero(as_tuple=True)[0]    # (T_tgt,)
        ctx_idx = context_mask.nonzero(as_tuple=True)[0]   # (T_ctx,)

        if len(tgt_idx) == 0 or len(ctx_idx) == 0:
            return x.new_zeros(()).requires_grad_(True)

        # ---- apply feature masking -----------------------------------
        x_in = x
        if feat_mask_cols:
            x_in = x.clone()
            col_idx = torch.tensor(feat_mask_cols, dtype=torch.long, device=device)
            x_in[:, :, col_idx] = 0.0

        # ---- padding masks (True = padding / ignore) -----------------
        arange = torch.arange(T, device=device).unsqueeze(0)   # (1, T)
        valid  = arange < lengths.unsqueeze(1)                  # (B, T)

        ctx_pad_mask = ~valid[:, ctx_idx]   # (B, T_ctx)
        tgt_pad_mask = ~valid[:, tgt_idx]   # (B, T_tgt)

        # Guard: if any sequence has no valid context tokens the transformer
        # would compute softmax(-inf, …) → NaN, crashing the CUDA kernel.
        if not (~ctx_pad_mask).any(dim=1).all():
            return x.new_zeros(()).requires_grad_(True)

        # Guard: if any sequence has no valid target tokens the loss has no
        # signal and the target-encoder attention over all-padding rows may
        # produce NaN that propagates through the predictor.
        if not (~tgt_pad_mask).any(dim=1).all():
            return x.new_zeros(()).requires_grad_(True)

        # ---- context encoder ----------------------------------------
        x_ctx = x_in[:, ctx_idx, :]                               # (B, T_ctx, F)
        _, ctx_latents = self.context_encoder(x_ctx, ctx_idx, ctx_pad_mask)
        # ctx_latents: (B, T_ctx, D)

        # ---- target encoder (EMA, no gradient) ----------------------
        x_tgt = x_in[:, tgt_idx, :]                               # (B, T_tgt, F)
        with torch.no_grad():
            _, tgt_latents = self.target_encoder(x_tgt, tgt_idx, tgt_pad_mask)
            # Normalise target latents per-token to prevent trivial scale solutions
            tgt_latents = nn.functional.layer_norm(
                tgt_latents, [tgt_latents.shape[-1]]
            )

        # ---- predictor ----------------------------------------------
        pred_latents = self.predictor(ctx_latents, ctx_idx, tgt_idx, ctx_pad_mask)
        # pred_latents: (B, T_tgt, D)

        # ---- loss on valid target positions only --------------------
        valid_tgt = ~tgt_pad_mask   # (B, T_tgt)
        if valid_tgt.sum() == 0:
            return x.new_zeros(()).requires_grad_(True)

        return nn.functional.smooth_l1_loss(
            pred_latents[valid_tgt],
            tgt_latents[valid_tgt].detach(),
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def embed(
        self,
        x:            torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Encode the full sequence (no masking) and return the CLS token embedding.

        x            : (B, T, F)
        padding_mask : (B, T) bool, True = padding  [optional]

        Returns: (B, embed_dim)
        """
        T   = x.shape[1]
        dev = x.device
        all_positions = torch.arange(T, device=dev)
        cls_emb, _ = self.context_encoder(x, all_positions, padding_mask)
        return cls_emb
