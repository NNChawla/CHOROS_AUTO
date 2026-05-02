"""
VR motion encoder: small Transformer trained with masked autoencoder (MAE) objective.

Features are selected via the KINEMATICS string (see src/features.py):
  P — position/orientation  (default, 21 features)
  V — velocity              (+18 features)
  A — acceleration          (+18 features)
  J — jerk                  (+18 features)
Eye gaze columns are 100% NaN across the dataset and are excluded.
"""

import math
import torch
import torch.nn as nn

from features import build_feature_cols
from flash_transformer import build_transformer_encoder
from masking import random_timestep_mask, span_mask

KINEMATICS   = "P"                   # change here or pass --kinematics to the training script
FEATURE_COLS = build_feature_cols(KINEMATICS)
N_FEATURES   = len(FEATURE_COLS)     # 21 for "P"

# Training hyperparameters
MAX_LEN    = 128    # timesteps per window (shorter seqs padded, longer seqs cropped/windowed)
MASK_RATIO = 0.30   # fraction of valid timesteps masked during training
EMBED_DIM  = 128    # transformer model dimension
N_HEADS    = 4
N_LAYERS   = 4
FFN_DIM    = 512
DROPOUT    = 0.1


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = MAX_LEN + 1):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class VREncoder(nn.Module):
    """
    Transformer encoder for VR motion sequences.

    Self-supervised training: randomly mask MASK_RATIO of valid timesteps in the
    input, reconstruct the original values, and compute MSE only on masked positions.
    The CLS token output is used as the sequence embedding at inference time.
    """

    def __init__(
        self,
        n_features: int  = N_FEATURES,
        embed_dim:  int  = EMBED_DIM,
        n_heads:    int  = N_HEADS,
        n_layers:   int  = N_LAYERS,
        ffn_dim:    int  = FFN_DIM,
        dropout:    float = DROPOUT,
        max_len:    int  = MAX_LEN,
    ):
        super().__init__()
        self.embed_dim  = embed_dim
        self.n_features = n_features

        self.input_proj = nn.Linear(n_features, embed_dim)
        self.cls_token  = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_enc    = PositionalEncoding(embed_dim, max_len + 1)

        self.transformer = build_transformer_encoder(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=ffn_dim,
            dropout=dropout, num_layers=n_layers,
        )
        self.norm        = nn.LayerNorm(embed_dim)
        self.recon_head  = nn.Linear(embed_dim, n_features)

    # ------------------------------------------------------------------
    # Core forward (used both in training and for direct embedding calls)
    # ------------------------------------------------------------------

    def forward(
        self,
        x:            torch.Tensor,               # (B, T, F)
        padding_mask: torch.Tensor | None = None,  # (B, T) bool, True = padding
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        cls_emb : (B, embed_dim)  — CLS token embedding
        recon   : (B, T, F)       — reconstructed features at every timestep
        """
        B = x.size(0)
        tokens = self.input_proj(x)                         # (B, T, embed_dim)
        cls    = self.cls_token.expand(B, -1, -1)           # (B, 1, embed_dim)
        tokens = torch.cat([cls, tokens], dim=1)            # (B, T+1, embed_dim)
        tokens = self.pos_enc(tokens)

        if padding_mask is not None:
            cls_pad   = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
            full_mask = torch.cat([cls_pad, padding_mask], dim=1)  # (B, T+1)
        else:
            full_mask = None

        out = self.transformer(tokens, src_key_padding_mask=full_mask)
        out = self.norm(out)

        cls_emb = out[:, 0]          # (B, embed_dim)
        recon   = self.recon_head(out[:, 1:])  # (B, T, F)
        return cls_emb, recon

    # ------------------------------------------------------------------
    # Masked autoencoder training step
    # ------------------------------------------------------------------

    def mae_loss(
        self,
        x:              torch.Tensor,            # (B, T, F)  normalised, NaNs filled
        lengths:        torch.Tensor,            # (B,)        actual sequence lengths
        mask_ratio:     float          = MASK_RATIO,
        mask_type:      str            = "random",   # "random" | "span"
        n_span_blocks:  int            = 4,           # span blocks (mask_type="span")
        feat_mask_cols: list[int] | None = None,      # feature columns zeroed globally
    ) -> torch.Tensor:
        """
        Mask *mask_ratio* of valid timesteps per sequence, reconstruct them, and
        return mean MSE over masked positions against the original clean input.

        mask_type
            "random"  — uniform random timestep selection (default)
            "span"    — n_span_blocks contiguous spans, each ≈ mask_ratio/n_blocks long

        feat_mask_cols
            Optional list of feature column indices to zero out across ALL timesteps
            before temporal masking.  The reconstruction target is always the original
            clean input, so the model must infer missing sensor/kinematic channels from
            the remaining context.  Compute this list with masking.feat_col_indices().
        """
        B, T, F = x.shape
        device  = x.device

        arange       = torch.arange(T, device=device).unsqueeze(0)  # (1, T)
        padding_mask = arange >= lengths.unsqueeze(1)                # (B, T) bool

        # Build temporal mask
        if mask_type == "span":
            tmask = span_mask(lengths, T, mask_ratio, n_span_blocks, device)
        else:
            tmask = random_timestep_mask(lengths, T, mask_ratio, device)

        # Build corrupted input (feature mask + temporal mask)
        x_in = x.clone()
        if feat_mask_cols:
            col_idx = torch.tensor(feat_mask_cols, dtype=torch.long, device=device)
            x_in[:, :, col_idx] = 0.0   # global feature dropout across all timesteps
        x_in[tmask] = 0.0               # zero remaining features at masked positions

        _, recon = self.forward(x_in, padding_mask)   # (B, T, F)

        # Loss on masked non-padding positions against the original clean input
        loss_mask = tmask & ~padding_mask
        if loss_mask.sum() == 0:
            return x.new_zeros(1).squeeze()

        return nn.functional.mse_loss(recon[loss_mask], x[loss_mask])

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def embed(
        self,
        x:            torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return CLS-token embedding, no gradient."""
        cls_emb, _ = self.forward(x, padding_mask)
        return cls_emb
