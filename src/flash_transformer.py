"""
FlashAttention-backed transformer encoder.

Provides build_transformer_encoder() — returns FlashTransformerEncoder when
flash-attn >= 2 is installed, otherwise falls back to nn.TransformerEncoder.

FlashTransformerEncoder subclasses both nn.TransformerEncoder and
nn.TransformerEncoderLayer, preserving identical parameter names and shapes so
that checkpoints are interchangeable between the flash and fallback backends.

The varlen API packs sequences by removing padding before attention and
unpacks afterward, so padding tokens never participate in attention computation.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_varlen_func
    _FLASH_AVAILABLE = True
except ImportError:
    _FLASH_AVAILABLE = False


class FlashTransformerEncoderLayer(nn.TransformerEncoderLayer):
    """
    Subclass of nn.TransformerEncoderLayer that replaces SDPA with
    flash_attn_varlen_func.  Weights are identical in name and shape to the
    parent class so checkpoints are interchangeable.

    Use forward_packed() rather than forward() — input must already be packed
    (padding removed) by the enclosing FlashTransformerEncoder.
    """

    def forward(self, src, cu_seqlens=None, max_seqlen=None, **kwargs):
        if cu_seqlens is not None:
            return self.forward_packed(src, cu_seqlens, max_seqlen)
        return super().forward(src, **kwargs)

    def forward_packed(
        self,
        src:        torch.Tensor,   # (total_valid, D)
        cu_seqlens: torch.Tensor,   # (B+1,) int32 — cumulative valid lengths
        max_seqlen: int,
    ) -> torch.Tensor:              # (total_valid, D)
        total    = src.shape[0]
        nhead    = self.self_attn.num_heads
        head_dim = self.self_attn.embed_dim // nhead

        # Pre-norm attention block
        residual = src
        normed   = self.norm1(src)
        qkv = F.linear(normed,
                        self.self_attn.in_proj_weight,
                        self.self_attn.in_proj_bias)         # (total, 3*D)
        q, k, v = qkv.reshape(total, 3, nhead, head_dim).unbind(dim=1)

        attn_out = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            dropout_p=self.self_attn.dropout if self.training else 0.0,
            causal=False,
        )                                                     # (total, nhead, head_dim)
        attn_out = F.linear(
            attn_out.reshape(total, self.self_attn.embed_dim),
            self.self_attn.out_proj.weight,
            self.self_attn.out_proj.bias,
        )
        x = residual + self.dropout1(attn_out)

        # Pre-norm FFN block — reuses parent weights (linear1, linear2, norm2, dropout*)
        x = x + self.dropout2(
            self.linear2(self.dropout(self.activation(self.linear1(self.norm2(x)))))
        )
        return x


class FlashTransformerEncoder(nn.TransformerEncoder):
    """
    Subclass of nn.TransformerEncoder that packs sequences before attention and
    unpacks afterward, enabling flash_attn_varlen_func.

    Interface matches nn.TransformerEncoder: pass (src, src_key_padding_mask).
    """

    def forward(
        self,
        src:                  torch.Tensor,            # (B, T, D)
        mask:                 torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        is_causal:            bool | None = None,
    ) -> torch.Tensor:                                 # (B, T, D)
        B, T, D  = src.shape
        device   = src.device

        valid = (~src_key_padding_mask
                 if src_key_padding_mask is not None
                 else src.new_ones(B, T, dtype=torch.bool))   # (B, T) True=valid

        lengths    = valid.sum(dim=1).to(torch.int32)
        cu_seqlens = torch.zeros(B + 1, dtype=torch.int32, device=device)
        cu_seqlens[1:] = lengths.cumsum(0)
        max_seqlen = int(lengths.max().item())

        x = src[valid]                                        # (total_valid, D)
        for layer in self.layers:
            x = layer(x, cu_seqlens, max_seqlen)
        if self.norm is not None:
            x = self.norm(x)

        out = torch.zeros_like(src)
        out[valid] = x
        return out


def build_transformer_encoder(
    d_model:         int,
    nhead:           int,
    dim_feedforward: int,
    dropout:         float,
    num_layers:      int,
) -> nn.Module:
    """
    Returns FlashTransformerEncoder if flash-attn >= 2 is installed,
    otherwise falls back to nn.TransformerEncoder.
    Both use batch_first=True and norm_first=True (pre-norm).
    """
    layer_cls   = FlashTransformerEncoderLayer if _FLASH_AVAILABLE else nn.TransformerEncoderLayer
    encoder_cls = FlashTransformerEncoder      if _FLASH_AVAILABLE else nn.TransformerEncoder

    layer = layer_cls(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        batch_first=True,
        norm_first=True,
    )
    return encoder_cls(layer, num_layers=num_layers, enable_nested_tensor=False)
