"""GPU auto-detection for training default configuration."""
from __future__ import annotations
import torch

_HIGH_VRAM_GB = 20  # 3090=24GB → high; 3060=12GB → low


def _detect_attn_backend(sm_major: int) -> str:
    """
    Select the best available FlashAttention backend for the given SM major version.
      SM >= 9  (Hopper/Blackwell) → flash3 if flash-attn >= 3 installed, else flash2
      SM >= 8  (Ampere/Ada)       → flash2 if flash-attn >= 2 installed
      otherwise                   → sdpa (PyTorch built-in fallback)
    """
    try:
        import flash_attn
        fa_major = int(flash_attn.__version__.split('.')[0])
        if sm_major >= 9 and fa_major >= 3:
            return 'flash3'
        if sm_major >= 8 and fa_major >= 2:
            return 'flash2'
    except ImportError:
        pass
    return 'sdpa'


def get_gpu_profile() -> dict:
    """
    Detect the active GPU (device 0) and return a profile dict with recommended defaults.

    Returns keys: batch_size, num_workers, compile, precision, profile_name, gpu_name, vram_gb.
    All settings are defaults only — CLI args override them.
    """
    if not torch.cuda.is_available():
        return {
            'batch_size':   256,
            'num_workers':  4,
            'compile':      False,
            'precision':    'fp32',
            'attn_backend': 'sdpa',
            'profile_name': 'cpu',
            'gpu_name':     'CPU (no CUDA)',
            'vram_gb':      0.0,
        }
    props     = torch.cuda.get_device_properties(0)
    vram_gb   = props.total_memory / (1024 ** 3)
    precision = 'bf16' if torch.cuda.is_bf16_supported() else 'fp16'
    backend   = _detect_attn_backend(props.major)
    if vram_gb >= _HIGH_VRAM_GB:
        return {
            'batch_size':   256,
            'num_workers':  6,
            'compile':      True,
            'precision':    precision,
            'attn_backend': backend,
            'profile_name': 'high-VRAM',
            'gpu_name':     props.name,
            'vram_gb':      vram_gb,
        }
    return {
        'batch_size':   256,
        'num_workers':  4,
        'compile':      False,
        'precision':    precision,
        'attn_backend': backend,
        'profile_name': 'low-VRAM',
        'gpu_name':     props.name,
        'vram_gb':      vram_gb,
    }


def print_gpu_profile(profile: dict) -> None:
    print(
        f"[GPU] {profile['gpu_name']}  VRAM={profile['vram_gb']:.1f}GB  "
        f"profile={profile['profile_name']}  "
        f"→ batch_size={profile['batch_size']}  num_workers={profile['num_workers']}  "
        f"compile={profile['compile']}  precision={profile['precision']}  "
        f"attn={profile['attn_backend']}",
        flush=True,
    )
