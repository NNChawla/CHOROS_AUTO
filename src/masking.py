"""
masking.py — Shared masking utilities for VR encoder training.

Temporal strategies
-------------------
  random_timestep_mask   Uniform random selection of ratio of valid positions,
                         applied independently per sequence.
  span_mask              n_blocks contiguous spans per sequence, each of length
                         ≈ ratio×L / n_blocks, applied independently per sequence.

Feature strategies (return column index lists for input corruption)
-------------------
  device_col_indices     Indices belonging to selected devices (head/left/right).
  group_col_indices      Indices belonging to selected kinematic groups (P/V/A/J).
  feat_col_indices       Union of device and group indices.

Feature masking semantics
--------------------------
  feature masking zeroes the selected columns across ALL timesteps in the input.
  The model sees an incomplete feature vector everywhere and must reconstruct the
  original clean features at temporally-masked positions.  This trains cross-sensor
  and cross-kinematic-group representations.
"""

import math

import numpy as np
import torch

_ALL_SENSORS = ("head", "left", "right")

# Number of feature columns contributed per sensor per kinematic group
_GROUP_SIZES = {"P": 7, "V": 6, "A": 6, "J": 6}


# ---------------------------------------------------------------------------
# Feature-column index helpers
# ---------------------------------------------------------------------------

def device_col_indices(kinematics: str, devices: list[str]) -> list[int]:
    """
    Feature-vector column indices for the given devices across all kinematic groups.

    Parameters
    ----------
    kinematics : e.g. "PVAJ" — same string used to build the dataset
    devices    : subset of ["head", "left", "right"]
    """
    valid = set(_ALL_SENSORS)
    bad   = set(devices) - valid
    if bad:
        raise ValueError(f"Unknown devices: {bad}. Valid: {valid}")

    cols   = []
    offset = 0
    seen   = set()
    for letter in kinematics.upper():
        if letter in seen:
            continue
        if letter not in _GROUP_SIZES:
            raise ValueError(f"Unknown kinematic group '{letter}'. Valid: {set(_GROUP_SIZES)}")
        seen.add(letter)
        size = _GROUP_SIZES[letter]
        for s_idx, sensor in enumerate(_ALL_SENSORS):
            if sensor in devices:
                cols.extend(range(offset + s_idx * size, offset + (s_idx + 1) * size))
        offset += len(_ALL_SENSORS) * size
    return sorted(cols)


def group_col_indices(kinematics: str, groups: list[str]) -> list[int]:
    """
    Feature-vector column indices for the given kinematic groups.

    Parameters
    ----------
    kinematics : e.g. "PVAJ"
    groups     : subset of ["P", "V", "A", "J"]
    """
    groups = {g.upper() for g in groups}
    valid  = set(_GROUP_SIZES)
    bad    = groups - valid
    if bad:
        raise ValueError(f"Unknown kinematic groups: {bad}. Valid: {valid}")

    cols   = []
    offset = 0
    seen   = set()
    for letter in kinematics.upper():
        if letter in seen:
            continue
        seen.add(letter)
        n = len(_ALL_SENSORS) * _GROUP_SIZES[letter]
        if letter in groups:
            cols.extend(range(offset, offset + n))
        offset += n
    return sorted(cols)


def feat_col_indices(
    kinematics: str,
    devices:    list[str] | None = None,
    groups:     list[str] | None = None,
) -> list[int]:
    """
    Union of device and group column indices (deduplicated, sorted).

    Returns an empty list when both devices and groups are None or empty.
    """
    cols: set[int] = set()
    if devices:
        cols |= set(device_col_indices(kinematics, devices))
    if groups:
        cols |= set(group_col_indices(kinematics, groups))
    return sorted(cols)


# ---------------------------------------------------------------------------
# Temporal masking
# ---------------------------------------------------------------------------

def random_timestep_mask(
    lengths: torch.Tensor,   # (B,)  actual sequence lengths
    T:       int,
    ratio:   float,
    device:  torch.device,
) -> torch.Tensor:
    """
    (B, T) bool mask.  For each sequence, selects ratio of valid (non-padding)
    timesteps uniformly at random, independently per sequence.

    Computed on CPU to avoid GPU-CPU synchronization overhead from .item() calls
    inside Python loops — a single .to(device) transfer is done at the end.
    """
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"mask ratio must be in [0, 1], got {ratio}")
    lengths_np = lengths.cpu().numpy()
    B          = len(lengths_np)
    mask       = np.zeros((B, T), dtype=bool)
    for i in range(B):
        L = int(lengths_np[i])
        n = int(round(L * ratio))
        if n == 0:
            continue
        perm       = np.random.permutation(L)[:n]
        mask[i, perm] = True
    return torch.from_numpy(mask).to(device)


def span_mask(
    lengths:  torch.Tensor,   # (B,)
    T:        int,
    ratio:    float,
    n_blocks: int,
    device:   torch.device,
) -> torch.Tensor:
    """
    (B, T) bool mask.  For each sequence, places n_blocks contiguous spans
    until exactly round(ratio*L) positions are masked, independently per sequence.
    Spans may overlap during placement; a random fallback covers any remainder.

    Computed on CPU to avoid GPU-CPU synchronization overhead from .item() calls
    inside Python loops — a single .to(device) transfer is done at the end.
    """
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"ratio must be in [0, 1], got {ratio}")
    if n_blocks < 1:
        raise ValueError(f"n_blocks must be >= 1, got {n_blocks}")
    lengths_np = lengths.cpu().numpy()
    B          = len(lengths_np)
    mask       = np.zeros((B, T), dtype=bool)
    for i in range(B):
        L = int(lengths_np[i])
        if L <= 0:
            continue
        target = int(round(L * ratio))
        if target <= 0:
            continue
        block_len = max(1, math.ceil(target / n_blocks))
        attempts  = 0
        while int(mask[i, :L].sum()) < target and attempts < 1000:
            remaining = target - int(mask[i, :L].sum())
            this_len  = min(block_len, remaining)
            max_start = max(0, L - this_len)
            start     = int(np.random.randint(0, max_start + 1))
            mask[i, start : start + this_len] = True
            attempts += 1
        # Fallback: fill any remaining gap randomly.
        current = int(mask[i, :L].sum())
        if current < target:
            available = np.where(~mask[i, :L])[0]
            np.random.shuffle(available)
            extra     = available[: target - current]
            mask[i, extra] = True
    return torch.from_numpy(mask).to(device)
