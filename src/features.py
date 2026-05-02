"""
features.py — Shared kinematic feature-column utilities for VR encoder scripts.

Each letter in the kinematics string selects one order of kinematics for all
three sensors (head, left, right), including both linear and angular channels:

  P — Position / orientation  (xyz pos + xyzw quaternion  →  7 columns/sensor)
  V — Velocity                (linear xyz + angular xyz   →  6 columns/sensor)
  A — Acceleration            (linear xyz + angular xyz   →  6 columns/sensor)
  J — Jerk                    (linear xyz + angular xyz   →  6 columns/sensor)

Examples
--------
  build_feature_cols("P")    →  21 features   (position/orientation only)
  build_feature_cols("PVAJ") →  75 features   (all kinematics)
  build_feature_cols("AJ")   →  36 features   (acceleration + jerk only)
"""

_SENSORS = ("head", "left", "right")

_POSITION_COLS: list[str] = [
    col
    for s in _SENSORS
    for col in (
        [f"{s}_pos_{ax}" for ax in ("x", "y", "z")] +
        [f"{s}_rot_{c}"  for c  in ("x", "y", "z", "w")]
    )
]

_VELOCITY_COLS: list[str] = [
    f"{s}_{kind}vel_{ax}"
    for s    in _SENSORS
    for kind in ("lin_", "ang_")
    for ax   in ("x", "y", "z")
]

_ACCELERATION_COLS: list[str] = [
    f"{s}_{kind}acc_{ax}"
    for s    in _SENSORS
    for kind in ("lin_", "ang_")
    for ax   in ("x", "y", "z")
]

_JERK_COLS: list[str] = [
    f"{s}_{kind}jerk_{ax}"
    for s    in _SENSORS
    for kind in ("lin_", "ang_")
    for ax   in ("x", "y", "z")
]

_ORDER_MAP: dict[str, list[str]] = {
    "P": _POSITION_COLS,
    "V": _VELOCITY_COLS,
    "A": _ACCELERATION_COLS,
    "J": _JERK_COLS,
}


def build_feature_cols(kinematics: str) -> list[str]:
    """
    Return the ordered list of parquet column names for the given kinematic
    order string.

    Parameters
    ----------
    kinematics : str
        Any non-empty combination of the letters P, V, A, J (case-insensitive).
        Order is preserved; duplicate letters are silently ignored.

    Raises
    ------
    ValueError
        If kinematics is empty or contains an unrecognised letter.
    """
    if not kinematics:
        raise ValueError("kinematics must be a non-empty string, e.g. 'P' or 'PVAJ'.")

    seen: set[str] = set()
    cols: list[str] = []
    for letter in kinematics.upper():
        if letter not in _ORDER_MAP:
            raise ValueError(
                f"Unknown kinematic order '{letter}'. Valid letters: P, V, A, J."
            )
        if letter not in seen:
            cols.extend(_ORDER_MAP[letter])
            seen.add(letter)
    return cols
