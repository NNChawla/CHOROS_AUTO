#!/usr/bin/env python3
"""Generate training graphs from JEPA training logs."""

import argparse
import re
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Generate training graphs from JEPA training logs.")
parser.add_argument("base_dir", nargs="?",
                    default="/srv/CHOROS_AUTO/outputs/checkpoints/5_15_26_explorations",
                    help="Root directory containing run subdirectories")
parser.add_argument("--gpu", nargs="+", metavar="GPU",
                    help="Only include runs from these GPUs (e.g. --gpu 3090 3060)")
parser.add_argument("--index", nargs="+", type=int, metavar="N",
                    help="Only include runs with these per-GPU indices (e.g. --index 1 2)")
args = parser.parse_args()

BASE_DIR = Path(args.base_dir)
FILTER_GPUS   = set(args.gpu)   if args.gpu   else None
FILTER_INDICES = set(args.index) if args.index else None

GRAPHS_DIR = BASE_DIR / "graphs"
GRAPHS_DIR.mkdir(exist_ok=True)

# ── Regex patterns ──────────────────────────────────────────────────────────
RE_EPOCH = re.compile(
    r"Epoch\s+(\d+)/\d+\s+train=([\d.]+)\s+val=([\d.]+)"
)
RE_PROBE = re.compile(
    r"\[Probe\] objective=\S+ target=(\S+) split=(\w+)"
    r" Balanced Acc\.: [\d.]+\s+Acc\.: ([\d.]+)\s+MCC: ([-\d.]+)"
)


def gpu_from_name(name: str) -> str:
    m = re.search(r"_(3\d{3})_", name)
    return m.group(1) if m else "unknown"


def parse_log(log_path: Path):
    jepa = {"epoch": [], "train": [], "val": []}
    # probe[target][split] = {"epoch": [], "mcc": [], "acc": []}
    probe = defaultdict(lambda: defaultdict(lambda: {"epoch": [], "mcc": [], "acc": []}))

    NAN = float("nan")
    current_epoch = 0
    with open(log_path) as f:
        for line in f:
            m = RE_EPOCH.search(line)
            if m:
                ep, tr, vl = int(m.group(1)), float(m.group(2)), float(m.group(3))
                current_epoch = ep
                jepa["epoch"].append(ep)
                jepa["train"].append(tr if tr > 0 else NAN)
                continue

            m = RE_PROBE.search(line)
            if m:
                target, split, acc, mcc = m.group(1), m.group(2), float(m.group(3)), float(m.group(4))
                probe[target][split]["epoch"].append(current_epoch)
                probe[target][split]["mcc"].append(mcc)
                probe[target][split]["acc"].append(acc)

    # Re-collect val JEPA loss (separate pass easier to keep aligned)
    val_by_epoch = {}
    with open(log_path) as f:
        for line in f:
            m = RE_EPOCH.search(line)
            if m:
                ep, _, vl = int(m.group(1)), float(m.group(2)), float(m.group(3))
                if vl > 0:
                    val_by_epoch[ep] = vl
    jepa["val_epochs"] = sorted(val_by_epoch.keys())
    jepa["val_vals"]   = [val_by_epoch[e] for e in jepa["val_epochs"]]

    return jepa, dict(probe)


def aggregate_probe(probe):
    """Average MCC and Acc across all targets, grouped by epoch."""
    agg = {}
    for split in ("val", "train"):
        by_epoch = defaultdict(lambda: {"mcc": [], "acc": []})
        for splits in probe.values():
            d = splits.get(split, {})
            for ep, mcc, acc in zip(d.get("epoch", []), d.get("mcc", []), d.get("acc", [])):
                by_epoch[ep]["mcc"].append(mcc)
                by_epoch[ep]["acc"].append(acc)
        epochs = sorted(by_epoch)
        agg[split] = {
            "epoch": epochs,
            "mcc":   [sum(by_epoch[e]["mcc"]) / len(by_epoch[e]["mcc"]) for e in epochs],
            "acc":   [sum(by_epoch[e]["acc"]) / len(by_epoch[e]["acc"]) for e in epochs],
        }
    return agg


# ── Collect all runs ─────────────────────────────────────────────────────────
# Sort by path (directory names start with timestamp, so this is chronological).
gpu_counters: dict[str, int] = defaultdict(int)
runs = []
for log_path in sorted(BASE_DIR.rglob("train.log")):
    gpu = gpu_from_name(log_path.parent.name)
    gpu_counters[gpu] += 1
    idx = gpu_counters[gpu]
    if FILTER_GPUS and gpu not in FILTER_GPUS:
        continue
    if FILTER_INDICES and idx not in FILTER_INDICES:
        continue
    label = f"{gpu}_{idx}"
    jepa, probe = parse_log(log_path)
    runs.append({"label": label, "jepa": jepa, "probe": probe})

if not runs:
    print(f"No train.log files found under {BASE_DIR}")
    sys.exit(1)

print(f"Found {len(runs)} run(s): {[r['label'] for r in runs]}")

COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


# ── 1. JEPA Loss ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
for i, run in enumerate(runs):
    c = COLORS[i % len(COLORS)]
    j = run["jepa"]
    if j["epoch"]:
        ax.plot(j["epoch"], j["train"], color=c, linestyle="-",
                label=f"{run['label']} train")
    if j["val_epochs"]:
        ax.plot(j["val_epochs"], j["val_vals"], color=c, linestyle="--",
                label=f"{run['label']} val")

ax.set_xlabel("Epoch")
ax.set_ylabel("JEPA Loss")
ax.set_title("JEPA Train / Val Loss")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
out = GRAPHS_DIR / "jepa_loss.png"
fig.tight_layout()
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"Saved: {out}")


# ── 2. Probe MCC and Acc per target ──────────────────────────────────────────
all_targets = set()
for run in runs:
    all_targets.update(run["probe"].keys())

SPLITS = ["val", "train"]
SPLIT_LS = {"val": "-", "train": "--"}

for target in sorted(all_targets):
    for metric, metric_key in [("MCC", "mcc"), ("Accuracy", "acc")]:
        fig, ax = plt.subplots(figsize=(9, 5))
        for i, run in enumerate(runs):
            c = COLORS[i % len(COLORS)]
            for split in SPLITS:
                d = run["probe"].get(target, {}).get(split)
                if d and d["epoch"]:
                    ax.plot(d["epoch"], d[metric_key], color=c,
                            linestyle=SPLIT_LS[split],
                            label=f"{run['label']} {split}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.set_title(f"Probe {metric} — {target}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        safe_target = re.sub(r"[^\w]", "_", target)
        out = GRAPHS_DIR / f"probe_{metric_key}_{safe_target}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Saved: {out}")

# ── 3. Aggregate probe MCC and Acc ───────────────────────────────────────────
for metric, metric_key in [("MCC", "mcc"), ("Accuracy", "acc")]:
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, run in enumerate(runs):
        c = COLORS[i % len(COLORS)]
        agg = aggregate_probe(run["probe"])
        for split in SPLITS:
            d = agg.get(split, {})
            if d.get("epoch"):
                ax.plot(d["epoch"], d[metric_key], color=c,
                        linestyle=SPLIT_LS[split],
                        label=f"{run['label']} {split}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.set_title(f"Probe {metric} — aggregate (mean across targets)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    out = GRAPHS_DIR / f"probe_{metric_key}_aggregate.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")

print("Done.")
