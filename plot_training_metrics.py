#!/usr/bin/env python3
"""Plot training metrics from DINOv2 training_metrics.json log files.

Usage:
    python plot_training_metrics.py <path_to_training_metrics.json> [--output_dir <dir>]

Generates:
    - loss_curves.png: All loss components + total loss over iterations
    - loss_curves_smoothed.png: Same but with exponential moving average smoothing
    - lr_schedule.png: Learning rate over iterations
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


def load_metrics(filepath):
    """Load JSONL metrics file into a list of dicts."""
    records = []
    with open(filepath, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Warning: skipping malformed line {line_num}")
    print(f"Loaded {len(records)} records from {filepath}")
    return records


def ema_smooth(values, alpha=0.05):
    """Exponential moving average smoothing."""
    smoothed = np.empty_like(values)
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed


def plot_loss_curves(records, output_dir, smoothed=False):
    """Plot individual loss components and total loss."""
    iterations = np.array([r["iteration"] for r in records])

    # Detect available loss keys (raw values, not _median/_global_avg suffixed)
    loss_keys = []
    for key in records[0]:
        if key.endswith("_loss") and not key.endswith(("_median", "_global_avg")):
            loss_keys.append(key)

    has_total = "total_loss" in records[0] and not any(
        k == "total_loss" for k in loss_keys
    )
    if "total_loss" not in loss_keys and has_total:
        loss_keys.insert(0, "total_loss")

    if not loss_keys:
        print("No loss keys found in metrics file!")
        return

    fig, axes = plt.subplots(len(loss_keys), 1, figsize=(12, 4 * len(loss_keys)), sharex=True)
    if len(loss_keys) == 1:
        axes = [axes]

    suffix = "_smoothed" if smoothed else ""
    title_suffix = " (EMA smoothed, α=0.05)" if smoothed else ""

    for ax, key in zip(axes, loss_keys):
        values = np.array([r.get(key, float("nan")) for r in records])
        valid = ~np.isnan(values)

        if smoothed:
            plot_values = ema_smooth(values[valid])
            # Also show raw as faint background
            ax.plot(iterations[valid], values[valid], alpha=0.15, color="gray", linewidth=0.5)
            ax.plot(iterations[valid], plot_values, linewidth=1.5, label=f"{key} (smoothed)")
        else:
            plot_values = values[valid]
            ax.plot(iterations[valid], plot_values, linewidth=0.8, label=key)

        # Also plot the median and global_avg if available
        median_key = key + "_median"
        global_avg_key = key + "_global_avg"
        if median_key in records[0] and not smoothed:
            median_vals = np.array([r.get(median_key, float("nan")) for r in records])
            ax.plot(iterations[valid], median_vals[valid], linewidth=1, linestyle="--",
                    alpha=0.7, label=f"{key} (median window)")
        if global_avg_key in records[0] and not smoothed:
            global_vals = np.array([r.get(global_avg_key, float("nan")) for r in records])
            ax.plot(iterations[valid], global_vals[valid], linewidth=1, linestyle=":",
                    alpha=0.7, label=f"{key} (global avg)")

        ax.set_ylabel(key)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_title(f"DINOv2 Loss Curves{title_suffix}")
    axes[-1].set_xlabel("Iteration")
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"loss_curves{suffix}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def plot_lr_schedule(records, output_dir):
    """Plot learning rate schedule."""
    iterations = np.array([r["iteration"] for r in records])
    lr_key = "lr"
    if lr_key not in records[0]:
        print("No lr key found, skipping LR plot")
        return

    lr = np.array([r[lr_key] for r in records])

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(iterations, lr, linewidth=1)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "lr_schedule.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def plot_gradient_summary(records, output_dir):
    """Plot total_loss raw value vs median vs global_avg to show the logging discrepancy."""
    if "total_loss_median" not in records[0]:
        print("No _median/_global_avg keys found (old log format), skipping comparison plot")
        return

    iterations = np.array([r["iteration"] for r in records])
    raw = np.array([r.get("total_loss", float("nan")) for r in records])
    median = np.array([r.get("total_loss_median", float("nan")) for r in records])
    global_avg = np.array([r.get("total_loss_global_avg", float("nan")) for r in records])

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(iterations, raw, linewidth=0.5, alpha=0.5, label="raw (per-step)")
    ax.plot(iterations, ema_smooth(raw), linewidth=1.5, label="raw (EMA smoothed)")
    ax.plot(iterations, median, linewidth=1, linestyle="--", label="median (window=20)")
    ax.plot(iterations, global_avg, linewidth=1, linestyle=":", label="global_avg (cumulative)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Total Loss")
    ax.set_title("Total Loss: Raw vs Median vs Global Average\n(shows why global_avg appears flat)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "loss_logging_comparison.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot DINOv2 training metrics")
    parser.add_argument("metrics_file", help="Path to training_metrics.json")
    parser.add_argument("--output_dir", default=None,
                        help="Directory to save plots (default: same dir as metrics file)")
    args = parser.parse_args()

    if not os.path.exists(args.metrics_file):
        print(f"Error: {args.metrics_file} not found")
        sys.exit(1)

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.metrics_file))
    os.makedirs(output_dir, exist_ok=True)

    records = load_metrics(args.metrics_file)
    if len(records) < 2:
        print("Not enough data points to plot")
        sys.exit(1)

    plot_loss_curves(records, output_dir, smoothed=False)
    plot_loss_curves(records, output_dir, smoothed=True)
    plot_lr_schedule(records, output_dir)
    plot_gradient_summary(records, output_dir)

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
