"""
Post-Training Results Visualization

Plots training and validation loss curves from the post-training CSV log.
Overlays the pre-training final validation loss as a reference baseline.

Saves the resulting plot to results/post_training_results.png

Usage:
    python show_results.py
    python show_results.py --log log/log.csv           # custom log path
    python show_results.py --no-baseline                # skip pre-training baseline
"""
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ========================================= CLI Arguments =====================================================================

parser = argparse.ArgumentParser(description="Visualize post-training results")
parser.add_argument("--log", type=str, default="log/log.csv", help="Path to post-training CSV log")
parser.add_argument("--pretrain-log", type=str, default="../logs/log124M_050226/log.txt",
                    help="Path to pre-training log (for baseline val loss)")
parser.add_argument("--no-baseline", action="store_true", help="Don't show pre-training baseline")
parser.add_argument("--output", type=str, default="results/post_training_results.png", help="Output plot path")
args = parser.parse_args()

# ========================================= Load Post-Training Log ============================================================

def load_post_training_csv(log_path):
    """Load post-training CSV log.
    Columns: step, train_loss, val_loss, lr, grad_norm, tokens_per_sec, mfu, dt_ms
    """
    df = pd.read_csv(log_path)
    streams = {}

    for col, key in [("train_loss", "train"), ("val_loss", "val"), ("lr", "lr"),
                      ("grad_norm", "grad_norm"), ("tokens_per_sec", "tokens_per_sec"),
                      ("dt_ms", "dt_ms")]:
        if col in df.columns:
            col_data = df[["step", col]].dropna(subset=[col])
            if not col_data.empty:
                # Handle string values (the CSV writes formatted strings)
                col_data[col] = pd.to_numeric(col_data[col], errors='coerce')
                col_data = col_data.dropna(subset=[col])
                if not col_data.empty:
                    streams[key] = dict(zip(col_data["step"].astype(int), col_data[col].astype(float)))

    return streams


def load_pretrain_log(log_path):
    """Load pre-training log to get the final validation loss as baseline.
    Supports both .txt (legacy) and .csv formats.
    """
    if not os.path.exists(log_path):
        return None

    if log_path.endswith(".csv"):
        df = pd.read_csv(log_path)
        if "val_loss" in df.columns:
            val_data = df[["step", "val_loss"]].dropna(subset=["val_loss"])
            if not val_data.empty:
                return val_data["val_loss"].iloc[-1]
    else:
        # Legacy .txt format: "step stream value"
        last_val = None
        with open(log_path, "r") as f:
            for line in f:
                parts = line.strip().split(" ")
                if len(parts) >= 3 and parts[1] == "val":
                    last_val = float(parts[2])
        return last_val

    return None


# ========================================= Load Data =========================================================================

print(f"Loading post-training log: {args.log}")
streams = load_post_training_csv(args.log)

for k, v in sorted(streams.items()):
    print(f"  {k}: {len(v)} entries")

# Load pre-training baseline
pretrain_val_loss = None
if not args.no_baseline:
    pretrain_val_loss = load_pretrain_log(args.pretrain_log)
    if pretrain_val_loss is not None:
        print(f"\nPre-training final val loss (baseline): {pretrain_val_loss:.4f}")
    else:
        print(f"\nCould not load pre-training baseline from {args.pretrain_log}")

# ========================================= Create Plots ======================================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Post-Training Results — OD Documentation", fontsize=16, fontweight='bold')

# ---- Plot 1: Train & Val Loss ----
ax1 = axes[0, 0]

if "train" in streams:
    xs, ys = zip(*sorted(streams["train"].items()))
    ax1.plot(xs, ys, label='Train loss', color='#2ca02c', alpha=0.7, linewidth=1.2)

if "val" in streams:
    xs, ys = zip(*sorted(streams["val"].items()))
    ax1.plot(xs, ys, label='Val loss', color='#d62728', alpha=0.9, linewidth=2.0, marker='o', markersize=4)

if pretrain_val_loss is not None:
    ax1.axhline(y=pretrain_val_loss, color='#1f77b4', linestyle='--',
                label=f'Pre-training val loss ({pretrain_val_loss:.2f})', linewidth=1.5, alpha=0.8)

ax1.set_xlabel("Step", fontsize=11)
ax1.set_ylabel("Loss", fontsize=11)
ax1.set_title("Training & Validation Loss", fontsize=13)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# ---- Plot 2: Learning Rate Schedule ----
ax2 = axes[0, 1]

if "lr" in streams:
    xs, ys = zip(*sorted(streams["lr"].items()))
    ax2.plot(xs, ys, color='#9467bd', linewidth=1.5)

ax2.set_xlabel("Step", fontsize=11)
ax2.set_ylabel("Learning Rate", fontsize=11)
ax2.set_title("Learning Rate Schedule", fontsize=13)
ax2.ticklabel_format(axis='y', style='scientific', scilimits=(-2, -2))
ax2.grid(True, alpha=0.3)

# ---- Plot 3: Gradient Norm ----
ax3 = axes[1, 0]

if "grad_norm" in streams:
    xs, ys = zip(*sorted(streams["grad_norm"].items()))
    ax3.plot(xs, ys, color='#ff7f0e', alpha=0.7, linewidth=1.0)

ax3.set_xlabel("Step", fontsize=11)
ax3.set_ylabel("Gradient Norm", fontsize=11)
ax3.set_title("Gradient Norm", fontsize=13)
ax3.grid(True, alpha=0.3)

# ---- Plot 4: Tokens/sec and Step Time ----
ax4 = axes[1, 1]

if "dt_ms" in streams:
    xs, ys = zip(*sorted(streams["dt_ms"].items()))
    ax4.plot(xs, ys, color='#17becf', alpha=0.7, linewidth=1.0, label='Step time (ms)')
    ax4.set_ylabel("Step Time (ms)", fontsize=11, color='#17becf')
    ax4.tick_params(axis='y', labelcolor='#17becf')

    # Secondary y-axis for tokens/sec
    if "tokens_per_sec" in streams:
        ax4_twin = ax4.twinx()
        xs2, ys2 = zip(*sorted(streams["tokens_per_sec"].items()))
        ax4_twin.plot(xs2, ys2, color='#e377c2', alpha=0.7, linewidth=1.0, label='Tokens/sec')
        ax4_twin.set_ylabel("Tokens/sec", fontsize=11, color='#e377c2')
        ax4_twin.tick_params(axis='y', labelcolor='#e377c2')

ax4.set_xlabel("Step", fontsize=11)
ax4.set_title("Throughput", fontsize=13)
ax4.grid(True, alpha=0.3)

plt.tight_layout()

# ========================================= Save Plot =========================================================================

output_dir = os.path.dirname(args.output)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)
plt.savefig(args.output, dpi=150, bbox_inches='tight')
print(f"\nSaved plot to {args.output}")

# ========================================= Print Summary =====================================================================

if "train" in streams:
    train_steps = sorted(streams["train"].items())
    print(f"\nTraining summary:")
    print(f"  Steps:           {len(train_steps)}")
    print(f"  Initial loss:    {train_steps[0][1]:.4f}")
    print(f"  Final loss:      {train_steps[-1][1]:.4f}")
    print(f"  Loss reduction:  {train_steps[0][1] - train_steps[-1][1]:.4f}")

if "val" in streams:
    val_steps = sorted(streams["val"].items())
    print(f"\nValidation summary:")
    print(f"  Eval points:     {len(val_steps)}")
    print(f"  Initial val loss:{val_steps[0][1]:.4f}")
    print(f"  Best val loss:   {min(v for _, v in val_steps):.4f}")
    print(f"  Final val loss:  {val_steps[-1][1]:.4f}")

    if pretrain_val_loss is not None:
        improvement = pretrain_val_loss - min(v for _, v in val_steps)
        print(f"  vs pre-training: {improvement:+.4f} ({'improved' if improvement > 0 else 'degraded'})")
