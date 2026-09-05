"""
Generates the headline baseline-vs-policy comparison chart (Section 9.6), reading directly
from the already-saved baseline_vs_policy_summary.json — no re-computation needed.

Usage:
    python3 src/evaluation/plot_baseline_vs_policy.py
"""
import os
import glob
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "experiments")

candidates = sorted(glob.glob(os.path.join(EXPERIMENTS_DIR, "dropout_ablation_p0.1_*")))
out_dir = candidates[-1]
summary_path = os.path.join(out_dir, "baseline_vs_policy_summary.json")

with open(summary_path) as f:
    summary = json.load(f)

baseline_pct = summary["baseline_unsafe_rate"] * 100
policy_pct = summary["policy_unsafe_rate"] * 100

fig, ax = plt.subplots(figsize=(6.5, 5.5))
bars = ax.bar(["Always Grasp\n(baseline)", "Uncertainty-aware\npolicy"],
              [baseline_pct, policy_pct], color=["#d65b5b", "#5bb88d"], width=0.55)

for bar, val in zip(bars, [baseline_pct, policy_pct]):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.4, f"{val:.2f}%", ha="center", fontsize=12, fontweight="bold")

ax.set_ylabel("Unsafe grasp rate (% of ALL encounters)")
ax.set_title(f"Safety Comparison: {summary['relative_reduction_factor']:.1f}x Fewer Unsafe Grasps")
ax.set_ylim(0, baseline_pct * 1.2)
fig.tight_layout()

save_path = os.path.join(out_dir, "baseline_vs_policy_plot.png")
fig.savefig(save_path, dpi=150)
plt.close(fig)
print(f"Saved: {save_path}")
