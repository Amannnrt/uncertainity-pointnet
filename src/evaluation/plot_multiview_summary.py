"""
Generates a summary chart of the four Re-scan validation experiments (Section 9.5), reading
from already-saved results where available. Experiment 1's raw per-sample CSV was overwritten
by Experiment 2 (same filename, not re-run), so its bar uses the already-reported headline
number directly rather than re-deriving it — this is noted in the plot itself.

Usage:
    python3 src/evaluation/plot_multiview_summary.py
"""
import os
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "experiments")

candidates = sorted(glob.glob(os.path.join(EXPERIMENTS_DIR, "dropout_ablation_p0.1_*")))
out_dir = candidates[-1]

# Reported correct-promotion rates (Re-scan -> Grasp, correct) for each experiment, as
# printed and recorded in PROGRESS_REPORT.md Section 9.5.
experiments = [
    "Exp 1: Spatial fusion\n(fixed budget)",
    "Exp 2: Spatial fusion\n(natural budget)",
    "Exp 3: Probability\naveraging",
    "Exp 4: Agreement-based\naggregation",
]
correct_promotion_rates = [20.0, 26.1, 0.0, 0.6]
oracle_rate = 1.3  # separate diagnostic, not a promotion rate — shown as a reference line

fig, ax = plt.subplots(figsize=(9, 5.5))
bars = ax.bar(experiments, correct_promotion_rates, color=["#5b8dd6", "#5b8dd6", "#d65b5b", "#d6a05b"])
ax.axhline(oracle_rate, color="gray", linestyle="--",
           label=f"Oracle ceiling: {oracle_rate}% (info actually present in a 2nd view)")

for bar, val in zip(bars, correct_promotion_rates):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.5, f"{val}%", ha="center", fontsize=10)

ax.set_ylabel("Correct promotion rate\n(Re-scan tier -> Grasp, correct)")
ax.set_title("Re-scan Validation: Four Fusion Strategies vs. the Information Ceiling")
ax.set_ylim(0, max(correct_promotion_rates) + 8)
ax.legend()
fig.tight_layout()

save_path = os.path.join(out_dir, "multiview_summary_plot.png")
fig.savefig(save_path, dpi=150)
plt.close(fig)
print(f"Saved: {save_path}")
print("Note: Exp 1's value is the previously reported/printed figure (its raw CSV was "
      "overwritten by Exp 2's run, same output filename) — not re-derived here.")
