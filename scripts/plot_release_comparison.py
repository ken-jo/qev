"""Render the measured comparison as a shareable, source-backed figure."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

root = Path(__file__).resolve().parents[1] / "reports/release-comparison"
names = ["laya-base", "laya-specialist", "qev"]
labels = ["LAYA base", "LAYA specialist", "QEV"]
colors = ["#9ba9b5", "#596c83", "#137f78"]
suites = ["typed_decisions", "ag_news", "emotion"]
titles = ["Typed decisions", "News topic", "Emotion"]
data = {name: json.loads((root / (name + ".json")).read_text("utf-8")) for name in names}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "svg.fonttype": "none"})
fig, axes = plt.subplots(1, 3, figsize=(13, 5.8), sharey=True)
fig.patch.set_facecolor("#fcfcfa")
for ax, suite, title in zip(axes, suites, titles, strict=True):
    ax.set_facecolor("#fcfcfa")
    metrics = [data[name]["suites"][suite]["overall"] for name in names]
    values = np.array([m["accuracy"] * 100 for m in metrics])
    intervals = np.array([m["group_bootstrap_accuracy_95_ci"] for m in metrics]) * 100
    errors = np.stack([values - intervals[:, 0], intervals[:, 1] - values])
    ax.bar(range(3), values, color=colors, width=0.65, zorder=3)
    ax.errorbar(range(3), values, yerr=errors, fmt="none", color="#24313d", capsize=4, zorder=4)
    for i, value in enumerate(values):
        ax.text(i, intervals[i, 1] + 2, f"{value:.2f}%", ha="center", fontsize=12, weight="bold")
    ax.set_title(
        title + (" · 2,000 questions" if suite == "typed_decisions" else " · 400 examples"),
        fontsize=12,
        pad=14,
        loc="left",
    )
    ax.set_xticks(range(3), ["LAYA\nbase", "LAYA\nspecialist", "QEV"])
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", color="#dde1e2", linewidth=0.7, zorder=0)
    ax.tick_params(axis="both", length=0, pad=8)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
axes[0].set_ylabel("Accuracy (%) · all answers, including abstentions")
fig.suptitle(
    "QEV and LAYA on the same English inputs",
    x=0.055,
    y=0.97,
    ha="left",
    fontsize=21,
    weight="bold",
    color="#203039",
)
fig.text(
    0.055,
    0.895,
    "Frozen checkpoints. Shared labels and metrics. RTX 4060 Ti 8 GB.",
    color="#52616a",
    fontsize=12,
)
fig.text(
    0.055,
    0.12,
    "Typed decisions: adapted benchmark for QEV and LAYA specialist. "
    "News/emotion: no QEV task-specific adaptation.\n"
    "LAYA reports news in its training mix and emotion held out. "
    "Pretraining overlap is unknown.\n"
    "Whiskers: descriptive 95% group-bootstrap intervals. "
    "Source: reports/release-comparison (2026-10-01).",
    va="top",
    color="#52616a",
    fontsize=10,
    linespacing=1.6,
)
fig.subplots_adjust(left=0.075, right=0.98, bottom=0.245, top=0.79, wspace=0.18)
fig.savefig(root / "accuracy.png", dpi=170, facecolor=fig.get_facecolor())
fig.savefig(root / "accuracy.svg", facecolor=fig.get_facecolor())
print(root / "accuracy.png")
