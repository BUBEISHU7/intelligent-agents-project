import csv
import os
from collections import defaultdict
from statistics import mean

import matplotlib.pyplot as plt


def generate(out_dir: str):
    csv_path = os.path.join(out_dir, "member_c_week1_all_runs.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing CSV: {csv_path}")

    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["num_bots"] = int(r["num_bots"])
            for k in ("coverage", "collisions", "performance_score"):
                r[k] = float(r[k])
            rows.append(r)

    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["mode"], r["num_bots"])].append(r)

    modes = ["no_coordination", "implicit_coordination", "explicit_coordination"]
    robot_counts = sorted({r["num_bots"] for r in rows})

    summary = []
    for mode in modes:
        for n in robot_counts:
            rs = grouped.get((mode, n), [])
            if not rs:
                continue
            summary.append(
                {
                    "mode": mode,
                    "num_bots": n,
                    "coverage": mean([x["coverage"] for x in rs]),
                    "collisions": mean([x["collisions"] for x in rs]),
                    "performance_score": mean([x["performance_score"] for x in rs]),
                }
            )

    metrics = ["coverage", "collisions", "performance_score"]
    titles = {
        "coverage": "Average Coverage",
        "collisions": "Average Collisions",
        "performance_score": "Average Performance Score",
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        for mode in modes:
            xs, ys = [], []
            for n in robot_counts:
                vals = [s[metric] for s in summary if s["mode"] == mode and s["num_bots"] == n]
                if vals:
                    xs.append(n)
                    ys.append(vals[0])
            if xs:
                ax.plot(xs, ys, marker="o", linewidth=2, label=mode)
        ax.set_title(titles[metric])
        ax.set_xlabel("Number of Robots")
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    plt.tight_layout()
    chart_path = os.path.join(out_dir, "member_c_week1_comparison.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    table_data = [
        [
            s["mode"],
            s["num_bots"],
            f"{s['coverage']:.4f}",
            f"{s['collisions']:.2f}",
            f"{s['performance_score']:.3f}",
        ]
        for s in summary
    ]
    fig2, ax2 = plt.subplots(figsize=(11, 0.45 * max(4, len(table_data)) + 1.8))
    ax2.axis("off")
    table = ax2.table(
        cellText=table_data,
        colLabels=["mode", "num_bots", "avg_coverage", "avg_collisions", "avg_score"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.3)
    plt.title("Member C Week1 Summary", pad=12)
    table_path = os.path.join(out_dir, "member_c_week1_summary_table.png")
    plt.savefig(table_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)

    print(f"Generated: {chart_path}")
    print(f"Generated: {table_path}")


if __name__ == "__main__":
    out = os.path.join("..", "data", "member_c_week1")
    generate(out)

