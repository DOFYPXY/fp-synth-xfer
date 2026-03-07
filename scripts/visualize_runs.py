import os
import re
import sys
import glob
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


def find_run_dirs(base_dir: str) -> list[str]:
    """Return every outputs/* directory that contains eval.runconf."""
    pattern = os.path.join(base_dir, "outputs", "*", "eval.runconf")
    confs = sorted(glob.glob(pattern))
    return [os.path.dirname(c) for c in confs]


def parse_mutation_flags(run_dir: str) -> str:
    conf = os.path.join(run_dir, "eval.runconf")
    with open(conf) as fh:
        for line in fh:
            m = re.search(r"mutation_flags\s*:\s*(.+)", line)
            if m:
                return m.group(1).strip()
    return os.path.basename(run_dir)          # fallback: use folder name


def parse_final_exact(log_path: str) -> float | None:
    """Extract the Exact% from the 'Final Soln' line."""
    with open(log_path) as fh:
        for line in fh:
            if line.startswith("Final Soln"):
                m = re.search(r"Exact\s+([\d.]+)%", line)
                if m:
                    return float(m.group(1))
    return None


def load_all_results(base_dir: str) -> dict[str, dict[str, float]]:
    """
    Returns:
        { run_label: { op_name: exact_pct, ... }, ... }
    """
    results: dict[str, dict[str, float]] = {}
    for run_dir in find_run_dirs(base_dir):
        label = parse_mutation_flags(run_dir)
        ops: dict[str, float] = {}
        for log in sorted(glob.glob(os.path.join(run_dir, "*", "stdout.log"))):
            op = os.path.basename(os.path.dirname(log))
            val = parse_final_exact(log)
            if val is not None:
                ops[op] = val
        if ops:
            results[label] = ops
    return results


PALETTE = [
    "#E63946", 
    "#2A9D8F",
    "#F4A261",
    "#457B9D",
    "#8338EC", 
    "#06D6A0", 
]


def plot_grouped_bar(results: dict[str, dict[str, float]], out_path: str | None = None):
    # Collect all ops in a consistent order
    all_ops = sorted({op for ops in results.values() for op in ops})
    run_labels = list(results.keys())
    n_runs = len(run_labels)
    n_ops  = len(all_ops)

    x = np.arange(n_ops)
    bar_w = 0.8 / n_runs          # bars share the unit interval

    fig, ax = plt.subplots(figsize=(max(12, n_ops * 1.4), 6))

    for i, (label, ops) in enumerate(results.items()):
        vals = [ops.get(op, 0.0) for op in all_ops]
        offset = (i - (n_runs - 1) / 2) * bar_w
        color  = PALETTE[i % len(PALETTE)]
        bars   = ax.bar(x + offset, vals, bar_w * 0.92,
                        label=label, color=color, alpha=0.88,
                        edgecolor="white", linewidth=0.6, zorder=3)

        # value labels on top of each bar
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.15,
                        f"{v:.1f}%",
                        ha="center", va="bottom",
                        fontsize=7, color="#333333", zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(all_ops, rotation=30, ha="right", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax.set_ylabel("Final Soln — Exact %", fontsize=11, labelpad=8)
    ax.set_title("KnownBits (4-bit) - Mutation Configs",
                 fontsize=14, fontweight="bold", pad=14)

    ax.set_ylim(0, max(
        v for ops in results.values() for v in ops.values()
    ) * 1.18)

    ax.yaxis.grid(True, color="#e0e0e0", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    # legend: wrap long labels
    legend = ax.legend(title="mutation_flags", title_fontsize=9,
                       fontsize=8, loc="upper right",
                       framealpha=0.9, edgecolor="#cccccc")
    for text in legend.get_texts():
        text.set_text("\n".join(text.get_text().split(",")))   # line-wrap on comma

    plt.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150)
        print(f"Saved → {out_path}")
    else:
        plt.show()


def plot_heatmap(results: dict[str, dict[str, float]], out_path: str | None = None):
    all_ops = sorted({op for ops in results.values() for op in ops})
    run_labels = list(results.keys())

    matrix = np.array([
        [results[label].get(op, np.nan) for op in all_ops]
        for label in run_labels
    ])

    fig, ax = plt.subplots(figsize=(max(10, len(all_ops) * 1.1),
                                    max(3, len(run_labels) * 1.0 + 1.5)))

    im = ax.imshow(matrix, aspect="auto", cmap="YlGn", vmin=0)
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Exact %", fontsize=10)

    ax.set_xticks(range(len(all_ops)))
    ax.set_xticklabels(all_ops, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(run_labels)))
    # wrap y-labels on comma
    ax.set_yticklabels([l.replace(",", ",\n") for l in run_labels], fontsize=8)

    for i in range(len(run_labels)):
        for j in range(len(all_ops)):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        fontsize=8,
                        color="white" if v > matrix[~np.isnan(matrix)].max() * 0.65 else "#333")

    ax.set_title("Exact % Heatmap — Final Soln", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150)
        print(f"Saved → {out_path}")
    else:
        plt.show()


def main():
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    base_dir = os.path.abspath(base_dir)

    print(f"Scanning: {base_dir}")
    results = load_all_results(base_dir)

    if not results:
        print("No run directories with valid logs found. Check the path.")
        sys.exit(1)

    print(f"\nFound {len(results)} run(s):")
    for label, ops in results.items():
        avg = np.mean(list(ops.values()))
        print(f"  [{label}]  {len(ops)} ops  |  mean Exact = {avg:.2f}%")

    save = "--show" not in sys.argv
    plot_grouped_bar(results, out_path="results_bar.png"     if save else None)
    plot_heatmap    (results, out_path="results_heatmap.png" if save else None)

    if save:
        print("\nDone. Outputs: results_bar.png, results_heatmap.png")


if __name__ == "__main__":
    main()