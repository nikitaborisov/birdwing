#!/usr/bin/env python3
"""Plot GPU multiply benchmark CSV with error bars (log-log)."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_csv(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "limb_bits", "L_arg", "L", "N", "mean_ms", "stddev_ms",
            "min_ms", "max_ms",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = required - set(reader.fieldnames or [])
            raise ValueError(f"CSV missing columns: {sorted(missing)}")

        for row in reader:
            rows.append({
                "limb_bits": int(row["limb_bits"]),
                "L_arg": int(row["L_arg"]),
                "L": int(row["L"]),
                "N": int(row["N"]),
                "logN": int(row["logN"]),
                "mean_ms": float(row["mean_ms"]),
                "stddev_ms": float(row["stddev_ms"]),
                "min_ms": float(row["min_ms"]),
                "max_ms": float(row["max_ms"]),
            })
    if not rows:
        raise ValueError(f"No data rows in {path}")
    return rows


def group_rows(rows: list[dict], x_key: str) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        label = f"{row['limb_bits']}-bit"
        groups[label].append(row)

    for label in groups:
        groups[label].sort(key=lambda r: r[x_key])
    return groups


def error_intervals(series: list[dict], mode: str) -> tuple[np.ndarray, np.ndarray]:
    y = np.array([r["mean_ms"] for r in series], dtype=float)

    if mode == "stddev":
        err = np.array([r["stddev_ms"] for r in series], dtype=float)
        lower = np.maximum(y - err, np.finfo(float).tiny)
        upper = y + err
        return lower, upper

    lower = np.array([r["min_ms"] for r in series], dtype=float)
    upper = np.array([r["max_ms"] for r in series], dtype=float)
    lower = np.maximum(lower, np.finfo(float).tiny)
    return lower, upper


def print_variability(series: list[dict], mode: str) -> None:
    rel = []
    for r in series:
        if mode == "stddev":
            rel.append(100.0 * r["stddev_ms"] / r["mean_ms"])
        else:
            rel.append(100.0 * (r["max_ms"] - r["min_ms"]) / r["mean_ms"])
    print(
        f"  spread ({mode}): "
        f"median={np.median(rel):.2f}%, max={np.max(rel):.2f}% of mean"
    )
    if mode == "stddev" and np.max(rel) < 3.0:
        print(
            "  note: stddev is <3% of mean everywhere — error bars are tiny on a "
            "log-y plot; try --error minmax or --error band"
        )


def is_pow2(v: float) -> bool:
    n = int(v)
    return n > 0 and (n & (n - 1)) == 0


def configure_xaxis(ax, x_key: str, x_values: np.ndarray) -> None:
    """Use base-2 tick marks and labels on the x-axis."""
    if x_key == "L_arg":
        ticks = sorted({int(v) for v in x_values})
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"$2^{{{t}}}$" for t in ticks])
        return

    ax.set_xscale("log", base=2)

    unique = sorted({float(v) for v in x_values})
    if unique and all(is_pow2(v) for v in unique):
        exps = [int(round(np.log2(v))) for v in unique]
        ticks = [2 ** e for e in exps]
        labels = [f"$2^{{{e}}}$" for e in exps]
    else:
        x_min = max(float(np.min(x_values)), 1.0)
        x_max = float(np.max(x_values))
        exp_min = int(np.floor(np.log2(x_min)))
        exp_max = int(np.ceil(np.log2(x_max)))
        exps = list(range(exp_min, exp_max + 1))
        ticks = [2 ** e for e in exps]
        labels = [f"$2^{{{e}}}$" for e in exps]

    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)


def plot_benchmark(
    rows: list[dict],
    *,
    x_key: str,
    error_mode: str,
    output: Path | None,
    title: str | None,
    show: bool,
) -> None:
    groups = group_rows(rows, x_key)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_yscale("log")

    all_x: list[float] = []

    for label, series in sorted(groups.items()):
        x = np.array([r[x_key] for r in series], dtype=float)
        all_x.extend(x.tolist())
        y = np.array([r["mean_ms"] for r in series], dtype=float)
        lower, upper = error_intervals(series, error_mode)

        print(label + ":")
        print_variability(series, error_mode)

        if error_mode == "band":
            ax.fill_between(x, lower, upper, alpha=0.25, label=label)
            ax.plot(x, y, marker="o", linewidth=1.5)
        else:
            yerr = np.vstack([y - lower, upper - y])
            ax.errorbar(
                x, y,
                yerr=yerr,
                marker="o",
                capsize=4,
                capthick=1.2,
                elinewidth=1.2,
                linewidth=1.5,
                label=label,
            )

    x_labels = {
        "L": "Limb count",
        "N": "NTT size",
        "L_arg": "Limb count",
    }
    error_labels = {
        "stddev": "mean ± stddev",
        "minmax": "min – max",
        "band": "mean ± stddev (shaded)",
    }
    configure_xaxis(ax, x_key, np.array(all_x))
    ax.set_xlabel(x_labels.get(x_key, x_key))
    ax.set_ylabel("Time (ms)")
    ax.set_title(title or "GPU full multiply benchmark")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    subtitle = error_labels[error_mode]
    ax.text(
        0.98, 0.02, subtitle,
        transform=ax.transAxes,
        fontsize=9,
        alpha=0.7,
        verticalalignment="bottom",
        horizontalalignment="right",
    )

    if len(groups) > 1:
        ax.legend(loc="upper left")

    fig.tight_layout()

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Wrote {output}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot gpu_multiply_bench.csv with error bars.",
    )
    parser.add_argument(
        "csv",
        nargs="?",
        default="gpu_multiply_bench.csv",
        help="Benchmark CSV (default: gpu_multiply_bench.csv)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output image path (default: <csv>.png)",
    )
    parser.add_argument(
        "--x",
        choices=("L", "N", "L_arg"),
        default="L",
        help="X-axis column (default: L)",
    )
    parser.add_argument(
        "--error",
        choices=("stddev", "minmax", "band"),
        default="minmax",
        help="Error display: stddev, minmax (default), or band (shaded stddev)",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Plot title",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show interactive plot window",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    output = args.output
    if output is None:
        output = csv_path.with_suffix(".png")

    rows = load_csv(csv_path)
    plot_benchmark(
        rows,
        x_key=args.x,
        error_mode=args.error,
        output=output,
        title=args.title,
        show=args.show,
    )


if __name__ == "__main__":
    main()
