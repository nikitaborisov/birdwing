#!/usr/bin/env python3
"""Plot GPU multiply benchmark CSV: total timing + stacked breakdown."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CORE_COLUMNS = {
    "L_arg", "L", "N", "mean_ms", "stddev_ms", "min_ms", "max_ms",
}

PIPELINE_LABELS = {
    "32": "32-bit",
    "hybrid": "hybrid",
    "64bit": "64-bit",
}


def resolve_host_limb_bits(row: dict) -> int:
    if "host_limb_bits" in row:
        return int(row["host_limb_bits"])
    pipeline = row.get("pipeline")
    if pipeline == "64bit":
        return 64
    if pipeline in ("32", "hybrid"):
        return 32
    # Legacy CSV: limb_bits was NTT width (64 for hybrid and 64-bit).
    if row.get("limb_bits") == 32:
        return 32
    return 32


def enrich_operand_bits(entry: dict) -> None:
    host_bits = resolve_host_limb_bits(entry)
    entry["host_limb_bits"] = host_bits
    if "operand_bits" not in entry:
        entry["operand_bits"] = int(entry["L"]) * host_bits


def series_label(row: dict) -> str:
    if "pipeline" in row and row["pipeline"]:
        return PIPELINE_LABELS.get(row["pipeline"], row["pipeline"])
    # Legacy CSV: limb_bits was NTT width (64 for both hybrid and 64-bit).
    return f"{row['limb_bits']}-bit"

EXECUTE_STACK_LAYERS: tuple[tuple[str, str], ...] = (
    ("ingress_fwd_mean_ms", "H2D + fwd pad+NTT"),
    ("mul_mean_ms", "Pointwise mul"),
    ("intt_mean_ms", "INTT"),
    ("crt_mean_ms", "CRT"),
    ("carry_mean_ms", "Carry"),
    ("d2h_mean_ms", "D2H"),
)

# Diagnostic columns (overlapping across streams — not stacked together).
EXECUTE_DIAG_COLUMNS = frozenset({
    "h2d_mean_ms",
    "fwd_pad_ntt_mean_ms",
    "fwd_pad_ntt_a_mean_ms",
    "fwd_pad_ntt_b_mean_ms",
})

SEQUENTIAL_EXECUTE_COLUMNS = (
    "mul_mean_ms",
    "intt_mean_ms",
    "crt_mean_ms",
    "carry_mean_ms",
    "d2h_mean_ms",
)

INFRA_LAYERS: tuple[tuple[str, str], ...] = (
    ("setup_pinned_ms", "Setup: pinned"),
    ("upload_twiddle_ms", "Upload: twiddles"),
    ("upload_mod_constants_ms", "Upload: mod/n⁻¹"),
    ("upload_garner_ms", "Upload: garner"),
    ("setup_alloc_ms", "Setup: alloc"),
    ("teardown_free_ctx_ms", "Teardown: free ctx"),
    ("teardown_free_pre_ms", "Teardown: free pre"),
    ("teardown_free_pinned_ms", "Teardown: free pinned"),
)

PRECOMPUTE_LAYERS: tuple[tuple[str, str], ...] = (
    ("pre_factors_ms", "Factors"),
    ("pre_params_ms", "NTTParameters"),
    ("pre_twiddle_host_ms", "Twiddle (host)"),
    ("pre_garner_host_ms", "Garner (host)"),
)

INFRA_COLUMNS = {key for key, _ in INFRA_LAYERS}
PRECOMPUTE_COLUMNS = {key for key, _ in PRECOMPUTE_LAYERS}
EXECUTE_STACK_COLUMNS = {key for key, _ in EXECUTE_STACK_LAYERS}
OPTIONAL_BREAKDOWN_COLUMNS = (
    EXECUTE_STACK_COLUMNS | INFRA_COLUMNS | PRECOMPUTE_COLUMNS | EXECUTE_DIAG_COLUMNS
)

EXECUTE_COLORS = [plt.cm.tab10(i) for i in range(len(EXECUTE_STACK_LAYERS))]

# tab20 — distinct palettes per panel
INFRA_COLORS = [plt.cm.tab20(i + 2) for i in range(len(INFRA_LAYERS))]
PRECOMPUTE_COLORS = [plt.cm.Set2(i) for i in range(len(PRECOMPUTE_LAYERS))]


def derive_ingress_fwd(row: dict) -> float:
    """Wall-clock ingress phase; residual from total when column is absent."""
    if "ingress_fwd_mean_ms" in row:
        return float(row["ingress_fwd_mean_ms"])
    sequential = sum(float(row[col]) for col in SEQUENTIAL_EXECUTE_COLUMNS)
    return max(float(row["mean_ms"]) - sequential, 0.0)


def enrich_breakdown_row(entry: dict) -> None:
    if "ingress_fwd_mean_ms" not in entry:
        entry["ingress_fwd_mean_ms"] = derive_ingress_fwd(entry)
    # Legacy CSV: single precompute total without stage columns.
    if "pre_factors_ms" not in entry and "setup_precompute_ms" in entry:
        entry["pre_twiddle_host_ms"] = float(entry["setup_precompute_ms"])


def has_breakdown_columns(fieldnames: list[str] | None) -> bool:
    if fieldnames is None:
        return False
    names = set(fieldnames)
    if not set(SEQUENTIAL_EXECUTE_COLUMNS).issubset(names):
        return False
    has_execute = (
        "ingress_fwd_mean_ms" in names
        or EXECUTE_DIAG_COLUMNS.intersection(names)
    )
    has_infra = INFRA_COLUMNS.issubset(names) or {
        "setup_pinned_ms", "setup_alloc_ms",
        "teardown_free_ctx_ms", "teardown_free_pinned_ms",
    }.issubset(names)
    return has_execute and has_infra


def has_precompute_columns(fieldnames: list[str] | None) -> bool:
    if fieldnames is None:
        return False
    names = set(fieldnames)
    return (
        PRECOMPUTE_COLUMNS.issubset(names)
        or "setup_precompute_ms" in names
    )


def load_csv(path: Path) -> tuple[list[dict], bool]:
    rows: list[dict] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or not CORE_COLUMNS.issubset(reader.fieldnames):
            missing = CORE_COLUMNS - set(reader.fieldnames or [])
            raise ValueError(f"CSV missing columns: {sorted(missing)}")
        if "pipeline" not in reader.fieldnames and "limb_bits" not in reader.fieldnames:
            raise ValueError("CSV missing pipeline or limb_bits column")

        has_breakdown = has_breakdown_columns(reader.fieldnames)

        for row in reader:
            entry: dict = {
                "L_arg": int(row["L_arg"]),
                "L": int(row["L"]),
                "N": int(row["N"]),
                "logN": int(row["logN"]) if "logN" in row else int(math.log2(int(row["N"]))),
                "mean_ms": float(row["mean_ms"]),
                "stddev_ms": float(row["stddev_ms"]),
                "min_ms": float(row["min_ms"]),
                "max_ms": float(row["max_ms"]),
            }
            if "operand_bits" in row and row["operand_bits"]:
                entry["operand_bits"] = int(row["operand_bits"])
            if "pipeline" in row and row["pipeline"]:
                entry["pipeline"] = row["pipeline"]
            if "host_limb_bits" in row and row["host_limb_bits"]:
                entry["host_limb_bits"] = int(row["host_limb_bits"])
            elif "limb_bits" in row and row["limb_bits"]:
                entry["limb_bits"] = int(row["limb_bits"])
            enrich_operand_bits(entry)
            if has_breakdown:
                for key in OPTIONAL_BREAKDOWN_COLUMNS:
                    if key in row:
                        entry[key] = float(row[key])
                if "setup_precompute_ms" in row:
                    entry["setup_precompute_ms"] = float(row["setup_precompute_ms"])
                enrich_breakdown_row(entry)
            rows.append(entry)

    if not rows:
        raise ValueError(f"No data rows in {path}")
    return rows, has_breakdown


def group_rows(rows: list[dict], x_key: str) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[series_label(row)].append(row)

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


X_AXIS_LABELS = {
    "operand_bits": "Operand size (bits per input)",
    "L": "Limb count",
    "N": "NTT size",
    "L_arg": "Limb count ($2^{L_{arg}}$)",
}


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


def x_tick_labels(x_key: str, x_values: list[float]) -> list[str]:
    if x_key == "L_arg":
        return [f"$2^{{{int(v)}}}$" for v in x_values]
    if all(is_pow2(v) for v in x_values):
        return [f"$2^{{{int(round(math.log2(v)))}}}$" for v in x_values]
    return [str(int(v)) if float(v).is_integer() else f"{v:g}" for v in x_values]


def stacked_output_path(total_output: Path) -> Path:
    return total_output.with_name(f"{total_output.stem}_stacked{total_output.suffix}")


def plot_total(
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

    x_labels = X_AXIS_LABELS
    error_labels = {
        "stddev": "mean ± stddev",
        "minmax": "min – max",
        "band": "mean ± stddev (shaded)",
    }
    configure_xaxis(ax, x_key, np.array(all_x))
    ax.set_xlabel(x_labels.get(x_key, x_key))
    ax.set_ylabel("Execute time (ms)")
    ax.set_title(title or "GPU full multiply — execute time")
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


def draw_stacked_bars(
    ax: plt.Axes,
    groups: dict[str, list[dict]],
    layers: tuple[tuple[str, str], ...],
    colors: list,
    x_key: str,
    *,
    ylabel: str,
    show_xlabels: bool = True,
) -> tuple[list, list[str], list[str]]:
    n_groups = len(groups)
    group_labels = sorted(groups)
    all_x_values: list[float] = []
    for series in groups.values():
        all_x_values.extend(r[x_key] for r in series)
    unique_x = sorted({float(v) for v in all_x_values})
    x_to_idx = {v: i for i, v in enumerate(unique_x)}

    bar_width = min(0.75 / max(n_groups, 1), 0.35)
    handles: list = []
    labels: list[str] = []

    for g_idx, group_label in enumerate(group_labels):
        series = groups[group_label]
        x_vals = [float(r[x_key]) for r in series]
        indices = np.array([x_to_idx[v] for v in x_vals], dtype=float)
        offset = (g_idx - (n_groups - 1) / 2.0) * bar_width
        x_pos = indices + offset

        bottoms = np.zeros(len(series), dtype=float)
        for layer_idx, (col, label) in enumerate(layers):
            heights = np.array([
                max(float(r.get(col, 0.0)), 0.0) for r in series
            ])
            bars = ax.bar(
                x_pos,
                heights,
                bar_width,
                bottom=bottoms,
                color=colors[layer_idx],
                edgecolor="white",
                linewidth=0.4,
                label=label,
            )
            if g_idx == 0:
                handles.append(bars)
                labels.append(label)
            bottoms += heights

    tick_positions = np.arange(len(unique_x))
    ax.set_xticks(tick_positions)
    if show_xlabels:
        ax.set_xticklabels(x_tick_labels(x_key, unique_x))
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    return handles, labels, group_labels


def plot_stacked_breakdown(
    rows: list[dict],
    *,
    x_key: str,
    output: Path | None,
    title: str | None,
    show: bool,
    has_precompute: bool,
) -> None:
    groups = group_rows(rows, x_key)

    if has_precompute:
        fig, (ax_mult, ax_setup, ax_pre) = plt.subplots(
            3, 1,
            figsize=(11, 11),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 2, 4]},
        )
    else:
        fig, (ax_mult, ax_setup) = plt.subplots(
            2, 1,
            figsize=(11, 8),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 2]},
        )
        ax_pre = None

    mult_handles, mult_labels, group_labels = draw_stacked_bars(
        ax_mult,
        groups,
        EXECUTE_STACK_LAYERS,
        EXECUTE_COLORS,
        x_key,
        ylabel="Multiply (ms)",
        show_xlabels=False,
    )
    setup_handles, setup_labels, _ = draw_stacked_bars(
        ax_setup,
        groups,
        INFRA_LAYERS,
        INFRA_COLORS,
        x_key,
        ylabel="Setup / teardown (ms)",
        show_xlabels=False,
    )

    pre_handles: list = []
    pre_labels: list[str] = []
    if ax_pre is not None:
        pre_handles, pre_labels, _ = draw_stacked_bars(
            ax_pre,
            groups,
            PRECOMPUTE_LAYERS,
            PRECOMPUTE_COLORS,
            x_key,
            ylabel="Precompute (ms)",
            show_xlabels=True,
        )

    x_labels = X_AXIS_LABELS
    if ax_pre is not None:
        ax_pre.set_xlabel(x_labels.get(x_key, x_key))
    else:
        ax_setup.set_xlabel(x_labels.get(x_key, x_key))

    ax_mult.set_title(title or "GPU full multiply — timing breakdown")
    ax_mult.text(
        0.02, 0.98,
        "Multiply bucket — per-iteration mean.",
        transform=ax_mult.transAxes,
        fontsize=8,
        alpha=0.8,
        verticalalignment="top",
    )
    ax_setup.text(
        0.02, 0.98,
        "Setup / teardown — once per L (amortize over many multiplies at same size).",
        transform=ax_setup.transAxes,
        fontsize=8,
        alpha=0.8,
        verticalalignment="top",
    )
    if ax_pre is not None:
        ax_pre.text(
            0.02, 0.98,
            "Precompute bucket — host-only, once per N.",
            transform=ax_pre.transAxes,
            fontsize=8,
            alpha=0.8,
            verticalalignment="top",
        )

    n_groups = len(groups)
    if n_groups > 1:
        group_handles = [
            plt.Line2D([0], [0], color="black", linewidth=0, marker="s",
                       markerfacecolor="none", markeredgecolor="black",
                       label=label)
            for label in group_labels
        ]
        mult_legend = ax_mult.legend(
            handles=mult_handles,
            labels=mult_labels,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            fontsize=7,
            title="Multiply",
        )
        ax_mult.add_artist(mult_legend)
        ax_mult.legend(
            handles=group_handles,
            loc="upper left",
            bbox_to_anchor=(1.02, 0.45),
            fontsize=7,
            title="Series",
        )
        ax_setup.legend(
            handles=setup_handles,
            labels=setup_labels,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            fontsize=7,
            title="Setup / teardown",
        )
        if ax_pre is not None:
            ax_pre.legend(
                handles=pre_handles,
                labels=pre_labels,
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                fontsize=7,
                title="Precompute",
            )
    else:
        ax_mult.legend(
            handles=mult_handles,
            labels=mult_labels,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            fontsize=7,
            title="Multiply",
        )
        ax_setup.legend(
            handles=setup_handles,
            labels=setup_labels,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            fontsize=7,
            title="Setup / teardown",
        )
        if ax_pre is not None:
            ax_pre.legend(
                handles=pre_handles,
                labels=pre_labels,
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                fontsize=7,
                title="Precompute",
            )

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
        description="Plot gpu_multiply_bench.csv (total + stacked breakdown).",
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
        help="Total-timing image path (default: <csv>.png); "
             "stacked breakdown -> <stem>_stacked.png",
    )
    parser.add_argument(
        "--x",
        choices=("operand_bits", "L", "N", "L_arg"),
        default="operand_bits",
        help="X-axis column (default: operand_bits = L × host limb width)",
    )
    parser.add_argument(
        "--error",
        choices=("stddev", "minmax", "band"),
        default="stddev",
        help="Error display on total plot (default: stddev)",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Title for total plot (stacked plot gets a derived title)",
    )
    parser.add_argument(
        "--stacked-title",
        default=None,
        help="Title for stacked breakdown plot",
    )
    parser.add_argument(
        "--no-stacked",
        action="store_true",
        help="Skip stacked breakdown plot",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show interactive plot windows",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    total_output = args.output if args.output is not None else csv_path.with_suffix(".png")
    stacked_output = stacked_output_path(total_output)

    rows, has_breakdown = load_csv(csv_path)
    fieldnames: list[str] | None = None
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])

    plot_total(
        rows,
        x_key=args.x,
        error_mode=args.error,
        output=total_output,
        title=args.title,
        show=args.show,
    )

    if args.no_stacked:
        return

    if not has_breakdown:
        print(
            "Skipping stacked plot: CSV lacks fine-grained timing columns "
            "(re-run bench_full_multiply with current benchmark)."
        )
        return

    stacked_title = args.stacked_title
    if stacked_title is None and args.title is not None:
        stacked_title = f"{args.title} — breakdown"

    plot_stacked_breakdown(
        rows,
        x_key=args.x,
        output=stacked_output,
        title=stacked_title,
        show=args.show,
        has_precompute=has_precompute_columns(fieldnames),
    )


if __name__ == "__main__":
    main()
