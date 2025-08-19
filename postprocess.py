import os
import csv
import glob
import time
import math
import argparse
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# —— Visual style tweaks only (no numerical effect) ——
mpl.rcParams["lines.antialiased"] = True
mpl.rcParams["path.simplify"] = True
mpl.rcParams["path.simplify_threshold"] = 0.0
mpl.rcParams["axes.titlepad"] = 6

# -----------------------------
# Robust CSV reading utilities
# -----------------------------
USECOLS_PREFERRED = [
    "iteration",
    "num_labeled",
    "strategy_label",
    "strategy_internal",
    "auc",
    "cindex",
    "thr_at_fpr_0p1",
    "fpr_at_thr",
    "tpr_at_thr",
    "ppv_at_thr",
    "f1_at_thr",
    "mse_val",
    "mse_proba_val",
    "w_uncertainty",
    "w_diversity",
    "w_qbc",
]


def _peek_header_cols(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader([f.readline()])
        for row in reader:
            return row
    return []


def _read_csv_robust(path: str, usecols=None) -> pd.DataFrame:
    header_cols = _peek_header_cols(path)
    final_usecols = (
        [c for c in (usecols or []) if c in header_cols] if usecols else None
    )
    try:
        return pd.read_csv(path, usecols=final_usecols)
    except pd.errors.ParserError:
        return pd.read_csv(
            path,
            engine="python",
            on_bad_lines="skip",
            quotechar='"',
            escapechar="\\",
            usecols=final_usecols,
        )


# -----------------------------
# File discovery
# -----------------------------
def iter_metric_files_for_strategy(logs_root: str, strategy_label: str):
    patt = os.path.join(logs_root, strategy_label, "**", "per_iter_metrics.csv")
    for p in glob.glob(patt, recursive=True):
        yield p


# -----------------------------
# Series helpers
# -----------------------------
def _series_from_df(df: pd.DataFrame, col: str) -> List[float]:
    if col not in df.columns:
        return []
    df2 = df.copy()
    if "iteration" in df2.columns:
        df2 = df2.sort_values("iteration")
    s = df2[col].astype(float)
    s = s.ffill().bfill()
    return s.tolist()


def _final_value_per_run(run_hist: List[float]) -> float:
    arr = np.array(run_hist, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return np.nan
    idx = np.where(np.isfinite(arr))[0]
    if idx.size == 0:
        return np.nan
    return arr[idx[-1]]


# -----------------------------
# Alignment helper (pad runs by last value)
# -----------------------------
def _pad_runs_constant_tail(histories: List[List[float]]) -> np.ndarray:
    nonempty = [r for r in histories if len(r) > 0]
    if not nonempty:
        return np.zeros((0, 0), dtype=float)
    T = max(len(r) for r in nonempty)
    rows = []
    for r in histories:
        if len(r) == 0:
            rows.append([np.nan] * T)
            continue
        row = list(r) + [r[-1]] * (T - len(r))
        rows.append(row[:T])
    return np.array(rows, dtype=float)


# -----------------------------
# Plotting
# -----------------------------
DEFAULT_STRATEGY_COLORS = {
    "random": "C0",
    "uncertainty": "C1",
    "diversity": "C2",
    "qbc": "C7",
    "hybrid_rl": "C3",
    "S_star_baseline": "C4",
    "S_true_oracle": "C5",
}

# Only keep main metrics (no confusion matrix counts)
METRIC_LABELS = {
    "AUC": "Validation AUC",
    "F1": "F1 @ FPR=0.1",
    "TPR": "TPR @ FPR=0.1",
    "PPV": "PPV @ FPR=0.1",
    "MSE": "Probability MSE (↓)",
}


def _mean_ci(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(A, axis=0)
    std = np.nanstd(A, axis=0)
    n = np.sum(~np.isnan(A), axis=0)
    n = np.clip(n, 1, None)
    ci = 1.96 * std / np.sqrt(n)
    return mean, mean - ci, mean + ci


def plot_facets(
    metric_runs: Dict[str, Dict[str, List[List[float]]]],
    strategy_colors: Dict[str, str],
    out_path: str,
    ncols: int = 3,
    coverage: float = 1.0,  # kept for compatibility
    smooth_win: int = 1,  # kept for compatibility
    right_labels: bool = True,
    exclude_metrics: List[str] | None = None,
):
    exclude_set = set(exclude_metrics or [])

    # collect non-empty metrics
    metrics_to_plot = []
    for m, d in metric_runs.items():
        if m in exclude_set:
            continue
        if isinstance(d, dict) and any(len(r) > 0 for runs in d.values() for r in runs):
            metrics_to_plot.append(m)
    if not metrics_to_plot:
        print("[WARN] No metrics to plot.")
        return

    n_panels = len(metrics_to_plot)
    ncols = min(ncols, n_panels)
    nrows = math.ceil(n_panels / ncols)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.6 * ncols + 2, 3.4 * nrows + 1), squeeze=False
    )
    axes = axes.ravel()

    strategy_order = list(strategy_colors.keys())

    def _pad_to_len(histories: List[List[float]], T: int) -> np.ndarray:
        """Pad each run to the same length T using last value."""
        if T <= 0:
            return np.zeros((0, 0), dtype=float)
        rows = []
        for r in histories:
            if len(r) == 0:
                rows.append([np.nan] * T)
            else:
                rows.append((list(r) + [r[-1]] * (T - len(r)))[:T])
        return np.array(rows, dtype=float)

    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        mdict = metric_runs[metric]

        global_T = 0
        for runs in mdict.values():
            for r in runs:
                global_T = max(global_T, len(r))
        if global_T == 0:
            ax.axis("off")
            continue
        x = np.arange(global_T)

        any_plotted = False
        ymins, ymaxs = [], []

        for strategy in strategy_order:
            if strategy not in mdict:
                continue
            runs = mdict[strategy]
            A = _pad_to_len(runs, global_T)
            if A.size == 0 or not np.isfinite(A).any():
                continue

            mean, lo, hi = _mean_ci(A)
            color = strategy_colors.get(strategy, None)

            is_baseline = strategy in ("S_star_baseline", "S_true_oracle")
            lw = 2.8 if is_baseline else 2.2
            z = 10 if is_baseline else 2
            alpha = 1.0 if is_baseline else 0.98

            ax.plot(
                x,
                mean,
                label=strategy,
                color=color,
                linewidth=lw,
                alpha=alpha,
                zorder=z,
                solid_capstyle="round",
                solid_joinstyle="round",
            )
            ax.fill_between(
                x,
                lo,
                hi,
                alpha=(0.10 if is_baseline else 0.08),
                color=color,
                linewidth=0,
                zorder=z - 1,
            )

            if right_labels:
                dx = max(2, int(0.03 * (x[-1] + 1))) - 0.5
                ax.text(
                    x[-1] + dx,
                    mean[-1],
                    strategy,
                    color=color,
                    va="center",
                    ha="left",
                    fontsize=9,
                    clip_on=False,
                    zorder=z + 1,
                )

            ymins.append(np.nanmin(lo))
            ymaxs.append(np.nanmax(hi))
            any_plotted = True

        ax.set_xlabel("AL Iteration")
        ax.set_ylabel(METRIC_LABELS.get(metric, metric))
        ax.set_title(metric)
        ax.grid(True, alpha=0.25)
        for sp in ax.spines.values():
            sp.set_alpha(0.3)
        ax.margins(x=0.02)
        if ymins and ymaxs:
            lo, hi = float(np.nanmin(ymins)), float(np.nanmax(ymaxs))
            pad = 0.02 * max(1e-12, hi - lo)
            ax.set_ylim(lo - 0.25 * pad, hi + pad)
        if right_labels:
            dx_global = max(2, int(0.03 * (global_T + 1)))
            ax.set_xlim(0, global_T + dx_global)
        if not any_plotted:
            ax.axis("off")

    for j in range(len(metrics_to_plot), len(axes)):
        axes[j].axis("off")

    handles, labels = [], []
    for i, metric in enumerate(metrics_to_plot):
        h, l = axes[i].get_legend_handles_labels()
        handles += h
        labels += l
    uniq, seen = [], set()
    for h, l in zip(handles, labels):
        if l not in seen:
            uniq.append((h, l))
            seen.add(l)
    if uniq:
        ncols_leg = min(len(uniq), 7)
        fig.legend(
            [h for h, _ in uniq],
            [l for _, l in uniq],
            loc="lower center",
            ncol=ncols_leg,
            frameon=False,
            bbox_to_anchor=(0.5, -0.02),
        )
        plt.subplots_adjust(bottom=0.14)

    fig.suptitle(
        "Validation metrics vs. AL iteration (mean ± 95% CI)",
        y=0.995,
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.06, 1, 0.98])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out_path}")


# -----------------------------
# Main aggregation logic
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs_root", type=str, default="outputs/iter_logs")
    parser.add_argument("--out_dir", type=str, default="outputs/agg")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=[
            "S_star_baseline",
            "S_true_oracle",
            "random",
            "uncertainty",
            "diversity",
            "qbc",
            "hybrid_rl",
        ],
    )
    parser.add_argument("--ncols", type=int, default=3)
    parser.add_argument(
        "--coverage",
        type=float,
        default=1.0,
        help="(Kept for compatibility; not used).",
    )
    parser.add_argument(
        "--smooth_win", type=int, default=1, help="(Kept for compatibility; not used)."
    )
    parser.add_argument("--no_right_labels", action="store_true")
    parser.add_argument(
        "--plot_cindex", action="store_true", help="Include C-index panel."
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Map logical metric names to CSV columns
    COL_MAP = {
        "AUC": "auc",
        "C-index": "cindex",
        "F1": "f1_at_thr",
        "TPR": "tpr_at_thr",
        "PPV": "ppv_at_thr",
    }

    metric_runs: Dict[str, Dict[str, List[List[float]]]] = {
        "AUC": {},
        "C-index": {},  # still collected; default not plotted
        "F1": {},
        "TPR": {},
        "PPV": {},
        "MSE": {},
    }

    for strat in args.strategies:
        for m in metric_runs.keys():
            metric_runs[m].setdefault(strat, [])

        for path in iter_metric_files_for_strategy(args.logs_root, strat):
            df = _read_csv_robust(path, usecols=USECOLS_PREFERRED)
            if df.empty:
                continue
            if "iteration" in df.columns:
                df = df.sort_values("iteration")

            for m, col in COL_MAP.items():
                seq = _series_from_df(df, col)
                metric_runs[m][strat].append(seq)

            # MSE: prefer proba
            if "mse_proba_val" in df.columns:
                seq_mse = _series_from_df(df, "mse_proba_val")
            elif "mse_val" in df.columns:
                seq_mse = _series_from_df(df, "mse_val")
            else:
                seq_mse = []
            metric_runs["MSE"][strat].append(seq_mse)

    # Plot
    fig_path = os.path.join(
        args.out_dir, f"metrics_over_iterations_{time.strftime('%Y%m%d-%H%M%S')}.png"
    )
    exclude = [] if args.plot_cindex else ["C-index"]
    plot_facets(
        metric_runs,
        DEFAULT_STRATEGY_COLORS,
        fig_path,
        ncols=args.ncols,
        coverage=args.coverage,
        smooth_win=args.smooth_win,
        right_labels=not args.no_right_labels,
        exclude_metrics=exclude,
    )

    # Final summary table
    final_rows = []
    for metric_name, strat_dict in metric_runs.items():
        for strategy, runs in strat_dict.items():
            finals = np.array([_final_value_per_run(r) for r in runs], dtype=float)
            if finals.size == 0 or np.all(np.isnan(finals)):
                m = np.nan
                s = np.nan
                n = len(finals)
            else:
                m = float(np.nanmean(finals))
                s = float(np.nanstd(finals))
                n = int(np.sum(np.isfinite(finals)))
            final_rows.append(
                {
                    "metric": metric_name,
                    "strategy": strategy,
                    "final_mean": m,
                    "final_std": s,
                    "n_runs": n,
                }
            )

    final_df = pd.DataFrame(final_rows).sort_values(["metric", "strategy"])
    out_csv = os.path.join(args.out_dir, "final_metrics_summary.csv")
    final_df.to_csv(out_csv, index=False)
    print(f"[Saved] {out_csv}")

    # Backward-compatible separate AUC / C-index CSVs
    if (final_df["metric"] == "AUC").any():
        auc_df = final_df.query("metric == 'AUC'")[
            ["strategy", "final_mean", "final_std", "n_runs"]
        ].rename(columns={"final_mean": "final_auc_mean", "final_std": "final_auc_std"})
        out_auc = os.path.join(args.out_dir, "final_auc_summary.csv")
        auc_df.to_csv(out_auc, index=False)
        print(f"[Saved] {out_auc}")

    if (final_df["metric"] == "C-index").any():
        cidx_df = final_df.query("metric == 'C-index'")[
            ["strategy", "final_mean", "final_std", "n_runs"]
        ].rename(
            columns={"final_mean": "final_cindex_mean", "final_std": "final_cindex_std"}
        )
        out_cidx = os.path.join(args.out_dir, "final_cindex_summary.csv")
        cidx_df.to_csv(out_cidx, index=False)
        print(f"[Saved] {out_cidx}")

    # Pretty print to console
    print("\n=== Final Metrics Summary (mean ± std) ===")
    for metric in final_df["metric"].unique():
        sub = final_df[final_df["metric"] == metric]
        if sub.empty:
            continue
        print(f"\n-- {metric} --")
        for _, row in sub.iterrows():
            m, s, n = row["final_mean"], row["final_std"], int(row["n_runs"])
            if np.isnan(m):
                print(f"{row['strategy']:>18}: nan")
            else:
                print(f"{row['strategy']:>18}: {m:.4f} ± {s:.4f} (n={n})")


if __name__ == "__main__":
    main()
