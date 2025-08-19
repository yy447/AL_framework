import os
import re
import csv
import glob
import math
import time
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)

# ---------- Matplotlib polish (no numeric effect) ----------
mpl.rcParams["lines.antialiased"] = True
mpl.rcParams["path.simplify"] = True
mpl.rcParams["path.simplify_threshold"] = 0.0
mpl.rcParams["axes.titlepad"] = 6

# ---------- Strategy colors ----------
DEFAULT_STRATEGY_COLORS = {
    "random": "C0",
    "uncertainty": "C1",
    "diversity": "C2",
    "qbc": "C7",
    "hybrid_rl": "C3",
    "S_star_baseline": "C4",
    "S_true_oracle": "C5",
}

# ---------- Panel labels ----------
METRIC_LABELS = {
    "AUC": "Validation AUC",
    "F1": "F1 @ FPR=0.1",
    "TPR": "TPR @ FPR=0.1",
    "PPV": "PPV @ FPR=0.1",
    "MSE": "Probability MSE (↓)",
}

# ---------- Columns to read from proba files ----------
USECOLS_PROBA = [
    "y_true",
    "proba",
    "iteration",
    "num_labeled",
    "strategy_label",
    "run_seed",
    # optional subgroup metadata
    "PATID",
    "age_index",
    "age_group",
    "age_at_index",
    "SEX",
    "RACE",
    "HISPANIC",
    "smoking_status",
    "COPD",
]


# ===================== Robust CSV IO =====================
def _peek_header_cols(path: str) -> List[str]:
    """Read first line to get header columns."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        rdr = csv.reader([f.readline()])
        for row in rdr:
            return row
    return []


def _read_csv_robust(path: str, usecols=None) -> pd.DataFrame:
    """Read CSV; if malformed lines exist, skip them."""
    header = _peek_header_cols(path)
    final = [c for c in (usecols or []) if c in header] if usecols else None
    try:
        return pd.read_csv(path, usecols=final)
    except pd.errors.ParserError:
        return pd.read_csv(
            path,
            engine="python",
            on_bad_lines="skip",
            quotechar='"',
            escapechar="\\",
            usecols=final,
        )


def _read_one_run_df(run_dir: str) -> Optional[pd.DataFrame]:
    """Prefer all_iters.csv; else concat iter_###.csv files."""
    all_csv = os.path.join(run_dir, "all_iters.csv")
    if os.path.exists(all_csv):
        df = _read_csv_robust(all_csv, usecols=USECOLS_PROBA)
        if df is not None and not df.empty:
            df.columns = df.columns.str.strip()
            return df
        return None
    parts = []
    for p in sorted(glob.glob(os.path.join(run_dir, "iter_*.csv"))):
        d = _read_csv_robust(p, usecols=USECOLS_PROBA)
        if d is not None and not d.empty:
            parts.append(d)
    return pd.concat(parts, ignore_index=True) if parts else None


# ===================== Thresholding & Metrics =====================
def _select_thr_at_fpr(y_true: np.ndarray, proba: np.ndarray, target_fpr=0.10):
    """Pick threshold with max TPR under FPR<=target on the full validation slice."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    try:
        fpr, tpr, th = roc_curve(y_true, proba)
        ok = np.where(fpr <= target_fpr)[0]
        if ok.size > 0:
            i = ok[np.argmax(tpr[ok])]
            return float(th[i]), float(fpr[i]), float(tpr[i])
    except Exception:
        pass
    # Fallback
    return 0.5, np.nan, np.nan


def _cls_metrics_with_fixed_thr(y_true, proba, thr: Optional[float]):
    """Compute AUC and, with a fixed threshold, F1/TPR/PPV; also probability MSE."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)

    auc = np.nan
    if np.unique(y_true).size >= 2:
        try:
            auc = float(roc_auc_score(y_true, proba))
        except Exception:
            pass

    thr = 0.5 if thr is None else float(thr)
    preds = (proba >= thr).astype(int)

    ppv = float(precision_score(y_true, preds, zero_division=0))
    rec = float(recall_score(y_true, preds, zero_division=0))  # TPR
    f1 = float(f1_score(y_true, preds, zero_division=0))
    mse = float(np.mean((proba - y_true) ** 2))

    return {"AUC": auc, "F1": f1, "TPR": rec, "PPV": ppv, "MSE": mse}


# ===================== Plot helpers =====================
def _right_edge_label(ax, x, y, text, color, dx_data=0.0):
    """Right-side inline legend."""
    if len(x) == 0:
        return
    ax.text(
        x[-1] + dx_data,
        y[-1],
        text,
        color=color,
        va="center",
        ha="left",
        fontsize=9,
        clip_on=False,
        zorder=3,
    )


def _sanitize_filename(s: str) -> str:
    """Safe filename from arbitrary text."""
    s = str(s).strip().replace(" ", "_")
    s = re.sub(r"[^\w.\-]+", "_", s)
    return s[:120]


def _pad_to_len(histories: List[List[float]], T: int) -> np.ndarray:
    """Pad each run to length T by repeating the last value."""
    if T <= 0:
        return np.zeros((0, 0), dtype=float)
    rows = []
    for r in histories:
        rows.append(
            ([np.nan] * T) if len(r) == 0 else (list(r) + [r[-1]] * (T - len(r)))[:T]
        )
    return np.array(rows, dtype=float)


def _mean_ci(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean ± 95% CI across runs."""
    mean = np.nanmean(A, axis=0)
    std = np.nanstd(A, axis=0)
    n = np.clip(np.sum(~np.isnan(A), axis=0), 1, None)
    ci = 1.96 * std / np.sqrt(n)
    return mean, mean - ci, mean + ci


def plot_facets_per_subgroup(
    subgroup_name: str,
    subgroup_value: str,
    metric_runs: Dict[
        str, Dict[str, List[List[float]]]
    ],  # metric -> strategy -> [runs]
    out_path: str,
    strategy_colors: Dict[str, str],
    ncols: int = 3,
    right_labels: bool = True,
):
    """Plot mean±CI curves per metric for one subgroup value."""
    metrics = [m for m in METRIC_LABELS.keys() if m in metric_runs]
    if not metrics:
        print(f"[WARN] No metrics for {subgroup_name}={subgroup_value}")
        return

    n_panels = len(metrics)
    ncols = min(ncols, n_panels)
    nrows = math.ceil(n_panels / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.6 * ncols + 2, 3.4 * nrows + 1), squeeze=False
    )
    axes = axes.ravel()
    strategy_order = list(strategy_colors.keys())

    for i, metric in enumerate(metrics):
        ax = axes[i]
        mdict = metric_runs[metric]

        # Determine global length across strategies × runs
        global_T = max((len(r) for runs in mdict.values() for r in runs), default=0)
        if global_T == 0:
            ax.axis("off")
            continue
        x = np.arange(global_T)

        any_plotted, ymins, ymaxs = False, [], []
        for strategy in strategy_order:
            if strategy not in mdict:
                continue
            A = _pad_to_len(mdict[strategy], global_T)
            if A.size == 0 or not np.isfinite(A).any():
                continue

            mean, lo, hi = _mean_ci(A)
            color = strategy_colors.get(strategy, None)
            is_baseline = strategy in ("S_star_baseline", "S_true_oracle")

            ax.plot(
                x,
                mean,
                label=strategy,
                color=color,
                linewidth=(2.8 if is_baseline else 2.2),
                alpha=(1.0 if is_baseline else 0.98),
                zorder=(10 if is_baseline else 2),
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
                zorder=(9 if is_baseline else 1),
            )

            if right_labels:
                dx = max(2, int(0.03 * (x[-1] + 1))) - 0.5
                _right_edge_label(ax, x, mean, strategy, color, dx_data=dx)

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

    for j in range(len(metrics), len(axes)):
        axes[j].axis("off")

    # Shared legend at bottom
    handles, labels = [], []
    for i, metric in enumerate(metrics):
        h, l = axes[i].get_legend_handles_labels()
        handles += h
        labels += l
    uniq, seen = [], set()
    for h, l in zip(handles, labels):
        if l not in seen:
            uniq.append((h, l))
            seen.add(l)
    if uniq:
        fig.legend(
            [h for h, _ in uniq],
            [l for _, l in uniq],
            loc="lower center",
            ncol=min(len(uniq), 7),
            frameon=False,
            bbox_to_anchor=(0.5, -0.02),
        )
        plt.subplots_adjust(bottom=0.14)

    fig.suptitle(
        f"{subgroup_name} = {subgroup_value} | Validation metrics vs. AL iteration (mean ± 95% CI)",
        y=0.995,
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.06, 1, 0.98])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out_path}")


# ===================== Small helpers =====================
def ensure_age_group(df: pd.DataFrame, out_col: str = "age_group") -> pd.DataFrame:
    """Ensure an ordered categorical age_group; create from age_at_index/age_index if missing."""
    labels = ["<35", "35–49", "50–59", "60–70", ">70"]
    if out_col in df.columns:
        try:
            df[out_col] = pd.Categorical(df[out_col], categories=labels, ordered=True)
            return df
        except Exception:
            pass
    src_col = (
        "age_at_index"
        if "age_at_index" in df.columns
        else ("age_index" if "age_index" in df.columns else None)
    )
    if src_col is None:
        return df
    age = pd.to_numeric(df[src_col], errors="coerce")
    bins = [-np.inf, 34, 49, 59, 70, np.inf]
    try:
        df[out_col] = pd.cut(
            age, bins=bins, labels=labels, right=True, include_lowest=True, ordered=True
        )
        df[out_col] = pd.Categorical(df[out_col], categories=labels, ordered=True)
    except Exception:
        pass
    return df


def final_value_per_run(run_hist: List[float]) -> float:
    """Return the last finite value from a sequence."""
    arr = np.array(run_hist, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return np.nan
    idx = np.where(np.isfinite(arr))[0]
    return arr[idx[-1]] if idx.size > 0 else np.nan


# ===================== Summary helpers =====================
def summarize_best_per_subgroup_value(
    final_csv_path: str,
    primary_metric: str = "AUC",
    strategies_all: Optional[List[str]] = None,
    baseline_for_delta: Optional[str] = None,
):
    """
    Pick the best strategy for one subgroup value on a primary metric.
    Save <value>_best_<metric>_strategy.csv in the same folder.
    """
    if not os.path.exists(final_csv_path):
        print(f"[WARN] File not found: {final_csv_path}")
        cols = [
            "subgroup",
            "value",
            "primary_metric",
            "metric",
            "best_strategy",
            "best_mean",
            "best_std",
            "delta_vs_baseline",
        ]
        return pd.DataFrame([dict(zip(cols, [np.nan] * len(cols)))]), pd.DataFrame()

    df = pd.read_csv(final_csv_path)
    if df.empty:
        cols = [
            "subgroup",
            "value",
            "primary_metric",
            "metric",
            "best_strategy",
            "best_mean",
            "best_std",
            "delta_vs_baseline",
        ]
        return pd.DataFrame([dict(zip(cols, [np.nan] * len(cols)))]), pd.DataFrame()

    # Filter to the primary metric
    if "metric" in df.columns:
        sub = df[df["metric"].astype(str) == str(primary_metric)].copy()
    else:
        sub = df.copy()

    if strategies_all:
        sub = sub[
            sub["strategy"].astype(str).isin([str(s) for s in strategies_all])
        ].copy()

    sub["final_mean"] = pd.to_numeric(sub["final_mean"], errors="coerce")
    sub["final_std"] = pd.to_numeric(sub["final_std"], errors="coerce")

    if not sub["final_mean"].notna().any():
        cols = [
            "subgroup",
            "value",
            "primary_metric",
            "metric",
            "best_strategy",
            "best_mean",
            "best_std",
            "delta_vs_baseline",
        ]
        out = pd.DataFrame([dict(zip(cols, [np.nan] * len(cols)))])
        base = os.path.basename(final_csv_path).replace(
            "_final_metrics_summary.csv", ""
        )
        best_csv = os.path.join(
            os.path.dirname(final_csv_path),
            f"{base}_best_{primary_metric}_strategy.csv",
        )
        out.to_csv(best_csv, index=False)
        print(f"[Saved] {best_csv}")
        return out, out

    # Best by final_mean
    idx = sub["final_mean"].idxmax()
    best = sub.loc[idx]
    best_strategy = str(best["strategy"])
    best_mean = float(best["final_mean"]) if pd.notna(best["final_mean"]) else np.nan
    best_std = float(best["final_std"]) if pd.notna(best["final_std"]) else np.nan
    subgroup = best.get("subgroup", "")
    value = best.get("value", "")

    # Optional delta vs baseline
    delta = np.nan
    if baseline_for_delta is not None:
        base_rows = sub[sub["strategy"].astype(str) == str(baseline_for_delta)]
        if not base_rows.empty:
            base_mean = pd.to_numeric(base_rows["final_mean"], errors="coerce").iloc[0]
            if pd.notna(base_mean):
                delta = float(best_mean - base_mean)

    out_df = pd.DataFrame(
        [
            {
                "subgroup": subgroup,
                "value": value,
                "primary_metric": primary_metric,
                "metric": primary_metric,
                "best_strategy": best_strategy,
                "best_mean": best_mean,
                "best_std": best_std,
                "delta_vs_baseline": delta,
            }
        ]
    )

    base = os.path.basename(final_csv_path).replace("_final_metrics_summary.csv", "")
    best_csv = os.path.join(
        os.path.dirname(final_csv_path), f"{base}_best_{primary_metric}_strategy.csv"
    )
    out_df.to_csv(best_csv, index=False)
    print(f"[Saved] {best_csv}")

    return out_df, out_df


# ===================== Core aggregation =====================
def compute_global_thresholds(
    big: pd.DataFrame, target_fpr: float = 0.10
) -> pd.DataFrame:
    """For each (strategy, run, iteration), compute threshold at FPR<=target with max TPR."""
    rows = []
    grp = big.dropna(subset=["y_true", "proba", "iteration"]).groupby(
        ["strategy_label", "run_seed", "iteration"], sort=True
    )
    for (strategy, run_seed, it), d in grp:
        thr, fpr_at_thr, tpr_at_thr = _select_thr_at_fpr(
            d["y_true"].values, d["proba"].values, target_fpr
        )
        rows.append(
            dict(
                strategy_label=str(strategy),
                run_seed=int(run_seed),
                iteration=int(it),
                thr_global_at_fpr0p1=float(thr),
                fpr_at_thr=float(fpr_at_thr),
                tpr_at_thr=float(tpr_at_thr),
                n=len(d),
            )
        )
    return pd.DataFrame(rows)


def collect_runs_for_subgroup(
    df: pd.DataFrame,
    subgroup_col: str,
    thr_table: pd.DataFrame,
) -> Dict[str, Dict[str, Dict[str, List[List[float]]]]]:
    """
    Build nested dict: value -> metric -> strategy -> list_of_runs(metric_sequence_over_iterations).
    Use the FIXED per-iteration threshold from thr_table for F1/TPR/PPV/MSE.
    """
    out: Dict[str, Dict[str, Dict[str, List[List[float]]]]] = {}
    if subgroup_col not in df.columns:
        return out

    df = df.copy()
    if "strategy_label" not in df.columns:
        df["strategy_label"] = "unknown"
    if "run_seed" not in df.columns:
        df["run_seed"] = 0

    df = df.dropna(subset=["y_true", "proba", "iteration"])
    df["iteration"] = pd.to_numeric(df["iteration"], errors="coerce").astype(int)
    df = ensure_age_group(df)

    # Merge per-(strategy,run,iteration) threshold
    key_cols = ["strategy_label", "run_seed", "iteration"]
    df = df.merge(
        thr_table[key_cols + ["thr_global_at_fpr0p1"]], on=key_cols, how="left"
    )

    for (strategy, run_seed), dfr in df.groupby(["strategy_label", "run_seed"]):
        dfr = dfr.sort_values("iteration")
        for sg_val, df_sg in dfr.groupby(subgroup_col, dropna=False, observed=True):
            key_val = str(sg_val)
            if key_val not in out:
                out[key_val] = {m: {} for m in METRIC_LABELS.keys()}
            seqs = {m: [] for m in METRIC_LABELS.keys()}

            for it, df_it in df_sg.groupby("iteration"):
                if len(df_it) <= 0:
                    for m in seqs:
                        seqs[m].append(np.nan)
                    continue
                y = df_it["y_true"].astype(int).to_numpy()
                p = df_it["proba"].astype(float).to_numpy()
                thr = df_it["thr_global_at_fpr0p1"].iloc[0]
                res = _cls_metrics_with_fixed_thr(y, p, thr)
                for m in seqs:
                    seqs[m].append(res[m])

            for m in METRIC_LABELS.keys():
                out[key_val][m].setdefault(strategy, [])
                out[key_val][m][strategy].append(seqs[m])

    return out


def compute_and_save_rd(
    res: Dict[str, Dict[str, Dict[str, List[List[float]]]]],
    subgroup_name: str,
    out_dir: str,
    metrics_for_rd=("AUC", "F1", "TPR", "PPV", "MSE"),
):
    """Compute RD=max/min and abs-gap across subgroup values vs iteration; save one CSV per subgroup."""
    rows = []
    values = list(res.keys())
    if not values:
        return

    for metric in metrics_for_rd:
        # all strategies that appear in any subgroup value
        strategies = set()
        for v in values:
            strategies.update(res[v].get(metric, {}).keys())
        for strategy in sorted(strategies):
            # determine global T
            max_T = 0
            for v in values:
                runs = res[v].get(metric, {}).get(strategy, [])
                if runs:
                    max_T = max(max_T, max(len(r) for r in runs if len(r) > 0))
            if max_T == 0:
                continue
            # matrix [n_values × T] of mean over runs for each value
            mat = []
            for v in values:
                runs = res[v].get(metric, {}).get(strategy, [])
                A = _pad_to_len(runs, max_T)
                mat.append(
                    [np.nan] * max_T if A.size == 0 else list(np.nanmean(A, axis=0))
                )
            M = np.array(mat, dtype=float)  # [V,T]
            rd = np.nanmax(M, axis=0) / np.clip(np.nanmin(M, axis=0), 1e-12, None)
            gap = np.nanmax(M, axis=0) - np.nanmin(M, axis=0)
            for t in range(max_T):
                rows.append(
                    dict(
                        subgroup=subgroup_name,
                        metric=metric,
                        strategy=strategy,
                        iteration=t,
                        rd=float(rd[t]),
                        abs_gap=float(gap[t]),
                    )
                )

    rd_df = pd.DataFrame(rows)
    out_csv = os.path.join(
        out_dir, subgroup_name, f"RD_over_iterations_{subgroup_name}.csv"
    )
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    rd_df.to_csv(out_csv, index=False)
    print(f"[Saved] {out_csv}")


# ===================== Main =====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proba_root", type=str, default="outputs/probabilities")
    ap.add_argument("--out_dir", type=str, default="outputs/agg_subgroup")
    ap.add_argument(
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
    ap.add_argument(
        "--subgroups",
        nargs="+",
        default=["age_group", "SEX", "RACE", "HISPANIC", "smoking_status"],
    )
    ap.add_argument("--ncols", type=int, default=3)
    ap.add_argument("--target_fpr", type=float, default=0.10)
    ap.add_argument("--no_right_labels", action="store_true")
    ap.add_argument(
        "--primary_metric",
        type=str,
        default="AUC",
        choices=list(METRIC_LABELS.keys()),
        help="Metric used to pick the best strategy per subgroup value.",
    )
    ap.add_argument(
        "--baseline_for_delta",
        type=str,
        default="S_star_baseline",
        help="Baseline strategy for computing delta to the best.",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Read all runs
    dfs = []
    for strat in args.strategies:
        patt = os.path.join(args.proba_root, strat, "run_*")
        for run_dir in sorted(glob.glob(patt)):
            df = _read_one_run_df(run_dir)
            if df is None or df.empty:
                continue
            if "strategy_label" not in df.columns:
                df["strategy_label"] = strat
            if "run_seed" not in df.columns:
                m = re.search(r"run_(\d+)", run_dir)
                df["run_seed"] = int(m.group(1)) if m else 0
            df["strategy_label"] = df["strategy_label"].astype(str)
            dfs.append(df)

    if not dfs:
        print("[ERROR] No probability files found. Check --proba_root.")
        return

    big = pd.concat(dfs, ignore_index=True)
    for c in ["iteration", "num_labeled", "run_seed"]:
        if c in big.columns:
            big[c] = pd.to_numeric(big[c], errors="coerce")
    big = big.dropna(subset=["iteration", "y_true", "proba"])
    big["iteration"] = big["iteration"].astype(int)
    big = ensure_age_group(big)

    # 2) Compute global thresholds per (strategy, run, iteration)
    thr_df = compute_global_thresholds(big, target_fpr=args.target_fpr)
    thr_csv = os.path.join(
        args.out_dir, f"global_thresholds_{time.strftime('%Y%m%d-%H%M%S')}.csv"
    )
    thr_df.to_csv(thr_csv, index=False)
    print(f"[Saved] {thr_csv}")

    # 3) Process each subgroup column
    for sg in args.subgroups:
        if sg not in big.columns:
            print(f"[WARN] Subgroup column '{sg}' not found; skipping.")
            continue

        sub_df = big[big["strategy_label"].isin(args.strategies)].copy()
        res = collect_runs_for_subgroup(sub_df, subgroup_col=sg, thr_table=thr_df)

        # 3a) RD across subgroup values
        compute_and_save_rd(res, subgroup_name=sg, out_dir=args.out_dir)

        # 3b) For each subgroup value: plot + final summary + best strategy
        best_counter = {}  # metric -> {strategy: count}
        for sg_val, metric_dict in res.items():
            safe_val = _sanitize_filename(sg_val)
            fig_path = os.path.join(
                args.out_dir,
                sg,
                f"{safe_val}_metrics_{time.strftime('%Y%m%d-%H%M%S')}.png",
            )
            plot_facets_per_subgroup(
                subgroup_name=sg,
                subgroup_value=str(sg_val),
                metric_runs=metric_dict,
                out_path=fig_path,
                strategy_colors=DEFAULT_STRATEGY_COLORS,
                ncols=args.ncols,
                right_labels=not args.no_right_labels,
            )

            # Final-iteration summary per metric × strategy
            final_rows = []
            for metric_name, strat_runs in metric_dict.items():
                for strategy, runs in strat_runs.items():
                    finals = np.array(
                        [final_value_per_run(r) for r in runs], dtype=float
                    )
                    if finals.size == 0 or np.all(np.isnan(finals)):
                        m, s, n = np.nan, np.nan, len(finals)
                    else:
                        m, s, n = (
                            float(np.nanmean(finals)),
                            float(np.nanstd(finals)),
                            int(np.sum(np.isfinite(finals))),
                        )
                    final_rows.append(
                        dict(
                            subgroup=sg,
                            value=sg_val,
                            metric=metric_name,
                            strategy=strategy,
                            final_mean=m,
                            final_std=s,
                            n_runs=n,
                        )
                    )

            out_csv = os.path.join(
                args.out_dir, sg, f"{safe_val}_final_metrics_summary.csv"
            )
            os.makedirs(os.path.dirname(out_csv), exist_ok=True)
            pd.DataFrame(final_rows).sort_values(["metric", "strategy"]).to_csv(
                out_csv, index=False
            )
            print(f"[Saved] {out_csv}")

            # Best strategy (primary metric)
            best_df, _ = summarize_best_per_subgroup_value(
                out_csv,
                primary_metric=args.primary_metric,
                strategies_all=args.strategies,
                baseline_for_delta=args.baseline_for_delta,
            )
            best_csv = os.path.join(args.out_dir, sg, f"{safe_val}_best_strategy.csv")
            if not best_df.empty:
                best_df.to_csv(best_csv, index=False)
                print(f"[Saved] {best_csv}")
                for _, r in best_df.iterrows():
                    metric = r["metric"]
                    strat = r["best_strategy"]
                    best_counter.setdefault(metric, {})
                    best_counter[metric][strat] = best_counter[metric].get(strat, 0) + 1

        # 3c) Overview: how often each strategy is best across values
        if best_counter:
            rows_overview = []
            n_values = len(res.keys())
            for metric, cnts in best_counter.items():
                total = sum(cnts.values())
                for strat, nbest in sorted(cnts.items(), key=lambda x: (-x[1], x[0])):
                    prop = (nbest / total) if total > 0 else np.nan
                    rows_overview.append(
                        dict(
                            subgroup=sg,
                            metric=metric,
                            strategy=strat,
                            n_best=nbest,
                            n_values=n_values,
                            prop_among_best=prop,
                        )
                    )
            ov_df = pd.DataFrame(rows_overview).sort_values(
                ["metric", "n_best"], ascending=[True, False]
            )
            ov_csv = os.path.join(args.out_dir, sg, f"BEST_overview_{sg}.csv")
            os.makedirs(os.path.dirname(ov_csv), exist_ok=True)
            ov_df.to_csv(ov_csv, index=False)
            print(f"[Saved] {ov_csv}")


if __name__ == "__main__":
    main()
