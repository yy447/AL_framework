# al/main_loop.py
import os
import time
import numpy as np
import pandas as pd
from collections import deque
from typing import Tuple, List, Dict

from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, normalize

import matplotlib.pyplot as plt

from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

from rl.agent import PPO
from data.process_data import build_realdata_arrays


# =========================
# Real data config
# =========================
DATA_PATH = r"P:\Pro00110219 - Predictive modeling using EHR\yy447\lung_cancer_data\lung_cancer_small.csv"
LABEL_COL = "lung_outcome_60d"
TIME_COL = "event_time_60d"
SMOKING_STATUS_COL = "smoking_status"
DEMO_COLS = ["age_at_index", "SEX", "RACE", "HISPANIC"]


def generate_patients_ch(
    n=1000,
    d_X1=10,
    d_X2=10,
    beta_S_scale=1.5,
    S_star_scale=2,
    beta_Y_scale_X2=0.2,
    beta_Y_scale_S=5.0,
    seed=123,
):
    """Toy generator for synthetic patients."""
    np.random.seed(seed)

    # X1 drives S_true; X2 drives Y
    X1 = np.random.normal(size=(n, d_X1))
    X2 = np.random.normal(size=(n, d_X2))

    # X1 -> S_true
    beta_S = np.random.randn(d_X1) * beta_S_scale
    s_logit = X1 @ beta_S + np.random.normal(scale=0.3, size=n)
    p_S = 1 / (1 + np.exp(-s_logit))
    S_true = np.random.binomial(1, p_S)
    auc_s = roc_auc_score(S_true, p_S)

    # S_true -> S_star (noisy proxy)
    s_star_logit = S_true + np.random.normal(scale=S_star_scale, size=n)
    S_star = 1 / (1 + np.exp(-s_star_logit))
    auc_sstar = roc_auc_score(S_true, S_star)

    # (S_true + X2) -> Y
    y_coef_s = beta_Y_scale_S
    y_coef_x2 = np.random.normal(scale=beta_Y_scale_X2, size=X2.shape[1])
    y_logit = y_coef_s * S_true + X2 @ y_coef_x2 + np.random.normal(scale=0.5, size=n)
    p_Y = 1 / (1 + np.exp(-y_logit))
    Y = np.random.binomial(1, p_Y)
    auc_y = roc_auc_score(Y, y_logit)

    return X1, X2, Y, S_true, S_star, auc_s, auc_sstar, auc_y


# ========== Utils ==========
def safe_dense(m):
    """Convert sparse to dense if needed."""
    if hasattr(m, "toarray"):
        return m.toarray()
    return np.asarray(m)


def as_dataframe(Z, prefix):
    """Convert array to DataFrame with prefix columns."""
    Z = safe_dense(Z)
    cols = [f"{prefix}_{i}" for i in range(Z.shape[1])]
    return pd.DataFrame(Z, columns=cols)


def load_sim_arrays_for_run(
    run_seed: int,
    n: int = 1000,
    d_X1: int = 10,
    d_X2: int = 10,
    beta_S_scale: float = 1.5,
    S_star_scale: float = 2.0,
    beta_Y_scale_X2: float = 0.2,
    beta_Y_scale_S: float = 5.0,
):
    """Build synthetic arrays and a simple train/val split (stratified by Y)."""
    X1, X2, Y, S_true, S_star, auc_s, auc_sstar, auc_y = generate_patients_ch(
        n=n,
        d_X1=d_X1,
        d_X2=d_X2,
        beta_S_scale=beta_S_scale,
        S_star_scale=S_star_scale,
        beta_Y_scale_X2=beta_Y_scale_X2,
        beta_Y_scale_S=beta_Y_scale_S,
        seed=run_seed,
    )

    idx_all = np.arange(len(Y))
    idx_tr, idx_val = train_test_split(
        idx_all, test_size=0.2, stratify=Y, random_state=run_seed
    )

    val_meta_df = pd.DataFrame({"y_true": Y[idx_val]})

    return {
        "X1": X1,
        "X2": X2,
        "Y": Y,
        "S_true": S_true,
        "S_star": S_star,
        "idx_tr": idx_tr,
        "idx_val": idx_val,
        "T": None,
        "val_meta_df": val_meta_df,
        "auc_s_holdout": float(auc_s),
        "auc_sstar_holdout": float(auc_sstar),
        "auc_y_ref": float(auc_y),
    }


def load_real_arrays_for_run(run_seed: int):
    """Load real dataset, split train/val, build feature arrays and quick ref AUC."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"DATA_PATH not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    y_all = df[LABEL_COL].astype(int).values
    t_all = df[TIME_COL].values if TIME_COL in df.columns else None

    idx_all = np.arange(len(df))
    idx_tr, idx_val = train_test_split(
        idx_all, test_size=0.2, stratify=y_all, random_state=run_seed
    )

    X1, X2, Y, S_true, S_star, meta = build_realdata_arrays(
        df,
        label_col=LABEL_COL,
        smoking_status_col=SMOKING_STATUS_COL,
        demo_cols=DEMO_COLS,
        idx_train=idx_tr,
        random_state=run_seed,
        verbose=False,
    )

    # quick outer AUC on (S_true + X2)
    if X2.shape[1] > 0:
        scaler_q = StandardScaler(with_mean=False).fit(X2[idx_tr])
        X2_tr_s = scaler_q.transform(X2[idx_tr])
        X2_val_s = scaler_q.transform(X2[idx_val])
    else:
        X2_tr_s = np.zeros((len(idx_tr), 0))
        X2_val_s = np.zeros((len(idx_val), 0))

    Z_tr_q = np.hstack([S_true[idx_tr].reshape(-1, 1), X2_tr_s])
    Z_val_q = np.hstack([S_true[idx_val].reshape(-1, 1), X2_val_s])
    clf_q = LogisticRegression(
        max_iter=1000, class_weight="balanced", solver="liblinear"
    )
    clf_q.fit(Z_tr_q, Y[idx_tr])
    p_val_q = clf_q.predict_proba(Z_val_q)[:, 1]
    auc_y_ref = roc_auc_score(Y[idx_val], p_val_q)

    pack = {
        "X1": X1,
        "X2": X2,
        "Y": Y,
        "S_true": S_true,
        "S_star": S_star,
        "idx_tr": idx_tr,
        "idx_val": idx_val,
        "auc_s_holdout": meta.get("auc_true_holdout", np.nan),
        "auc_sstar_holdout": meta.get("auc_star_holdout", np.nan),
        "auc_y_ref": auc_y_ref,
    }
    if t_all is not None:
        pack["T"] = t_all

    # meta for per-iteration dumps
    meta_cols = [
        "PATID",
        "age_at_index",
        "SEX",
        "RACE",
        "HISPANIC",
        "smoking_status",
        LABEL_COL,
    ]
    present = [c for c in meta_cols if c in df.columns]
    val_meta_df = df.loc[idx_val, present].copy()
    if LABEL_COL in val_meta_df.columns:
        val_meta_df = val_meta_df.rename(columns={LABEL_COL: "y_true"})
    pack["val_meta_df"] = val_meta_df.reset_index(drop=True)

    return pack


# >>> Classification metrics at FPR <= 0.1
def compute_cls_metrics_from_proba(y_true, proba, target_fpr=0.10):
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba).ravel()
    fpr, tpr, th = roc_curve(y_true, proba)
    ok = np.where(fpr <= target_fpr)[0]
    if ok.size > 0:
        i = ok[np.argmax(tpr[ok])]
        thr = float(th[i])
        fpr_at_thr = float(fpr[i])
        tpr_at_thr = float(tpr[i])
    else:
        thr = 0.5
        preds_tmp = (proba >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds_tmp, labels=[0, 1]).ravel()
        fpr_at_thr = float(fp / (fp + tn + 1e-8))
        tpr_at_thr = float(tp / (tp + fn + 1e-8))
    preds = (proba >= thr).astype(int)
    ppv = float(precision_score(y_true, preds, zero_division=0))
    rec = float(recall_score(y_true, preds, zero_division=0))
    f1 = float(f1_score(y_true, preds, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    return {
        "thr_at_fpr_0p1": thr,
        "fpr_at_thr": fpr_at_thr,
        "tpr_at_thr": rec,
        "ppv_at_thr": ppv,
        "f1_at_thr": f1,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# -------- Cox helpers --------
def _sanitize_times_for_cox(T, E, jitter_eps=1e-6):
    """Filter invalid durations and optionally jitter ties."""
    T = np.asarray(T, dtype=float)
    E = np.asarray(E, dtype=int)
    finite = np.isfinite(T)
    keep = finite & (T >= 0)
    T = T[keep].copy()
    E = E[keep].copy()
    if len(np.unique(T)) < 3 and len(T) > 10:
        ranks = pd.Series(T).rank(method="average").to_numpy()
        T += (ranks - ranks.mean()) * jitter_eps
    return T, E, keep


def _drop_sparse_columns(df_tr, min_nnz_frac=0.01):
    """Keep z_* features with enough non-zeros."""
    feats = [c for c in df_tr.columns if c.startswith("z_")]
    if not feats:
        return df_tr, feats
    n = len(df_tr)
    keep_cols = []
    for c in feats:
        x = df_tr[c].to_numpy()
        nnz = np.count_nonzero(np.asarray(x))
        if n == 0 or (nnz / n) >= min_nnz_frac:
            keep_cols.append(c)
    kept = ["time", "event"] + keep_cols
    return df_tr[kept], keep_cols


def _drop_low_variance_columns(df_tr, min_std=1e-8):
    """Keep z_* features with non-trivial variance, overall or by event group."""
    feats = [c for c in df_tr.columns if c.startswith("z_")]
    if not feats:
        return df_tr, feats
    events = df_tr["event"].astype(bool).to_numpy()
    keep_cols = []
    for c in feats:
        x = df_tr[c].to_numpy()
        gstd = np.nanstd(x)
        std1 = np.nanstd(x[events]) if events.any() else 0.0
        std0 = np.nanstd(x[~events]) if (~events).any() else 0.0
        if np.isfinite(gstd) and (
            (gstd > min_std) or (std1 > min_std) or (std0 > min_std)
        ):
            keep_cols.append(c)
    kept = ["time", "event"] + keep_cols
    return df_tr[kept], keep_cols


def _select_features_by_univariate_cox(df_tr, max_k=30):
    """Univariate Cox pre-screening to select top-k features by p-value."""
    feats = [c for c in df_tr.columns if c.startswith("z_")]
    if not feats:
        return df_tr, feats
    pvals = []
    for c in feats:
        try:
            cph_u = CoxPHFitter(penalizer=0.0)
            cph_u.fit(
                df_tr[["time", "event", c]],
                duration_col="time",
                event_col="event",
                robust=True,
                show_progress=False,
            )
            p = float(cph_u.summary.loc[c, "p"])
        except Exception:
            p = 1.0
        pvals.append((c, p))
    pvals.sort(key=lambda t: t[1])
    chosen = [c for c, _ in pvals[:max_k]]
    cols = ["time", "event"] + chosen
    return df_tr[cols], chosen


def _enough_comparable_pairs(T, E):
    """Quick check for enough comparable pairs for C-index."""
    T = np.asarray(T)
    E = np.asarray(E).astype(int)
    finite = np.isfinite(T)
    if finite.sum() < 5:
        return False
    if E[finite].sum() < 1:
        return False
    if len(np.unique(T[finite])) < 3:
        return False
    return True


# ========== State normalizer ==========
class StateNormalizer:
    """EMA-based online state standardization with clipping."""

    def __init__(
        self, state_dim, min_window_size=20, clip_range=(-5, 5), ema_alpha=0.05
    ):
        self.state_dim = state_dim
        self.min_window_size = min_window_size
        self.clip_range = clip_range
        self.ema_alpha = ema_alpha
        self.state_history = []
        self.state_mean = np.zeros(state_dim)
        self.state_std = np.ones(state_dim)

    def normalize(self, state):
        self.update_stats(state)
        normalized = (state - self.state_mean) / (self.state_std + 1e-8)
        return np.clip(normalized, self.clip_range[0], self.clip_range[1])

    def update_stats(self, state):
        self.state_history.append(state)
        if len(self.state_history) == self.min_window_size + 1:
            hist = np.array(self.state_history)
            self.state_mean = hist.mean(axis=0)
            self.state_std = hist.std(axis=0)
            self.state_std[self.state_std < 1e-6] = 1.0
        elif len(self.state_history) > self.min_window_size + 1:
            self.state_mean = (
                1 - self.ema_alpha
            ) * self.state_mean + self.ema_alpha * state
            diff = state - self.state_mean
            variance = (1 - self.ema_alpha) * (self.state_std**2) + self.ema_alpha * (
                diff**2
            )
            self.state_std = np.sqrt(np.maximum(variance, 1e-6))


# ========== Main AL/RL system ==========
class CVDALRLSystem:
    """
    mode='auc'    -> train/eval logistic only (classification metrics), C-index is NaN
    mode='cindex' -> train/eval Cox only (C-index), classification metrics are NaN
    """

    def __init__(self, data, config):
        # ---- Core data ----
        self.X1 = data["X1"]
        self.X2 = data["X2"]
        self.S_true = data["S_true"]
        self.S_star = data["S_star"]
        self.Y = data["Y"]
        self.T = data.get("T", None)
        self.val_meta_df = data.get("val_meta_df", None)

        # caches
        self._last_val_proba = None
        self._last_val_risk = None
        self._last_val_time = None
        self._last_val_event = None
        self._last_keep_va = None
        self.idx_train_ext = data.get("idx_train", None)
        self.idx_val_ext = data.get("idx_val", None)

        # ---- Config ----
        self.config = config
        self.mode = config.get("mode", "auc").lower()
        assert self.mode in ("auc", "cindex")
        self.reward_metric = self.mode

        self.budget = int(config.get("budget", 200))
        self.batch_size = int(config.get("batch_size", 40))
        self.random_state = int(config.get("random_state", 42))
        self.val_size = config.get("val_size", 0.2)
        self.initial_samples = int(config.get("initial_samples", 0))
        self.reward_horizon = int(config.get("reward_horizon", 3))
        self.reward_gamma = float(config.get("reward_gamma", 0.9))
        self.strategy = config.get("strategy", "rl")
        self.ppo_rollout = int(config.get("ppo_rollout", 5))
        self.warmup_iters = int(config.get("warmup_iters", 0))

        # Cox params
        self.cox_penalizer = float(config.get("cox_penalizer", 0.05))
        self.cox_l1_ratio = float(config.get("cox_l1_ratio", 0.9))
        self.cox_max_features = int(config.get("cox_max_features", 30))
        self.cox_min_std = float(config.get("cox_min_std", 1e-8))
        self.cox_min_nnz_frac = float(config.get("cox_min_nnz_frac", 0.01))
        self.cox_jitter_eps = float(config.get("cox_jitter_eps", 1e-6))
        self.cox_features = config.get("cox_features", "S+X2")

        # ---- History ----
        self.auc_history: List[float] = []
        self.cindex_history: List[float] = []
        self.selection_history = []
        self.iteration = 0
        self.early_stop = False
        self._last_proba_mse = np.nan
        self._last_cls_metrics = None

        # ---- RL agent or fixed strategy ----
        if self.strategy != "rl":
            if self.strategy in ["random", "uncertainty", "diversity", "qbc"]:
                self.fixed_strategy = self.strategy
            elif self.strategy == "equal":
                self.fixed_strategy = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
            else:
                self.fixed_strategy = "uncertainty"
            self.agent = None
        else:
            state_dim = 8
            self.agent = PPO(state_dim, {**config, "action_dim": 3})

        self._traj_steps = 0
        self.state_normalizer = StateNormalizer(
            state_dim=8, min_window_size=20, clip_range=(-5, 5), ema_alpha=0.05
        )

        # reward cache
        self.metric_window = deque(maxlen=self.reward_horizon + 1)
        self.reward_stats = {"mean": 0, "std": 1}

        # ---- Setup ----
        self._initialize_data_splits()
        self._initialize_models()
        self._init_log_dir()

    # ----------------- per-iteration logging -----------------
    def _init_log_dir(self):
        root = self.config.get(
            "iter_log_root", os.path.join("outputs_simulation", "iter_logs")
        )
        strat_label = self.config.get("strategy_label", self.strategy)
        run_id = f"run_{self.random_state}"
        self.log_dir = os.path.join(root, strat_label, run_id)
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_csv_path = os.path.join(self.log_dir, "per_iter_metrics.csv")
        self.sel_csv_path = os.path.join(self.log_dir, "selected_indices.csv")
        if not os.path.exists(self.sel_csv_path):
            pd.DataFrame(columns=["iteration", "indices"]).to_csv(
                self.sel_csv_path, index=False
            )
        self.proba_dir = os.path.join(
            "outputs_simulation", "probabilities", strat_label, run_id
        )
        os.makedirs(self.proba_dir, exist_ok=True)
        self.proba_all_csv = os.path.join(self.proba_dir, "all_iters.csv")
        self.risk_dir = os.path.join("outputs_simulation", "risks", strat_label, run_id)
        os.makedirs(self.risk_dir, exist_ok=True)
        self.risk_all_csv = os.path.join(self.risk_dir, "all_iters.csv")

    def _cls_metrics_from_proba(self, y_true, proba, target_fpr=0.10):
        """Same as global helper, kept local for minimal dependency."""
        y_true = np.asarray(y_true).astype(int)
        fpr, tpr, th = roc_curve(y_true, proba)
        ok = np.where(fpr <= target_fpr)[0]
        if ok.size > 0:
            i = ok[np.argmax(tpr[ok])]
            thr = float(th[i])
            fpr_at_thr = float(fpr[i])
            tpr_at_thr = float(tpr[i])
        else:
            thr = 0.5
            preds_tmp = (proba >= thr).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, preds_tmp, labels=[0, 1]).ravel()
            fpr_at_thr = float(fp / (fp + tn + 1e-8))
            tpr_at_thr = float(tp / (tp + fn + 1e-8))
        preds = (proba >= thr).astype(int)
        ppv = float(precision_score(y_true, preds, zero_division=0))
        rec = float(recall_score(y_true, preds, zero_division=0))
        f1 = float(f1_score(y_true, preds, zero_division=0))
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
        return {
            "thr_at_fpr_0.1": thr,
            "fpr_at_thr": fpr_at_thr,
            "tpr_at_thr": rec,
            "ppv_at_thr": ppv,
            "f1_at_thr": f1,
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }

    def _log_iteration(self, weights=None, selected_indices=None):
        """Write per-iteration probability/risk and summary CSV rows."""
        if (self.mode == "auc") and (self._last_val_proba is not None):
            try:
                self._dump_val_probabilities(self._last_val_proba)
            except Exception as e:
                print(f"[warn] failed to dump per-iteration probabilities: {e}")
        if (
            (self.mode == "cindex")
            and (self._last_val_risk is not None)
            and (self._last_val_time is not None)
            and (self._last_val_event is not None)
        ):
            try:
                self._dump_val_risks()
            except Exception as e:
                print(f"[warn] failed to dump per-iteration risks: {e}")

        auc_val = self.auc_history[-1] if len(self.auc_history) else np.nan
        cidx_val = self.cindex_history[-1] if len(self.cindex_history) else np.nan

        cls = self._last_cls_metrics if self.mode == "auc" else None
        row = {
            "iteration": int(self.iteration),
            "num_labeled": int(self.labeled_mask.sum()),
            "strategy_label": self.config.get("strategy_label", self.strategy),
            "strategy_internal": self.strategy,
            "auc": float(auc_val) if np.isfinite(auc_val) else np.nan,
            "cindex": float(cidx_val) if np.isfinite(cidx_val) else np.nan,
        }
        if cls is not None:
            row.update(cls)
        else:
            row.update(
                {
                    "thr_at_fpr_0.1": np.nan,
                    "fpr_at_thr": np.nan,
                    "tpr_at_thr": np.nan,
                    "ppv_at_thr": np.nan,
                    "f1_at_thr": np.nan,
                    "tn": np.nan,
                    "fp": np.nan,
                    "fn": np.nan,
                    "tp": np.nan,
                }
            )

        if (~self.labeled_mask).any():
            u = ~self.labeled_mask
            row["mse_train_unlabeled"] = float(
                np.mean((self.S_star_train[u] - self.S_true_train[u]) ** 2)
            )
        else:
            row["mse_train_unlabeled"] = np.nan

        row["mse_val"] = float(np.mean((self.S_star_val - self.S_true_val) ** 2))
        row["mse_proba_val"] = (
            float(self._last_proba_mse)
            if (self.mode == "auc" and np.isfinite(self._last_proba_mse))
            else np.nan
        )

        if weights is not None:
            w = np.asarray(weights, dtype=float).ravel()
            row.update(
                {
                    "w_uncertainty": w[0] if w.size > 0 else np.nan,
                    "w_diversity": w[1] if w.size > 1 else np.nan,
                    "w_qbc": w[2] if w.size > 2 else np.nan,
                }
            )
        else:
            row.update(
                {"w_uncertainty": np.nan, "w_diversity": np.nan, "w_qbc": np.nan}
            )

        pd.DataFrame([row]).to_csv(
            self.log_csv_path,
            index=False,
            mode=("a" if os.path.exists(self.log_csv_path) else "w"),
            header=not os.path.exists(self.log_csv_path),
        )
        if selected_indices is not None and len(selected_indices) > 0:
            pd.DataFrame(
                [
                    {
                        "iteration": int(self.iteration),
                        "indices": ",".join(
                            map(str, np.asarray(selected_indices).tolist())
                        ),
                    }
                ]
            ).to_csv(self.sel_csv_path, index=False, mode="a", header=False)

    def _dump_val_probabilities(self, proba: np.ndarray):
        """Dump per-iteration validation probabilities."""
        if self.mode != "auc":
            return
        if self.val_meta_df is not None and len(self.val_meta_df) == len(proba):
            df_meta = self.val_meta_df.copy()
            if "y_true" not in df_meta.columns:
                df_meta["y_true"] = self.Y_val
        else:
            df_meta = pd.DataFrame({"y_true": self.Y_val})

        df_out = df_meta.copy()
        df_out["proba"] = np.asarray(proba).ravel()
        df_out["iteration"] = int(self.iteration)
        df_out["num_labeled"] = int(self.labeled_mask.sum())
        df_out["strategy_label"] = self.config.get("strategy_label", self.strategy)
        df_out["run_seed"] = int(self.random_state)

        per_iter_path = os.path.join(self.proba_dir, f"iter_{self.iteration:03d}.csv")
        df_out.to_csv(per_iter_path, index=False)

        if os.path.exists(self.proba_all_csv):
            df_out.to_csv(self.proba_all_csv, index=False, mode="a", header=False)
        else:
            df_out.to_csv(self.proba_all_csv, index=False)

    def _dump_val_risks(self):
        """Dump per-iteration validation risk scores (Cox)."""
        if self.mode != "cindex":
            return
        risk = np.asarray(self._last_val_risk).ravel()
        time_arr = np.asarray(self._last_val_time).ravel()
        event_arr = np.asarray(self._last_val_event).astype(int).ravel()

        if (
            (self.val_meta_df is not None)
            and (len(self.val_meta_df) == len(self.Y_val))
            and (self._last_keep_va is not None)
        ):
            df_meta = (
                self.val_meta_df.loc[self._last_keep_va].reset_index(drop=True).copy()
            )
        else:
            df_meta = pd.DataFrame()

        df_out = df_meta.copy()
        df_out["time"] = time_arr
        df_out["event"] = event_arr
        df_out["risk"] = risk
        df_out["iteration"] = int(self.iteration)
        df_out["num_labeled"] = int(self.labeled_mask.sum())
        df_out["strategy_label"] = self.config.get("strategy_label", self.strategy)
        df_out["run_seed"] = int(self.random_state)

        per_iter_path = os.path.join(self.risk_dir, f"iter_{self.iteration:03d}.csv")
        df_out.to_csv(per_iter_path, index=False)

        if os.path.exists(self.risk_all_csv):
            df_out.to_csv(self.risk_all_csv, index=False, mode="a", header=False)
        else:
            df_out.to_csv(self.risk_all_csv, index=False)

    # ----------------- init -----------------
    def _initialize_data_splits(self):
        """Train/val split, state mirrors for train/val, scalers, optional seeding."""
        # 1) split
        if (self.idx_train_ext is not None) and (self.idx_val_ext is not None):
            self.idx_train = np.asarray(self.idx_train_ext)
            self.idx_val = np.asarray(self.idx_val_ext)
        else:
            idx = np.arange(len(self.Y))
            self.idx_train, self.idx_val = train_test_split(
                idx,
                test_size=self.val_size,
                stratify=self.Y,
                random_state=self.random_state,
            )

        # 2) slice arrays
        self.X2_train = self.X2[self.idx_train]
        self.X2_val = self.X2[self.idx_val]
        self.S_star_train = self.S_star[self.idx_train]
        self.S_true_train = self.S_true[self.idx_train]
        self.S_true_val = self.S_true[self.idx_val]
        self.S_star_val = self.S_star[self.idx_val]
        self.Y_train = self.Y[self.idx_train]
        self.Y_val = self.Y[self.idx_val]
        if self.T is not None:
            self.T_train = self.T[self.idx_train]
            self.T_val = self.T[self.idx_val]
        else:
            self.T_train = None
            self.T_val = None

        # 3) train pool state: start with S*, replace with S_true when selected
        self.labeled_mask = np.zeros(len(self.idx_train), dtype=bool)
        self.S_train = self.S_star_train.copy()

        # 4) validation mirror state: start with S*_val, replace to S_true_val
        self.val_labeled_mask = np.zeros(len(self.idx_val), dtype=bool)
        self.S_val = self.S_star_val.copy()
        self.val_budget = int(self.config.get("val_budget", self.budget))
        self.val_batch_size = int(self.config.get("val_batch_size", self.batch_size))

        # scaler on train only
        self.scaler = StandardScaler(with_mean=False).fit(self.X2_train)
        self.X2_train_scaled = self.scaler.transform(self.X2_train)
        self.X2_val_scaled = self.scaler.transform(self.X2_val)

        # optional seeding
        if self.initial_samples > 0:
            self._init_seed_samples()

    def _init_seed_samples(self):
        """Optional cold-start: label a small balanced set by S_true."""
        rng = np.random.RandomState(self.random_state)
        pos_idx_S = np.where(self.S_true_train >= 0.5)[0]
        neg_idx_S = np.where(self.S_true_train < 0.5)[0]

        n_per_class = min(self.initial_samples // 2, len(pos_idx_S), len(neg_idx_S))
        if n_per_class <= 0:
            seed_idx = rng.choice(
                np.arange(len(self.S_true_train)),
                size=min(self.initial_samples, len(self.S_true_train)),
                replace=False,
            )
        else:
            seed_idx = np.concatenate(
                [
                    rng.choice(pos_idx_S, n_per_class, replace=False),
                    rng.choice(neg_idx_S, n_per_class, replace=False),
                ]
            )

        seed_idx = np.unique(seed_idx)
        if seed_idx.size == 0:
            return

        self.labeled_mask[seed_idx] = True
        self.S_train[seed_idx] = self.S_true_train[seed_idx]

    def _initialize_models(self):
        """Init selection model and initial evaluation with current S_train."""
        self.selection_model = LogisticRegression(
            max_iter=500, class_weight="balanced", solver="liblinear"
        )
        Z_all = np.hstack([self.S_train.reshape(-1, 1), self.X2_train_scaled])
        self.selection_model.fit(Z_all, self.Y_train)

        a0, c0 = self._evaluate_current()
        self.auc_history.append(a0)
        self.cindex_history.append(c0)
        self.metric_window.append(self._reward_metric_value(a0, c0))

    # ----------------- Z builders (Cox/logit) -----------------
    def _build_Z(self, split="train", for_model="cox"):
        """Compose [S, X2] (or each alone) for train/val, depending on config."""
        feats = self.cox_features if for_model == "cox" else "S+X2"

        if split == "train":
            S = self.S_train.reshape(-1, 1)
            X = self.X2_train_scaled
        elif split == "val":
            S = self.S_val.reshape(-1, 1)  # use evolving mirror on val
            X = self.X2_val_scaled
        else:
            raise ValueError(f"Unknown split: {split}")

        if feats == "S_only":
            return S
        elif feats == "X2_only":
            return X
        else:
            return np.hstack([S, X])

    # ----------------- Evaluation (mutually exclusive paths) -----------------
    def _evaluate_current(self) -> Tuple[float, float]:
        if self.mode == "auc":
            Z_tr = self._build_Z(split="train", for_model="logit")
            Z_va = self._build_Z(split="val", for_model="logit")

            clf = LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                class_weight="balanced",
                solver="liblinear",
            )
            try:
                clf.fit(Z_tr, self.Y_train)
                proba = clf.predict_proba(Z_va)[:, 1]
                auc = float(roc_auc_score(self.Y_val, proba))
            except Exception as e:
                print(f"[WARN] LogisticRegression failed: {e}")
                auc = 0.5
                proba = np.full(len(self.Y_val), 0.5)

            self._last_cls_metrics = self._cls_metrics_from_proba(
                self.Y_val, proba, target_fpr=0.10
            )
            self._last_proba_mse = float(np.mean((proba - self.Y_val) ** 2))
            self._last_val_proba = proba
            return auc, np.nan

        # Cox / C-index path
        if (self.T_train is None) or (self.T_val is None) or (len(self.Y_train) < 5):
            self._last_cls_metrics = None
            self._last_val_risk = None
            self._last_val_time = None
            self._last_val_event = None
            self._last_keep_va = None
            return np.nan, 0.5

        Z_tr = self._build_Z(split="train", for_model="cox")
        Z_va = self._build_Z(split="val", for_model="cox")

        T_tr, E_tr, keep_tr = _sanitize_times_for_cox(
            self.T_train, self.Y_train, jitter_eps=self.cox_jitter_eps
        )
        Z_tr = Z_tr[keep_tr]
        T_va, E_va, keep_va = _sanitize_times_for_cox(
            self.T_val, self.Y_val, jitter_eps=self.cox_jitter_eps
        )
        Z_va = Z_va[keep_va]

        if not _enough_comparable_pairs(T_tr, E_tr) or not _enough_comparable_pairs(
            T_va, E_va
        ):
            self._last_cls_metrics = None
            self._last_val_risk = None
            self._last_val_time = None
            self._last_val_event = None
            self._last_keep_va = None
            return np.nan, 0.5

        df_tr = pd.DataFrame({"time": T_tr, "event": E_tr})
        df_va = pd.DataFrame({"time": T_va, "event": E_va})
        df_tr = pd.concat([df_tr, as_dataframe(Z_tr, "z")], axis=1)
        df_va = pd.concat([df_va, as_dataframe(Z_va, "z")], axis=1)

        df_tr, kept = _drop_sparse_columns(df_tr, min_nnz_frac=self.cox_min_nnz_frac)
        if not kept:
            self._last_cls_metrics = None
            self._last_val_risk = None
            self._last_val_time = None
            self._last_val_event = None
            self._last_keep_va = None
            return np.nan, 0.5
        df_va = df_va[["time", "event"] + kept]

        df_tr, kept2 = _drop_low_variance_columns(df_tr, min_std=self.cox_min_std)
        if not kept2:
            self._last_cls_metrics = None
            self._last_val_risk = None
            self._last_val_time = None
            self._last_val_event = None
            self._last_keep_va = None
            return np.nan, 0.5
        df_va = df_va[["time", "event"] + kept2]

        n_events = int(df_tr["event"].sum())
        max_k = min(self.cox_max_features, max(3, n_events // 5))
        df_tr, chosen = _select_features_by_univariate_cox(df_tr, max_k=max_k)
        df_va = df_va[["time", "event"] + chosen]

        cph = CoxPHFitter(penalizer=self.cox_penalizer, l1_ratio=self.cox_l1_ratio)
        try:
            cph.fit(
                df_tr,
                duration_col="time",
                event_col="event",
                robust=True,
                show_progress=False,
            )
            risk_scores = cph.predict_partial_hazard(df_va).values.ravel()
            c_index = concordance_index(
                df_va["time"].to_numpy(), -risk_scores, df_va["event"].to_numpy()
            )
            c_index = float(c_index) if np.isfinite(c_index) else 0.5
            self._last_val_risk = risk_scores
            self._last_val_time = df_va["time"].to_numpy()
            self._last_val_event = df_va["event"].to_numpy()
            self._last_keep_va = keep_va
        except Exception as e:
            print(f"[WARN] CoxPHFitter failed: {e}")
            c_index = 0.5
            self._last_val_risk = None
            self._last_val_time = None
            self._last_val_event = None
            self._last_keep_va = None

        self._last_cls_metrics = None
        return np.nan, c_index

    # ----------------- Common views/ranking -----------------
    def _current_pool_views(self, split="train"):
        if split == "train":
            return (
                self.S_train,
                self.X2_train_scaled,
                self.labeled_mask,
                self.budget,
                self.batch_size,
            )
        elif split == "val":
            return (
                self.S_val,
                self.X2_val_scaled,
                self.val_labeled_mask,
                self.val_budget,
                self.val_batch_size,
            )
        else:
            raise ValueError(split)

    def _rank01_unlabeled_generic(self, v, labeled_mask):
        """Rank to [0,1] on unlabeled items only."""
        v = np.asarray(v, dtype=float)
        out = np.zeros_like(v, dtype=float)
        mask = ~labeled_mask
        if mask.sum() <= 1:
            return out
        order = np.argsort(np.argsort(v[mask]))
        denom = max(1, order.size - 1)
        out[mask] = order.astype(float) / denom
        return out

    # ----------------- Scoring on train -----------------
    def _compute_uncertainty_scores(self):
        """Entropy of predicted probability on unlabeled train pool."""
        unlabeled_mask = ~self.labeled_mask
        scores = np.zeros(len(self.S_train))
        if not unlabeled_mask.any():
            return scores
        Z_unlabeled = np.hstack(
            [
                self.S_train[unlabeled_mask].reshape(-1, 1),
                self.X2_train_scaled[unlabeled_mask],
            ]
        )
        if not hasattr(self.selection_model, "classes_"):
            scores[unlabeled_mask] = 0.0
            return scores
        try:
            proba = self.selection_model.predict_proba(Z_unlabeled)[:, 1]
            proba = np.clip(proba, 1e-6, 1 - 1e-6)
            entropy = -proba * np.log(proba) - (1 - proba) * np.log(1 - proba)
            scores[unlabeled_mask] = entropy
        except Exception:
            scores[unlabeled_mask] = 0.0
        return scores

    def _compute_diversity_scores(self, k=10):
        """Avg+std cosine distance to labeled set in normalized [S, X2]."""
        scores = np.zeros(len(self.S_train))
        unlabeled_mask = ~self.labeled_mask
        if not unlabeled_mask.any():
            return scores
        if self.labeled_mask.sum() == 0:
            scores[unlabeled_mask] = 1.0
            return scores
        Z_all = np.hstack(
            [self.S_train.reshape(-1, 1), safe_dense(self.X2_train_scaled)]
        )
        Z_all = normalize(Z_all)
        Z_sel = Z_all[self.labeled_mask]
        nbrs = NearestNeighbors(
            n_neighbors=min(k, len(Z_sel)), metric="cosine", algorithm="auto"
        )
        try:
            nbrs.fit(Z_sel)
            distances, _ = nbrs.kneighbors(Z_all[unlabeled_mask])
            diversity = distances.mean(axis=1) + 0.5 * distances.std(axis=1)
            scores[unlabeled_mask] = diversity
        except Exception:
            scores[unlabeled_mask] = 0.0
        return scores

    def _compute_qbc_scores(self, n_committee=5):
        """Variance across bootstrap committee probabilities (query-by-committee)."""
        scores = np.zeros(len(self.S_train))
        unlabeled_mask = ~self.labeled_mask
        if not unlabeled_mask.any():
            return scores

        Z_tr = np.hstack([self.S_train.reshape(-1, 1), self.X2_train_scaled])
        Z_unl = np.hstack(
            [
                self.S_train[unlabeled_mask].reshape(-1, 1),
                self.X2_train_scaled[unlabeled_mask],
            ]
        )
        y_tr = self.Y_train

        probs = []
        lab_idx = np.where(self.labeled_mask)[0]
        rng = np.random.RandomState(self.random_state)
        base_idx = lab_idx if len(lab_idx) >= 3 else np.arange(len(self.Y_train))

        for b in range(n_committee):
            boot = (
                rng.choice(base_idx, size=len(base_idx), replace=True)
                if len(base_idx) > 5
                else base_idx
            )
            clf = LogisticRegression(
                max_iter=500,
                class_weight="balanced",
                solver="liblinear",
                random_state=self.random_state + 17 * b,
            )
            try:
                clf.fit(Z_tr[boot], y_tr[boot])
                p = clf.predict_proba(Z_unl)[:, 1]
            except Exception:
                p = np.full(Z_unl.shape[0], 0.5)
            probs.append(p)

        P = np.vstack(probs)  # (B, U)
        var_p = np.var(P, axis=0)
        scores[unlabeled_mask] = var_p
        return scores

    # ----------------- Scoring on a given split (mirror for val) -----------------
    def _compute_uncertainty_scores_on(self, split="val"):
        S, X, labeled_mask, _, _ = self._current_pool_views(split)
        scores = np.zeros_like(S, dtype=float)
        if (~labeled_mask).sum() == 0 or not hasattr(self.selection_model, "classes_"):
            return scores
        Z_unl = np.hstack([S[~labeled_mask].reshape(-1, 1), X[~labeled_mask]])
        try:
            proba = self.selection_model.predict_proba(Z_unl)[:, 1]
            proba = np.clip(proba, 1e-6, 1 - 1e-6)
            ent = -proba * np.log(proba) - (1 - proba) * np.log(1 - proba)
            scores[~labeled_mask] = ent
        except Exception:
            pass
        return scores

    def _compute_diversity_scores_on(self, split="val", k=10):
        from sklearn.preprocessing import normalize as _norm

        S, X, labeled_mask, _, _ = self._current_pool_views(split)
        scores = np.zeros_like(S, dtype=float)
        if (~labeled_mask).sum() == 0:
            return scores
        if labeled_mask.sum() == 0:
            scores[~labeled_mask] = 1.0
            return scores
        Z = np.hstack([S.reshape(-1, 1), safe_dense(X)])
        Z = _norm(Z)
        Z_sel = Z[labeled_mask]
        nbrs = NearestNeighbors(
            n_neighbors=min(k, len(Z_sel)), metric="cosine", algorithm="auto"
        )
        try:
            nbrs.fit(Z_sel)
            distances, _ = nbrs.kneighbors(Z[~labeled_mask])
            diversity = distances.mean(axis=1) + 0.5 * distances.std(axis=1)
            scores[~labeled_mask] = diversity
        except Exception:
            pass
        return scores

    def _compute_qbc_scores_on(self, split="val", n_committee=5):
        S, X, labeled_mask, _, _ = self._current_pool_views(split)
        scores = np.zeros_like(S, dtype=float)
        if (~labeled_mask).sum() == 0:
            return scores

        # committee is trained on train split, scored on target split
        Z_tr = np.hstack([self.S_train.reshape(-1, 1), self.X2_train_scaled])
        y_tr = self.Y_train
        Z_unl = np.hstack([S[~labeled_mask].reshape(-1, 1), X[~labeled_mask]])

        probs = []
        lab_idx = np.where(self.labeled_mask)[0]
        rng = np.random.RandomState(self.random_state)
        base_idx = lab_idx if len(lab_idx) >= 3 else np.arange(len(self.Y_train))

        for b in range(n_committee):
            boot = (
                rng.choice(base_idx, size=len(base_idx), replace=True)
                if len(base_idx) > 5
                else base_idx
            )
            clf = LogisticRegression(
                max_iter=500,
                class_weight="balanced",
                solver="liblinear",
                random_state=self.random_state + 17 * b,
            )
            try:
                clf.fit(Z_tr[boot], y_tr[boot])
                p = clf.predict_proba(Z_unl)[:, 1]
            except Exception:
                p = np.full(Z_unl.shape[0], 0.5)
            probs.append(p)

        P = np.vstack(probs) if len(probs) > 0 else np.zeros((1, Z_unl.shape[0]))
        var_p = np.var(P, axis=0)
        scores[~labeled_mask] = var_p
        return scores

    # ----------------- Selection (train + mirror for val) -----------------
    def select_samples(self, strategy_or_weights):
        """Select indices on TRAIN pool based on a single strategy or a weighted mix."""
        if isinstance(strategy_or_weights, str):
            name = strategy_or_weights.lower()
            if name == "random":
                scores = np.random.rand(len(self.S_train))
            elif name == "uncertainty":
                scores = self._compute_uncertainty_scores()
            elif name == "diversity":
                scores = self._compute_diversity_scores()
            elif name == "qbc":
                scores = self._compute_qbc_scores()
            else:
                raise ValueError(f"Unknown strategy: {strategy_or_weights}")

            s_rank = self._rank01_unlabeled_generic(scores, self.labeled_mask)
            s_rank[self.labeled_mask] = -np.inf
            remaining = max(0, self.budget - int(self.labeled_mask.sum()))
            k = min(self.batch_size, (~self.labeled_mask).sum(), remaining)
            if k <= 0:
                return np.array([], dtype=int)
            idx_unlab = np.where(~self.labeled_mask)[0]
            eps = 1e-9
            rng = np.random.RandomState(self.random_state + 12345 + self.iteration)
            s_rank[idx_unlab] = s_rank[idx_unlab] + eps * rng.rand(idx_unlab.size)
            return idx_unlab[np.argsort(-s_rank[idx_unlab])[:k]]

        # weighted mixture of [uncertainty, diversity, qbc]
        w = np.asarray(strategy_or_weights, dtype=float).ravel()
        if w.size != 3:
            raise ValueError("weights must be length-3 [uncertainty, diversity, qbc]")
        u_r = self._rank01_unlabeled_generic(
            self._compute_uncertainty_scores(), self.labeled_mask
        )
        d_r = self._rank01_unlabeled_generic(
            self._compute_diversity_scores(), self.labeled_mask
        )
        q_r = self._rank01_unlabeled_generic(
            self._compute_qbc_scores(), self.labeled_mask
        )

        w = np.clip(w, 1e-8, None)
        w = w / (w.sum() + 1e-8)
        combined = w[0] * u_r + w[1] * d_r + w[2] * q_r
        combined[self.labeled_mask] = -np.inf
        remaining = max(0, self.budget - int(self.labeled_mask.sum()))
        k = min(self.batch_size, (~self.labeled_mask).sum(), remaining)
        if k <= 0:
            return np.array([], dtype=int)
        eps = 1e-9
        rng = np.random.RandomState(self.random_state + 23456 + self.iteration)
        idx_unlab = np.where(~self.labeled_mask)[0]
        combined[idx_unlab] = combined[idx_unlab] + eps * rng.rand(idx_unlab.size)
        return idx_unlab[np.argsort(-combined[idx_unlab])[:k]]

    def select_samples_on(self, split, strategy_or_weights):
        """Select indices on given split (train/val). Used to mirror val labeling."""
        S, X, labeled_mask, budget, batch_size = self._current_pool_views(split)

        if isinstance(strategy_or_weights, str):
            name = strategy_or_weights.lower()
            if name == "random":
                scores = np.random.rand(len(S))
            elif name == "uncertainty":
                scores = self._compute_uncertainty_scores_on(split)
            elif name == "diversity":
                scores = self._compute_diversity_scores_on(split)
            elif name == "qbc":
                scores = self._compute_qbc_scores_on(split)
            else:
                raise ValueError(f"Unknown strategy: {strategy_or_weights}")
            s_rank = self._rank01_unlabeled_generic(scores, labeled_mask)
        else:
            w = np.asarray(strategy_or_weights, dtype=float).ravel()
            if w.size != 3:
                raise ValueError(
                    "weights must be length-3 [uncertainty, diversity, qbc]"
                )
            u_r = self._rank01_unlabeled_generic(
                self._compute_uncertainty_scores_on(split), labeled_mask
            )
            d_r = self._rank01_unlabeled_generic(
                self._compute_diversity_scores_on(split), labeled_mask
            )
            q_r = self._rank01_unlabeled_generic(
                self._compute_qbc_scores_on(split), labeled_mask
            )
            w = np.clip(w, 1e-8, None)
            w = w / (w.sum() + 1e-8)
            s_rank = w[0] * u_r + w[1] * d_r + w[2] * q_r

        s_rank[labeled_mask] = -np.inf
        remaining = max(0, budget - int(labeled_mask.sum()))
        k = min(batch_size, (~labeled_mask).sum(), remaining)
        if k <= 0:
            return np.array([], dtype=int)
        idx_unlab = np.where(~labeled_mask)[0]
        eps = 1e-9
        rng = np.random.RandomState(self.random_state + 34567 + self.iteration)
        s_rank[idx_unlab] = s_rank[idx_unlab] + eps * rng.rand(idx_unlab.size)
        return idx_unlab[np.argsort(-s_rank[idx_unlab])[:k]]

    # ------- Reward helpers -------
    def _reward_metric_value(self, auc, cindex):
        return float(auc) if self.reward_metric == "auc" else float(cindex)

    def _reward_history(self):
        return self.auc_history if self.reward_metric == "auc" else self.cindex_history

    def _compute_long_term_reward(self, new_value):
        """Relative gain vs. moving baseline, with progress/trend factors."""
        baseline = np.mean(self.metric_window) if self.metric_window else 0.5
        relative_gain = (new_value - baseline) / (1.0 - baseline + 1e-8)
        progress = self.labeled_mask.sum() / self.budget
        progress_factor = 1.0 + 2.0 * progress
        trend_factor = 1.0
        if len(self.metric_window) > 3:
            recent = np.array(list(self.metric_window)[-3:] + [new_value])
            if recent[-1] > recent[0]:
                trend_factor = 1.2
        total_reward = relative_gain * progress_factor * trend_factor
        self.reward_stats["mean"] = 0.9 * self.reward_stats["mean"] + 0.1 * total_reward
        self.reward_stats["std"] = 0.9 * self.reward_stats["std"] + 0.1 * abs(
            total_reward
        )
        normalized_reward = (total_reward - self.reward_stats["mean"]) / (
            self.reward_stats["std"] + 1e-8
        )
        print(
            f"[REWARD] value={new_value:.4f}, base={baseline:.4f}, gain={relative_gain:.4f}, "
            f"progress={progress:.2f}, reward={total_reward:.4f}, norm={normalized_reward:.4f}"
        )
        self.metric_window.append(new_value)
        return float(normalized_reward)

    # ----------------- Update models -----------------
    def update_models(self, selected_indices):
        """Apply new labels on train, refit selection model, evaluate."""
        if len(selected_indices) > 0:
            self.S_train[selected_indices] = self.S_true_train[selected_indices]
            self.labeled_mask[selected_indices] = True

        Z_train = np.hstack([self.S_train.reshape(-1, 1), self.X2_train_scaled])
        try:
            self.selection_model.fit(Z_train, self.Y_train)
        except Exception as e:
            print(f"[WARN] Selection model update failed: {e}")

        a, c = self._evaluate_current()
        self.auc_history.append(a)
        self.cindex_history.append(c)
        return a, c

    # ----------------- One AL epoch -----------------
    def run_al_epoch(self):
        """One active learning step including val mirror and logging."""
        if self.early_stop or self.labeled_mask.sum() >= self.budget:
            return None

        # warm-up for RL (optional): use diversity for a few steps
        if (self.strategy == "rl") and (self.iteration < self.warmup_iters):
            selected_tr = self.select_samples("diversity")
            selected_va = self.select_samples_on("val", "diversity")
            if len(selected_va) > 0:
                self.S_val[selected_va] = self.S_true_val[selected_va]
                self.val_labeled_mask[selected_va] = True

            a, c = self.update_models(selected_tr)
            new_val = self._reward_metric_value(a, c)
            reward = float(
                new_val - (self.metric_window[-1] if self.metric_window else 0.5)
            )
            self.metric_window.append(new_val)

            self.iteration += 1
            self._check_early_stopping()
            self._log_iteration(weights=None, selected_indices=selected_tr)
            return {
                "iteration": self.iteration,
                "auc": a,
                "cindex": c,
                "reward": reward,
                "strategy_weights": np.array([0, 1, 0], dtype=float),
                "num_labeled": self.labeled_mask.sum(),
            }

        # fixed strategies (non-RL)
        if self.strategy != "rl":
            selected_tr = self.select_samples(self.strategy)
            selected_va = self.select_samples_on("val", self.strategy)
            if len(selected_va) > 0:
                self.S_val[selected_va] = self.S_true_val[selected_va]
                self.val_labeled_mask[selected_va] = True

            old_val = (
                self._reward_metric_value(self.auc_history[-1], self.cindex_history[-1])
                if (len(self.auc_history) > 0 and len(self.cindex_history) > 0)
                else self._reward_metric_value(0.5, 0.5)
            )
            a, c = self.update_models(selected_tr)
            new_val = self._reward_metric_value(a, c)
            reward = float(new_val - old_val)
            self.metric_window.append(new_val)

            self.iteration += 1
            self._check_early_stopping()
            self._log_iteration(weights=None, selected_indices=selected_tr)
            return {
                "iteration": self.iteration,
                "auc": a,
                "cindex": c,
                "reward": reward,
                "strategy_weights": None,
                "num_labeled": self.labeled_mask.sum(),
            }

        # RL strategy
        state = self._get_current_state()
        state = np.nan_to_num(state, nan=0.0)
        weights = np.asarray(self.agent.select_action(state), dtype=float).ravel()
        if weights.size != 3:
            weights = (
                np.pad(weights, (0, 3 - weights.size))
                if weights.size < 3
                else weights[:3]
            )
        weights = np.clip(weights, 1e-8, None)
        weights = weights**2.0
        weights = weights / (weights.sum() + 1e-8)
        if np.any(~np.isfinite(weights)):
            weights = np.array([0.5, 0.4, 0.1], dtype=float)

        selected_tr = self.select_samples(weights)
        selected_va = self.select_samples_on("val", weights)
        if len(selected_va) > 0:
            self.S_val[selected_va] = self.S_true_val[selected_va]
            self.val_labeled_mask[selected_va] = True

        a, c = self.update_models(selected_tr)
        new_val = self._reward_metric_value(a, c)
        reward = float(self._compute_long_term_reward(new_val))

        done = self.early_stop or (self.labeled_mask.sum() >= self.budget)
        self.agent.buffer.rewards.append(reward)
        self.agent.buffer.is_terminals.append(done)

        self._traj_steps += 1
        if self._traj_steps >= self.ppo_rollout or done:
            self.agent.update()
            self._traj_steps = 0

        self.iteration += 1
        self._check_early_stopping()
        self._log_iteration(weights=weights, selected_indices=selected_tr)

        return {
            "iteration": self.iteration,
            "auc": a,
            "cindex": c,
            "reward": reward,
            "strategy_weights": weights,
            "num_labeled": self.labeled_mask.sum(),
        }

    # ----------------- Helpers -----------------
    def _get_current_state(self):
        """Build an 8-dim state for RL: stats of scores, class balance, trend, budget left."""
        unl = ~self.labeled_mask
        ent = self._compute_uncertainty_scores()[unl]
        div = self._compute_diversity_scores()[unl]
        qbc = self._compute_qbc_scores()[unl]

        q50_ent = float(np.median(ent)) if ent.size else 0.0
        q80_ent = float(np.quantile(ent, 0.8)) if ent.size else 0.0
        mean_div = float(np.mean(div)) if div.size else 0.0
        q80_qbc = float(np.quantile(qbc, 0.8)) if qbc.size else 0.0

        pos_rate = (
            float((self.S_train[self.labeled_mask] >= 0.5).mean())
            if self.labeled_mask.any()
            else 0.5
        )

        hist = self._reward_history()
        slope, var_m = 0.0, 0.0
        if len(hist) >= 5:
            y = np.array(hist[-5:])
            x = np.arange(len(y))
            slope = float(np.polyfit(x, y, 1)[0])
            var_m = float(np.var(y))

        budget_left = 1.0 - self.labeled_mask.sum() / max(1, self.budget)
        state = np.array(
            [q50_ent, q80_ent, mean_div, q80_qbc, pos_rate, slope, var_m, budget_left]
        )
        return self.state_normalizer.normalize(state)

    def _check_early_stopping(self, patience=15, min_improvement=0.001):
        """Stop when progress plateaus late in budget."""
        hist = self._reward_history()
        progress = self.labeled_mask.sum() / self.budget
        if progress < 0.5 or len(hist) < patience + 5:
            return
        recent = hist[-patience:]
        if len(recent) < 2:
            return
        avg_improve = (recent[-1] - recent[0]) / len(recent)
        adaptive_th = min_improvement * (1.0 + 2.0 * progress)
        if avg_improve <= adaptive_th:
            print(
                f"Early stopping: improvement {avg_improve:.6f} < threshold {adaptive_th:.6f}"
            )
            self.early_stop = True

    def full_training_loop(self):
        """Run AL until budget or early stopping."""
        results = []
        start = time.time()
        print(f"Starting Active Learning Training... (mode={self.mode})")
        while not self.early_stop and self.labeled_mask.sum() < self.budget:
            r = self.run_al_epoch()
            if r is None:
                break
            results.append(r)
            metric_str = (
                f"AUC={r['auc']:.4f}"
                if self.mode == "auc"
                else f"C-index={r['cindex']:.4f}"
            )
            sw = r["strategy_weights"]
            sw_str = "None" if sw is None else np.round(sw, 3)
            print(
                f"Iter {r['iteration']:03d}: {metric_str} | "
                f"Reward={r['reward']:.4f} | Strategy={sw_str} | "
                f"Labeled={r['num_labeled']}/{self.budget} | "
            )
        dur = time.time() - start
        print(f"Training completed in {dur:.2f} seconds")
        return results


# =========================
# Experiment entry
# =========================
def main():
    strategies = [
        "S_star_baseline",
        "S_true_oracle",
        "random",
        "uncertainty",
        "diversity",
        "qbc",
        "hybrid_rl",
    ]

    strategy_colors = {
        "random": "C0",
        "uncertainty": "C1",
        "diversity": "C2",
        "qbc": "C7",
        "hybrid_rl": "C3",
        "S_star_baseline": "C4",
        "S_true_oracle": "C5",
    }

    rl_config = {
        "mode": "auc",  # 'auc' or 'cindex'
        "budget": 300,  # total label budget on train pool
        "batch_size": 15,  # max labels per AL iteration
        "val_size": 0.2,
        "reward_horizon": 5,
        "reward_gamma": 0.85,
        "lr_actor": 0.0001,
        "lr_critic": 0.0005,
        "K_epochs": 4,
        "entropy_coef": 0.01,
        "initial_samples": 0,
        "ppo_rollout": 16,
        "action_dim": 3,
        "warmup_iters": 0,
        # Cox
        "cox_penalizer": 0.05,
        "cox_l1_ratio": 0.9,
        "cox_max_features": 30,
        "cox_min_std": 1e-8,
        "cox_min_nnz_frac": 0.05,
        "cox_jitter_eps": 1e-6,
        "cox_features": "S+X2",
    }

    n_runs = 100
    strategy_auc_histories: Dict[str, List[List[float]]] = {s: [] for s in strategies}
    strategy_cidx_histories: Dict[str, List[List[float]]] = {s: [] for s in strategies}
    auc_s_list, auc_sstar_list, auc_y_list = [], [], []

    start_time = time.time()
    mode = rl_config["mode"]

    for strategy in strategies:
        print(f"\n== Running strategy: {strategy} | mode={mode} ==")
        all_auc_runs, all_cidx_runs = [], []

        for run in range(n_runs):
            pack = load_sim_arrays_for_run(run)
            X1, X2, Y = pack["X1"], pack["X2"], pack["Y"]
            S_true, S_star = pack["S_true"], pack["S_star"]
            idx_train, idx_val = pack["idx_tr"], pack["idx_val"]
            T = pack.get("T", None)

            auc_s_list.append(pack["auc_s_holdout"])
            auc_sstar_list.append(pack["auc_sstar_holdout"])
            auc_y_list.append(pack["auc_y_ref"])

            # standardize X2 for baselines
            if X2.shape[1] > 0:
                scaler = StandardScaler(with_mean=False).fit(X2[idx_train])
                X2_train_s = scaler.transform(X2[idx_train])
                X2_val_s = scaler.transform(X2[idx_val])
            else:
                X2_train_s = np.zeros((len(idx_train), 0))
                X2_val_s = np.zeros((len(idx_val), 0))

            # ====== baselines (no AL) ======
            if strategy in ["S_star_baseline", "S_true_oracle"]:
                S_used_train = (
                    S_star[idx_train]
                    if strategy == "S_star_baseline"
                    else S_true[idx_train]
                )
                S_used_val = (
                    S_star[idx_val]
                    if strategy == "S_star_baseline"
                    else S_true[idx_val]
                )
                Z_train = np.hstack([S_used_train.reshape(-1, 1), X2_train_s])
                Z_val = np.hstack([S_used_val.reshape(-1, 1), X2_val_s])

                if mode == "auc":
                    try:
                        model = LogisticRegression(
                            max_iter=1000, class_weight="balanced", solver="liblinear"
                        )
                        model.fit(Z_train, Y[idx_train])
                        proba_val = model.predict_proba(Z_val)[:, 1]
                        auc_val = float(roc_auc_score(Y[idx_val], proba_val))
                    except Exception:
                        proba_val = np.full(len(idx_val), 0.5, dtype=float)
                        auc_val = 0.5
                    cidx_val = np.nan

                    val_meta_df = pack.get("val_meta_df", None)
                    proba_dir = os.path.join(
                        "outputs_simulation", "probabilities", strategy, f"run_{run}"
                    )
                    os.makedirs(proba_dir, exist_ok=True)

                    if (val_meta_df is not None) and (
                        len(val_meta_df) == len(proba_val)
                    ):
                        df_meta = val_meta_df.copy()
                        if "y_true" not in df_meta.columns:
                            df_meta["y_true"] = Y[idx_val]
                    else:
                        df_meta = pd.DataFrame({"y_true": Y[idx_val]})

                    df_out = df_meta.copy()
                    df_out["proba"] = np.asarray(proba_val).ravel()
                    df_out["iteration"] = 0
                    df_out["num_labeled"] = 0
                    df_out["strategy_label"] = strategy
                    df_out["run_seed"] = int(run)
                    per_iter_path = os.path.join(proba_dir, "iter_000.csv")
                    df_out.to_csv(per_iter_path, index=False)
                    all_csv = os.path.join(proba_dir, "all_iters.csv")
                    if os.path.exists(all_csv):
                        df_out.to_csv(all_csv, index=False, mode="a", header=False)
                    else:
                        df_out.to_csv(all_csv, index=False)

                    cls = compute_cls_metrics_from_proba(
                        Y[idx_val], proba_val, target_fpr=0.10
                    )
                    mse_val = float(np.mean((S_star[idx_val] - S_true[idx_val]) ** 2))
                    mse_proba_val = float(np.mean((proba_val - Y[idx_val]) ** 2))

                    log_dir = os.path.join(
                        "outputs_simulation", "iter_logs", strategy, f"run_{run}"
                    )
                    os.makedirs(log_dir, exist_ok=True)
                    log_csv = os.path.join(log_dir, "per_iter_metrics.csv")
                    row = {
                        "iteration": 0,
                        "num_labeled": 0,
                        "strategy_label": strategy,
                        "strategy_internal": strategy,
                        "auc": auc_val if np.isfinite(auc_val) else np.nan,
                        "cindex": np.nan,
                        "thr_at_fpr_0p1": cls["thr_at_fpr_0p1"],
                        "fpr_at_thr": cls["fpr_at_thr"],
                        "tpr_at_thr": cls["tpr_at_thr"],
                        "ppv_at_thr": cls["ppv_at_thr"],
                        "f1_at_thr": cls["f1_at_thr"],
                        "tn": cls["tn"],
                        "fp": cls["fp"],
                        "fn": cls["fn"],
                        "tp": cls["tp"],
                        "mse_val": mse_val,
                        "mse_proba_val": mse_proba_val,
                        "w_uncertainty": np.nan,
                        "w_diversity": np.nan,
                        "w_qbc": np.nan,
                    }
                    pd.DataFrame([row]).to_csv(log_csv, index=False)

                else:
                    # cindex baseline
                    if T is not None:
                        if rl_config["cox_features"] == "S_only":
                            Z_train_cox = S_used_train.reshape(-1, 1)
                            Z_val_cox = S_used_val.reshape(-1, 1)
                        elif rl_config["cox_features"] == "X2_only":
                            Z_train_cox = X2_train_s
                            Z_val_cox = X2_val_s
                        else:
                            Z_train_cox = Z_train
                            Z_val_cox = Z_val

                        T_tr, E_tr, keep_tr = _sanitize_times_for_cox(
                            T[idx_train],
                            Y[idx_train],
                            jitter_eps=rl_config["cox_jitter_eps"],
                        )
                        T_va, E_va, keep_va = _sanitize_times_for_cox(
                            T[idx_val],
                            Y[idx_val],
                            jitter_eps=rl_config["cox_jitter_eps"],
                        )
                        Z_train_cox = safe_dense(Z_train_cox)[keep_tr]
                        Z_val_cox = safe_dense(Z_val_cox)[keep_va]

                        if _enough_comparable_pairs(
                            T_tr, E_tr
                        ) and _enough_comparable_pairs(T_va, E_va):
                            df_tr = pd.DataFrame({"time": T_tr, "event": E_tr})
                            df_val = pd.DataFrame({"time": T_va, "event": E_va})
                            df_tr = pd.concat(
                                [df_tr, as_dataframe(Z_train_cox, "z")], axis=1
                            )
                            df_val = pd.concat(
                                [df_val, as_dataframe(Z_val_cox, "z")], axis=1
                            )

                            df_tr, kept = _drop_sparse_columns(
                                df_tr, min_nnz_frac=rl_config["cox_min_nnz_frac"]
                            )
                            if kept:
                                df_val = df_val[["time", "event"] + kept]
                                df_tr, kept2 = _drop_low_variance_columns(
                                    df_tr, min_std=rl_config["cox_min_std"]
                                )
                                if kept2:
                                    df_val = df_val[["time", "event"] + kept2]
                                    n_events = int(df_tr["event"].sum())
                                    max_k = min(
                                        rl_config["cox_max_features"],
                                        max(3, n_events // 5),
                                    )
                                    df_tr, chosen = _select_features_by_univariate_cox(
                                        df_tr, max_k=max_k
                                    )
                                    df_val = df_val[["time", "event"] + chosen]

                                    cph = CoxPHFitter(
                                        penalizer=rl_config["cox_penalizer"],
                                        l1_ratio=rl_config["cox_l1_ratio"],
                                    )
                                    try:
                                        cph.fit(
                                            df_tr,
                                            duration_col="time",
                                            event_col="event",
                                            robust=True,
                                            show_progress=False,
                                        )
                                        risk_scores = cph.predict_partial_hazard(
                                            df_val
                                        ).values.ravel()
                                        cidx_val = float(
                                            concordance_index(
                                                df_val["time"],
                                                -risk_scores,
                                                df_val["event"],
                                            )
                                        )
                                        risk_dir = os.path.join(
                                            "outputs_simulation",
                                            "risks",
                                            strategy,
                                            f"run_{run}",
                                        )
                                        os.makedirs(risk_dir, exist_ok=True)
                                        val_meta_df = pack.get("val_meta_df", None)
                                        if (val_meta_df is not None) and (
                                            len(val_meta_df) == len(idx_val)
                                        ):
                                            df_meta = (
                                                val_meta_df.loc[keep_va]
                                                .reset_index(drop=True)
                                                .copy()
                                            )
                                        else:
                                            df_meta = pd.DataFrame()
                                        df_out = df_meta.copy()
                                        df_out["time"] = df_val["time"].to_numpy()
                                        df_out["event"] = (
                                            df_val["event"].astype(int).to_numpy()
                                        )
                                        df_out["risk"] = np.asarray(risk_scores).ravel()
                                        df_out["iteration"] = 0
                                        df_out["num_labeled"] = 0
                                        df_out["strategy_label"] = strategy
                                        df_out["run_seed"] = int(run)
                                        per_iter_path = os.path.join(
                                            risk_dir, "iter_000.csv"
                                        )
                                        df_out.to_csv(per_iter_path, index=False)
                                        all_csv = os.path.join(
                                            risk_dir, "all_iters.csv"
                                        )
                                        if os.path.exists(all_csv):
                                            df_out.to_csv(
                                                all_csv,
                                                index=False,
                                                mode="a",
                                                header=False,
                                            )
                                        else:
                                            df_out.to_csv(all_csv, index=False)
                                    except Exception:
                                        cidx_val = 0.5
                                else:
                                    cidx_val = 0.5
                            else:
                                cidx_val = 0.5
                        else:
                            cidx_val = 0.5
                    else:
                        cidx_val = 0.5
                    auc_val = np.nan

                max_len = 20
                all_auc_runs.append([auc_val] * max_len)
                all_cidx_runs.append([cidx_val] * max_len)
                continue

            # ====== AL / RL system ======
            data = {
                "X1": X1,
                "X2": X2,
                "S_true": S_true,
                "S_star": S_star,
                "Y": Y,
                "idx_train": idx_train,
                "idx_val": idx_val,
            }
            if T is not None:
                data["T"] = T
            if "val_meta_df" in pack:
                data["val_meta_df"] = pack["val_meta_df"]

            config_strategy = "rl" if strategy == "hybrid_rl" else strategy

            system = CVDALRLSystem(
                data=data,
                config={
                    **rl_config,
                    "strategy": config_strategy,
                    "strategy_label": strategy,
                    "iter_log_root": os.path.join("outputs_simulation", "iter_logs"),
                    "random_state": run,
                },
            )
            _ = system.full_training_loop()
            auc_hist = system.auc_history
            cidx_hist = system.cindex_history

            max_len = max(len(auc_hist), len(cidx_hist))
            if len(auc_hist) < max_len:
                auc_hist = auc_hist + [auc_hist[-1]] * (max_len - len(auc_hist))
            if len(cidx_hist) < max_len:
                cidx_hist = cidx_hist + [cidx_hist[-1]] * (max_len - len(cidx_hist))

            all_auc_runs.append(list(auc_hist))
            all_cidx_runs.append(list(cidx_hist))

        strategy_auc_histories[strategy] = all_auc_runs
        strategy_cidx_histories[strategy] = all_cidx_runs

    print(f"\nTotal experiment time: {time.time() - start_time:.2f} seconds")

    # ---------- helper to pad ----------
    def pad_to_matrix(histories):
        """Pad per-run histories to same length by repeating the last value."""
        nonempty_lengths = [len(r) for r in histories if len(r) > 0]
        max_len = max(nonempty_lengths) if nonempty_lengths else 1
        padded = []
        for r in histories:
            if len(r) == 0:
                padded.append([np.nan] * max_len)
            elif len(r) == 1:
                padded.append([r[0]] * max_len)
            else:
                row = list(r) + [r[-1]] * (max_len - len(r))
                padded.append(row[:max_len])
        return np.array(padded, dtype=float), max_len

    os.makedirs("outputs_simulation", exist_ok=True)

    # ================= Faceted figure: metrics vs. iteration (mean ± 95% CI) =================
    strategy_histories_all = {
        "AUC": locals().get("strategy_auc_histories", {}),
        "C-index": locals().get("strategy_cidx_histories", {}),
        "F1": locals().get("strategy_f1_histories", {}),
        "TPR": locals().get("strategy_tpr_histories", {}),
        "PPV": locals().get("strategy_ppv_histories", {}),
        "MSE": locals().get("strategy_mse_histories", {}),
    }

    metrics_to_plot = []
    for m, d in strategy_histories_all.items():
        if isinstance(d, dict) and len(d) > 0:
            any_valid = False
            for _, histories in d.items():
                A, _ = pad_to_matrix(histories)
                if np.isfinite(A).any():
                    any_valid = True
                    break
            if any_valid:
                metrics_to_plot.append(m)

    if len(metrics_to_plot) > 0:
        n_panels = len(metrics_to_plot)
        ncols = min(3, n_panels)
        nrows = int(np.ceil(n_panels / ncols))

        fig, axes = plt.subplots(
            nrows, ncols, figsize=(4.0 * ncols + 2, 3.2 * nrows + 1), squeeze=False
        )
        axes = axes.ravel()

        def _plot_one_metric(ax, metric_name, hist_dict):
            for strategy, histories in hist_dict.items():
                A, _ = pad_to_matrix(histories)
                if not np.isfinite(A).any():
                    continue
                mean = np.nanmean(A, axis=0)
                std = np.nanstd(A, axis=0)
                nval = np.sum(~np.isnan(A), axis=0)
                nval = np.clip(nval, 1, None)
                ci = 1.96 * std / np.sqrt(nval)
                x = np.arange(len(mean))
                color = strategy_colors.get(strategy, None)

                ax.plot(x, mean, label=strategy, color=color, linewidth=2.0, alpha=0.95)
                ax.fill_between(x, mean - ci, mean + ci, alpha=0.2, color=color)
                ax.text(
                    x[-1] + 0.2,
                    mean[-1],
                    strategy,
                    color=color,
                    fontsize=9,
                    va="center",
                )

            ax.set_xlabel("AL Iteration")
            ax.grid(True, alpha=0.3)
            ax.set_ylabel(metric_name)
            ax.set_title(metric_name)

        for i, m in enumerate(metrics_to_plot):
            _plot_one_metric(axes[i], m, strategy_histories_all[m])

        for j in range(len(metrics_to_plot), len(axes)):
            axes[j].axis("off")

        fig.suptitle(
            "Validation metrics vs. AL iteration (mean ± 95% CI)",
            y=0.995,
            fontsize=12,
            fontweight="bold",
        )
        handles, labels = [], []
        for ax in axes[: len(metrics_to_plot)]:
            h, l = ax.get_legend_handles_labels()
            handles += h
            labels += l
        uniq, seen = [], set()
        for h, l in zip(handles, labels):
            if l not in seen:
                uniq.append((h, l))
                seen.add(l)
        if len(uniq) > 0:
            fig.legend(
                [u[0] for u in uniq],
                [u[1] for u in uniq],
                loc="lower center",
                ncol=min(6, len(uniq)),
                frameon=False,
                bbox_to_anchor=(0.5, -0.02),
            )
            plt.subplots_adjust(bottom=0.12)

        plt.tight_layout()
        os.makedirs("outputs_simulation", exist_ok=True)
        fig_path = os.path.join(
            "outputs_simulation",
            f"metrics_over_iterations_{time.strftime('%Y%m%d-%H%M%S')}.png",
        )
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"[Saved] {fig_path}")
        plt.close(fig)

    # ================= Final metrics summary (CSV + console) =================
    def _final_value_per_run(run_hist):
        """Return the last non-NaN value of a run history."""
        arr = np.array(run_hist, dtype=float)
        if arr.size == 0 or np.all(np.isnan(arr)):
            return np.nan
        idx_last = np.where(~np.isnan(arr))[0]
        return arr[idx_last[-1]] if idx_last.size > 0 else np.nan

    final_rows_all = []
    for metric_name, hist_dict in strategy_histories_all.items():
        if not isinstance(hist_dict, dict) or len(hist_dict) == 0:
            continue
        for strategy, histories in hist_dict.items():
            finals = np.array([_final_value_per_run(r) for r in histories], dtype=float)
            if finals.size == 0 or np.all(np.isnan(finals)):
                m = s = np.nan
                n_runs_eff = len(finals)
            else:
                m = float(np.nanmean(finals))
                s = float(np.nanstd(finals))
                n_runs_eff = int(np.sum(np.isfinite(finals)))
            final_rows_all.append(
                {
                    "metric": metric_name,
                    "strategy": strategy,
                    "final_mean": m,
                    "final_std": s,
                    "n_runs": n_runs_eff,
                }
            )

    final_df_all = pd.DataFrame(final_rows_all).sort_values(["metric", "strategy"])
    os.makedirs("outputs_simulation", exist_ok=True)
    final_all_csv = os.path.join("outputs_simulation", "final_metrics_summary.csv")
    final_df_all.to_csv(final_all_csv, index=False)
    print(f"[Saved] {final_all_csv}")

    if "AUC" in strategy_histories_all and len(strategy_histories_all["AUC"]) > 0:
        final_auc_df = final_df_all.query("metric == 'AUC'")[
            ["strategy", "final_mean", "final_std", "n_runs"]
        ].rename(columns={"final_mean": "final_auc_mean", "final_std": "final_auc_std"})
        final_auc_csv = os.path.join("outputs_simulation", "final_auc_summary.csv")
        final_auc_df.to_csv(final_auc_csv, index=False)
        print(f"[Saved] {final_auc_csv}")

    if ("C-index" in strategy_histories_all) and len(
        strategy_histories_all["C-index"]
    ) > 0:
        final_cidx_df = final_df_all.query("metric == 'C-index'")[
            ["strategy", "final_mean", "final_std", "n_runs"]
        ].rename(
            columns={"final_mean": "final_cindex_mean", "final_std": "final_cindex_std"}
        )
        final_cidx_csv = os.path.join("outputs_simulation", "final_cindex_summary.csv")
        final_cidx_df.to_csv(final_cidx_csv, index=False)
        print(f"[Saved] {final_cidx_csv}")

    print("\n=== Final Metrics Summary (mean ± std) ===")
    for metric_name in metrics_to_plot:
        sub = final_df_all[final_df_all["metric"] == metric_name]
        if sub.empty:
            continue
        print(f"\n-- {metric_name} --")
        for _, row in sub.iterrows():
            m = row["final_mean"]
            s = row["final_std"]
            n = int(row["n_runs"])
            if np.isnan(m):
                print(f"{row['strategy']:>18}: nan")
            else:
                print(f"{row['strategy']:>18}: {m:.4f} ± {s:.4f} (n={n})")

    print("\n=== Data Quality Summary (per-run references) ===")
    try:
        print(
            f"AUC(S_true holdout on seeds) : {np.nanmean(auc_s_list):.4f} ± {np.nanstd(auc_s_list):.4f}"
        )
        print(
            f"AUC(S_star  holdout on seeds): {np.nanmean(auc_sstar_list):.4f} ± {np.nanstd(auc_sstar_list):.4f}"
        )
        print(
            f"AUC(Y | S_true + X2, outer) : {np.nanmean(auc_y_list):.4f} ± {np.nanstd(auc_y_list):.4f}"
        )
    except Exception:
        pass


if __name__ == "__main__":
    main()
