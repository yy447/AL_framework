import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


def generate_patients(n=1000, seed=42):
    np.random.seed(seed)

    # X1: for phenotype (S_true)
    X1 = np.random.normal(size=(n, 5))

    # X2: for outcome (Y)
    X2 = np.random.normal(size=(n, 10))

    # X1 -> S_true
    beta_S = np.random.randn(5) * 1.5
    s_logit = X1 @ beta_S + np.random.normal(scale=0.3, size=n)
    p_S = 1 / (1 + np.exp(-s_logit))
    S_true = np.random.binomial(1, p_S)
    auc_s = roc_auc_score(S_true, p_S)

    # S_true -> S_star
    s_star_logit = S_true * 2.5 + np.random.normal(scale=1.0, size=n)
    S_star = 1 / (1 + np.exp(-s_star_logit))
    auc_sstar = roc_auc_score(S_true, S_star)

    # S_true + X2 -> Y
    y_coef = np.concatenate([[2.0], np.random.randn(10) * 1.2])
    y_logit = S_true * y_coef[0] + X2 @ y_coef[1:] + np.random.normal(scale=0.4, size=n)
    p_Y = 1 / (1 + np.exp(-y_logit))
    Y = np.random.binomial(1, p_Y)
    auc_y = roc_auc_score(Y, y_logit)

    print(f"[Data] X1 ~ S_true AUC: {auc_s:.3f}")
    print(f"[Data] S_true ~ S* AUC: {auc_sstar:.3f}")
    print(f"[Data] S_true + X2 ~ Y AUC: {auc_y:.3f}")

    return X1, X2, Y, S_true, S_star


class CVDALSystem:
    def __init__(
        self,
        X1,
        X2,
        Y_true,
        S_true,
        S_star,  # S_true:binary，S_star:continous
        budget=200,
        val_size=0.2,
        random_state=42,
    ):
        # Split indices into training and validation sets
        indices = np.arange(len(X1))
        self.idx_train, self.idx_val = train_test_split(
            indices, test_size=val_size, stratify=Y_true, random_state=random_state
        )

        # Store training data
        self.X2_train = X2[self.idx_train]
        self.Y_train = Y_true[self.idx_train]
        self.S_train = S_star[self.idx_train].copy()  # continuous
        self.S_true_train = S_true[self.idx_train]  # binary
        self.label_budget = 200

        # Store validation data
        self.X2_val = X2[self.idx_val]
        self.Y_val = Y_true[self.idx_val]
        self.S_true_val = S_true[self.idx_val]

        # Standardize X2 features using training set
        self.scaler = StandardScaler().fit(self.X2_train)
        self.X2_train_scaled = self.scaler.transform(self.X2_train)
        self.X2_val_scaled = self.scaler.transform(self.X2_val)

        # AL setup
        self.budget = budget
        self.labeled_mask = np.zeros(len(self.idx_train), dtype=bool)
        self._init_seed_samples(random_state)
        self.selected_features = None

        # Classifier for downstream CVD prediction
        self.cvd_model = LogisticRegression(max_iter=1000)
        self.auc_history = []

        self._retrain()

    def _init_seed_samples(self, seed):
        """Select initial seed samples with balanced class distribution"""
        rng = np.random.RandomState(seed)
        pos_idx = np.where(self.S_true_train == 1)[0]
        neg_idx = np.where(self.S_true_train == 0)[0]
        n_per_class = min(15, len(pos_idx), len(neg_idx))

        # Sample positive and negative labels
        init_idx = np.concatenate(
            [
                rng.choice(pos_idx, n_per_class, replace=False),
                rng.choice(neg_idx, n_per_class, replace=False),
            ]
        )

        self.S_train[init_idx] = self.S_true_train[init_idx]
        self.labeled_mask[init_idx] = True

    def _retrain(self):
        """Train CVD model using current S_train"""
        Z_train = np.hstack([self.S_train.reshape(-1, 1), self.X2_train_scaled])

        self.cvd_model.fit(Z_train[self.labeled_mask], self.Y_train[self.labeled_mask])

        Z_val = np.hstack([self.S_true_val.reshape(-1, 1), self.X2_val_scaled])
        val_proba = self.cvd_model.predict_proba(Z_val)[:, 1]
        self.auc_history.append(roc_auc_score(self.Y_val, val_proba))

    def compute_knn_diversity(self, X2_train_scaled, k=5):
        if self.selected_features is None or len(self.selected_features) == 0:
            return np.random.rand(len(X2_train_scaled))

        actual_k = min(k, len(self.selected_features))
        try:
            nbrs = NearestNeighbors(n_neighbors=actual_k, algorithm="kd_tree")
            nbrs.fit(self.selected_features)
            distances, _ = nbrs.kneighbors(X2_train_scaled)

            mean_dist = distances.mean(axis=1)
            std_dist = distances.std(axis=1)
            diversity = 0.6 * mean_dist + 0.4 * std_dist

            progress = len(self.selected_features) / self.budget
            diversity *= np.exp(-2 * progress)

            return (diversity - diversity.min()) / (diversity.ptp() + 1e-8)

        except Exception as e:
            print(f"多样性计算异常: {str(e)}")
            return np.random.rand(len(X2_train_scaled))

    def _compute_scores(self):
        """Calculate uncertainty, expected gain, and diversity for selection"""
        Z = np.hstack([self.S_train.reshape(-1, 1), self.X2_train_scaled])
        proba = self.cvd_model.predict_proba(Z)[:, 1]
        proba = np.clip(proba, 1e-10, 1 - 1e-10)
        # Uncertainty using entropy
        uncertainty = -proba * np.log(proba) - (1 - proba) * np.log(1 - proba)
        # AUC-based gain score (using similarity with previous batch)
        auc_gain = np.zeros_like(proba)
        if len(self.auc_history) >= 2:
            delta_auc = np.diff(self.auc_history[-2:])
            avg_gain = delta_auc.mean()
            if hasattr(self, "selection_history") and self.selection_history:
                last_batch = self.selection_history[-1]
                last_X2 = self.X2_train_scaled[np.isin(self.idx_train, last_batch)]
                similarity = self.X2_train_scaled.dot(last_X2.T).mean(axis=1)
                auc_gain = similarity * avg_gain
        # Diversity
        diversity = self.compute_knn_diversity(self.X2_train_scaled)
        return uncertainty, auc_gain, diversity

    def select_samples(self, batch_size=40):
        uncertainty, auc_gain, diversity = self._compute_scores()
        progress = self.labeled_mask.sum() / self.budget
        w_uncertainty = 0.4 * (1 - progress)
        w_diversity = 0.3 * (1 + progress)

        def safe_norm(x):
            x = np.nan_to_num(x)
            if x.ptp() < 1e-6:
                return np.zeros_like(x)
            return (x - x.min()) / x.ptp()

        u_norm = safe_norm(uncertainty)
        g_norm = safe_norm(auc_gain)
        d_norm = safe_norm(diversity)

        combined = w_uncertainty * u_norm + 0.3 * g_norm + w_diversity * d_norm

        combined[self.labeled_mask] *= 0.2

        unlabeled = np.where(~self.labeled_mask)[0]
        min_new = max(1, int(batch_size * 0.2))
        new_candidates = unlabeled[np.argsort(-combined[unlabeled])[:min_new]]
        remaining = batch_size - len(new_candidates)

        top_k = np.concatenate([new_candidates, np.argsort(-combined)[:remaining]])[
            :batch_size
        ]

        new_features = self.X2_train_scaled[top_k]
        if self.selected_features is None:
            self.selected_features = new_features
        else:
            self.selected_features = np.vstack([self.selected_features, new_features])

        return top_k

    def al_step(self):
        if self.labeled_mask.sum() >= self.budget:
            return None

        candidates = self.select_samples(40)
        actual_batch = min(len(candidates), self.budget - self.labeled_mask.sum())
        selected = candidates[:actual_batch]

        self.S_train[selected] = self.S_true_train[selected]
        self.labeled_mask[selected] = True
        if not hasattr(self, "selected_features"):
            self.selected_features = self.X2_train_scaled[selected]
        else:
            self.selected_features = np.vstack(
                [self.selected_features, self.X2_train_scaled[selected]]
            )

        self._retrain()

        print(
            f"[AL Step] Labeled: {self.labeled_mask.sum()}/{self.budget} | "
            f"AUC: {self.auc_history[-1]:.3f}"
        )
        return self.auc_history[-1]

    def visualize(self):
        """Visualize AUC"""
        plt.figure(figsize=(6, 4))
        plt.plot(self.auc_history, marker="o", c="darkred")
        plt.title("Validation AUC Progress")
        plt.xlabel("AL Iterations")
        plt.ylabel("AUC")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def visualize_summary(all_auc_histories):
    all_auc = np.array(
        [
            np.pad(
                run,
                (0, max(map(len, all_auc_histories)) - len(run)),
                constant_values=np.nan,
            )
            for run in all_auc_histories
        ]
    )

    mean_auc = np.nanmean(all_auc, axis=0)
    std_auc = np.nanstd(all_auc, axis=0)
    ci_auc = 1.96 * std_auc / np.sqrt(len(all_auc_histories))

    plt.figure(figsize=(6, 4))
    plt.plot(mean_auc, color="darkred", label="Mean AUC")
    plt.fill_between(
        np.arange(len(mean_auc)),
        mean_auc - ci_auc,
        mean_auc + ci_auc,
        color="darkred",
        alpha=0.2,
        label="95% CI",
    )
    plt.title("Validation AUC Progress")
    plt.xlabel("AL Iteration")
    plt.ylabel("AUC")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    X1, X2, Y, S_true, S_star = generate_patients(n=1000)

    all_auc_histories = []

    for run in range(100):
        X1, X2, Y, S_true, S_star = generate_patients(n=1000, seed=run)
        system = CVDALSystem(
            X1, X2, Y, S_true, S_star, budget=200, val_size=0.2, random_state=run
        )

        while system.labeled_mask.sum() < system.budget:
            system.al_step()

        all_auc_histories.append(system.auc_history)
    visualize_summary(all_auc_histories)
