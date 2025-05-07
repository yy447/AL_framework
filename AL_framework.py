import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


# data generation
def generate_patients(n=1000, seed=42):
    np.random.seed(seed)

    # Generate 5D features
    X1 = np.random.normal(size=(n, 5))

    # Generate 10D features
    X2 = np.random.normal(size=(n, 10))

    # Generate 20D CUI-like features as nonlinear projections of (X1 + X2)
    W_cui = np.random.randn(X1.shape[1] + X2.shape[1], 20)
    X_combined = np.hstack([X1, X2])
    linear_proj = X_combined @ W_cui + np.random.normal(0, 1, size=(n, 20))
    p_CUI = 1 / (1 + np.exp(-linear_proj))
    CUI = (p_CUI > 0.7).astype(int)

    # Combine X1 and CUI to generate true phenotype S
    combined_for_S = np.hstack([X1, CUI])
    beta_S = np.random.randn(combined_for_S.shape[1])
    s_logit = combined_for_S @ beta_S + np.random.normal(scale=0.5, size=n)
    p_S1 = 1 / (1 + np.exp(-s_logit))
    S_true = (p_S1 > 0.5).astype(int)

    # Generate CVD outcome Y using S and X2
    y_coef = np.concatenate([[1.5], np.random.randn(10) * 0.8])
    y_logit = S_true * y_coef[0] + X2 @ y_coef[1:] + np.random.normal(scale=0.4, size=n)
    p_Y = 1 / (1 + np.exp(-y_logit))
    Y_true = (p_Y > 0.5).astype(int)

    return X1, X2, Y_true, CUI


class CVDLearner:
    def __init__(
        self, X1, X2, Y_true, CUI, budget=200, val_size=0.2, random_state=None
    ):
        # Split training and validation sets
        indices = np.arange(len(X1))
        self.idx_train, self.idx_val = train_test_split(
            indices, test_size=val_size, stratify=Y_true, random_state=random_state
        )

        # Store training data
        self.X1_train = X1[self.idx_train]
        self.X2_train = X2[self.idx_train]
        self.Y_train = Y_true[self.idx_train]
        self.CUI_train = CUI[self.idx_train]

        # Store validation data
        self.X1_val = X1[self.idx_val]
        self.X2_val = X2[self.idx_val]
        self.Y_val = Y_true[self.idx_val]

        # Standardize X2 features using training set only
        self.scaler = StandardScaler().fit(self.X2_train)
        self.X2_train_scaled = self.scaler.transform(self.X2_train)
        self.X2_val_scaled = self.scaler.transform(self.X2_val)

        # Initialize label mask and label array
        self.labeled_mask = np.zeros(len(self.idx_train), dtype=bool)
        self.golden_labels = np.zeros(len(self.idx_train), dtype=bool)
        self._init_labeling(random_state)

        # Initialize models for phenotype and CVD prediction
        self.pheno_model = LogisticRegression(max_iter=1000, solver="lbfgs")
        self.cvd_model = LogisticRegression(max_iter=1000, solver="lbfgs")

        # History tracking
        self.auc_history = []
        self.overlap_history = []
        self.selection_history = []

        self.budget = budget
        self._retrain()

    def _init_labeling(self, random_state):
        """Initialize labeling by randomly selecting 30 samples with both classes."""
        rng = np.random.RandomState(random_state)
        # Initialize label storage and mask
        self.golden_labels = np.zeros(len(self.X1_train), dtype=bool)
        self.labeled_mask = np.zeros(len(self.X1_train), dtype=bool)
        # Keep sampling until at least 2 label classes are included
        while True:
            init_idx = rng.choice(len(self.idx_train), 30, replace=False)
            labels = self.generate_golden_labels(init_idx)
            if len(np.unique(labels)) == 2:
                break
        self.golden_labels[init_idx] = labels
        self.labeled_mask[init_idx] = True

    def _retrain(self):
        """Retrain the phenotype and CVD prediction models based on labeled samples."""
        train_labeled = self.labeled_mask
        if train_labeled.sum() == 0:
            return

        # Train phenotype S using X1 and golden labels
        self.pheno_model.fit(
            self.X1_train[train_labeled], self.golden_labels[train_labeled]
        )

        # Construct new composite features Z = [pheno_prob | X2]
        self.Z_train = np.hstack(
            [
                self.pheno_model.predict_proba(self.X1_train)[:, 1].reshape(-1, 1),
                self.X2_train_scaled,
            ]
        )
        self.Z_val = np.hstack(
            [
                self.pheno_model.predict_proba(self.X1_val)[:, 1].reshape(-1, 1),
                self.X2_val_scaled,
            ]
        )

        # Train the CVD prediction model using labeled Z features
        self.cvd_model.fit(self.Z_train[train_labeled], self.Y_train[train_labeled])

    def _compute_scores(self):
        """Compute scores for uncertainty, expected AUC gain, and diversity."""
        # Predict probabilities on training set
        proba = self.cvd_model.predict_proba(self.Z_train)[:, 1]
        proba = np.clip(proba, 1e-10, 1 - 1e-10)  # 数值稳定

        # Compute uncertainty using entropy
        uncertainty = -proba * np.log(proba) - (1 - proba) * np.log(1 - proba)

        # Compute estimated AUC gain
        auc_gain = np.zeros_like(proba)
        if len(self.auc_history) >= 2:
            delta_auc = np.diff(self.auc_history[-2:])
            avg_gain = delta_auc.mean()
            if self.selection_history:
                last_batch = self.selection_history[-1]
                last_X2 = self.X2_train_scaled[np.isin(self.idx_train, last_batch)]
                similarity = self.X2_train_scaled.dot(last_X2.T).mean(axis=1)
                auc_gain = similarity * avg_gain

        # Penalize repeated selections for diversity
        overlap_penalty = np.zeros(len(proba))
        if self.selection_history:
            all_selected = set(np.concatenate(self.selection_history))
            overlap_penalty = np.array(
                [
                    1 if self.idx_train[i] in all_selected else 0
                    for i in range(len(proba))
                ]
            )

        diversity = 1 - overlap_penalty

        return uncertainty, auc_gain, diversity

    def select_candidates(self, batch_size=40):
        """Select top candidate samples based on combined scores."""
        uncertainty, auc_gain, diversity = self._compute_scores()

        # Normalize all scores to [0, 1]
        uncertainty_norm = (uncertainty - uncertainty.min()) / (
            uncertainty.ptp() + 1e-10
        )
        auc_gain_norm = (auc_gain - auc_gain.min()) / (auc_gain.ptp() + 1e-10)
        diversity_norm = (diversity - diversity.min()) / (diversity.ptp() + 1e-10)

        # Weighted sum of the scores
        weights = [0.3, 0.4, 0.3]
        combined_score = (
            weights[0] * uncertainty_norm
            + weights[1] * auc_gain_norm
            + weights[2] * diversity_norm
        )

        # Select top samples by score (return global indices)
        top_local_indices = np.argsort(-combined_score)[:batch_size]
        return self.idx_train[top_local_indices]

    def generate_golden_labels(self, train_idx):
        """Simulate golden labels using X1 and CUI features."""
        X_batch = self.X1_train[train_idx]
        CUI_batch = self.CUI_train[train_idx]
        features = np.hstack([X_batch, CUI_batch])

        beta = np.random.randn(features.shape[1])
        logit = features @ beta + np.random.normal(scale=0.5, size=len(train_idx))
        p = 1 / (1 + np.exp(-logit))
        base_label = (p > 0.5).astype(int)

        return base_label

    def al_step(self):
        """Perform one iteration of active learning."""
        if self.labeled_mask.sum() >= self.budget:
            return None

        # Select candidate samples (global indices)
        candidates = self.select_candidates(40)
        actual_batch = min(len(candidates), self.budget - self.labeled_mask.sum())
        candidates = candidates[:actual_batch]

        # Convert to internal training indices
        train_candidates = np.where(np.isin(self.idx_train, candidates))[0]

        # Generate labels and update label mask
        new_labels = self.generate_golden_labels(train_candidates)
        self.golden_labels[train_candidates] = new_labels
        self.labeled_mask[train_candidates] = True

        # Retrain models with updated data
        self._retrain()
        self.selection_history.append(candidates)

        # Evaluate validation AUC
        val_proba = self.cvd_model.predict_proba(self.Z_val)[:, 1]
        current_auc = roc_auc_score(self.Y_val, val_proba)
        self.auc_history.append(current_auc)

        # Compute overlap with previous selection
        overlap = 0.0
        if len(self.selection_history) > 1:
            prev = set(self.selection_history[-2])
            curr = set(self.selection_history[-1])
            overlap = len(prev & curr) / len(curr) if len(curr) > 0 else 0.0
        self.overlap_history.append(overlap)

        print(
            f"[AL Step] Labeled: {self.labeled_mask.sum()}/{self.budget} | "
            f"AUC: {current_auc:.3f} | Overlap: {overlap:.3f}"
        )
        return current_auc, overlap

    def visualize_progress(self):
        """Visualize the training progress including AUC and selection overlap."""
        plt.figure(figsize=(12, 5))
        # Plot AUC progression over AL iterations
        plt.subplot(121)
        plt.plot(self.auc_history, marker="o", color="darkgreen")
        plt.title("Validation AUC Progress")
        plt.xlabel("Iteration")
        plt.ylabel("AUC")
        plt.grid(True)
        # Plot selection overlap rate over AL iterations
        plt.subplot(122)
        plt.plot(self.overlap_history, marker="s", color="darkorange")
        plt.title("Selection Overlap Rate")
        plt.xlabel("Iteration")
        plt.ylabel("Overlap")
        plt.grid(True)

        plt.tight_layout()
        plt.show()


# Run AL
if __name__ == "__main__":
    X1, X2, Y, CUI = generate_patients(n=1000, seed=42)
    learner = CVDLearner(X1, X2, Y, CUI, budget=200, val_size=0.2, random_state=42)

    print("Iter | Labeled | AUC   | Overlap")
    while learner.labeled_mask.sum() < learner.budget:
        auc, overlap = learner.al_step()
        print(
            f"{len(learner.auc_history):3d} | "
            f"{learner.labeled_mask.sum():6d} | "
            f"{auc:.3f} | {overlap:.3f}"
        )

    learner.visualize_progress()
