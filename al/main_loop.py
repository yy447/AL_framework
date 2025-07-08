import numpy as np
from collections import deque
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from rl.agent import PPO
import time


def generate_patients_ch(
    n=1000,
    d_X1=10,
    d_X2=10,
    beta_S_scale=1.5,
    S_star_scale=2,
    beta_Y_scale_X2=0.2,
    beta_Y_scale_S=1.2,
    seed=123,
):
    np.random.seed(seed)

    # X1: for phenotype (S_true)
    X1 = np.random.normal(size=(n, d_X1))

    # X2: for outcome (Y)
    X2 = np.random.normal(size=(n, d_X2))

    # X1 -> S_true
    beta_S = np.random.randn(d_X1) * beta_S_scale
    s_logit = X1 @ beta_S + np.random.normal(scale=0.3, size=n)
    p_S = 1 / (1 + np.exp(-s_logit))
    S_true = np.random.binomial(1, p_S)
    auc_s = roc_auc_score(S_true, p_S)

    # S_true -> S_star ## the larger the scale, the lower the AUC
    s_star_logit = S_true + np.random.normal(scale=S_star_scale, size=n)
    S_star = 1 / (1 + np.exp(-s_star_logit))
    auc_sstar = roc_auc_score(S_true, S_star)

    # S_true + X2 -> Y
    y_coef_s = beta_Y_scale_S  # fixed effect of S_true # the larger the beta, the lower the AUC
    y_coef_x2 = np.random.normal(
        scale=beta_Y_scale_X2, size=X2.shape[1]
    )  # random effects for X2
    y_logit = (
        y_coef_s * S_true + X2 @ y_coef_x2 + np.random.normal(scale=0.5, size=n)
    )  # add noise
    p_Y = 1 / (1 + np.exp(-y_logit))  # sigmoid
    Y = np.random.binomial(1, p_Y)  # binary outcome
    auc_y = roc_auc_score(Y, y_logit)

    return X1, X2, Y, S_true, S_star, auc_s, auc_sstar, auc_y


class StateNormalizer:
    """Dynamic state normalizer, using EMA (Exponential Moving Average) to update mean and standard deviation"""

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


class CVDALRLSystem:
    def __init__(self, data, config):
        self.data = data
        self.X1 = data["X1"]
        self.X2 = data["X2"]
        self.S_true = data["S_true"]
        self.S_star = data["S_star"]
        self.Y = data["Y"]

        self.config = config
        self.budget = config.get("budget", 200)
        self.batch_size = config.get("batch_size", 40)
        self.random_state = config.get("random_state", 42)
        self.val_size = config.get("val_size", 0.2)
        self.initial_samples = config.get("initial_samples", 30)
        self.reward_horizon = config.get("reward_horizon", 3)
        self.reward_gamma = config.get("reward_gamma", 0.9)
        self.strategy = config.get("strategy", "rl")

        # Long-term reward parameters
        self.auc_window = deque(maxlen=self.reward_horizon + 1)
        self.reward_stats = {"mean": 0, "std": 1}

        # Initialization according to policy type
        if self.strategy != "rl":
            if self.strategy == "random":
                self.fixed_strategy = [0, 0, 1]  # only use random policy
            elif self.strategy == "uncertainty":
                self.fixed_strategy = [1, 0, 0]  # only use uncertainty policy
            elif self.strategy == "diversity":
                self.fixed_strategy = [0, 1, 0]  # only use diversity policy
            elif self.strategy == "equal":
                self.fixed_strategy = [0.33, 0.33, 0.34]
            self.agent = None
        else:
            # Initialize Rl agent
            state_dim = 7  # state dimension
            self.agent = PPO(state_dim, config)
        self.state_normalizer = StateNormalizer(
            state_dim=7, min_window_size=20, clip_range=(-5, 5), ema_alpha=0.05
        )
        # Initialize history
        self.auc_history = []
        self.selection_history = []
        self.iteration = 0
        self.early_stop = False
        self._initialize_data_splits()
        self._initialize_models()

    def _initialize_data_splits(self):
        """Initialize test set and val set"""
        idx = np.arange(len(self.Y))
        self.idx_train, self.idx_val = train_test_split(
            idx,
            test_size=self.val_size,
            stratify=self.Y,
            random_state=self.random_state,
        )

        # Extract the training set and validation set
        self.X2_train = self.X2[self.idx_train]
        self.X2_val = self.X2[self.idx_val]
        self.S_star_train = self.S_star[self.idx_train]
        self.S_true_train = self.S_true[self.idx_train]
        self.S_true_val = self.S_true[self.idx_val]
        self.Y_train = self.Y[self.idx_train]
        self.Y_val = self.Y[self.idx_val]

        # Initialize label masks and training labels
        self.labeled_mask = np.zeros(len(self.idx_train), dtype=bool)
        self.S_train = self.S_star_train.copy()

        # Feature standardization
        self.scaler = StandardScaler().fit(self.X2_train)
        self.X2_train_scaled = self.scaler.transform(self.X2_train)
        self.X2_val_scaled = self.scaler.transform(self.X2_val)

        self._init_seed_samples()

    def _init_seed_samples(self):
        """Initialize labeled seed samples"""
        rng = np.random.RandomState(self.random_state)
        pos_idx = np.where(self.S_true_train == 1)[0]
        neg_idx = np.where(self.S_true_train == 0)[0]

        n_per_class = min(max(1, self.initial_samples // 2), len(pos_idx), len(neg_idx))
        seed_idx = np.concatenate(
            [
                rng.choice(pos_idx, n_per_class, replace=False),
                rng.choice(neg_idx, n_per_class, replace=False),
            ]
        )

        self.labeled_mask[seed_idx] = True
        self.S_train[seed_idx] = self.S_true_train[seed_idx]
        self.selected_features = self.X2_train_scaled[seed_idx]

    def _initialize_models(self):
        """Initialize phenotype models and predictive models"""
        # Phenotype model (S_true prediction)
        self.pheno_model = LogisticRegression(
            max_iter=1000, random_state=self.random_state, warm_start=True
        )

        X1_train = self.X1[self.idx_train]
        if self.labeled_mask.sum() > 0:
            self.pheno_model.fit(
                X1_train[self.labeled_mask], self.S_true_train[self.labeled_mask]
            )

        # Prediction model (Y prediction)
        self.pred_model = LogisticRegression(
            max_iter=1000, random_state=self.random_state, warm_start=True
        )
        self._train_prediction_model()

        # record initial AUC
        initial_auc = self._evaluate_model()
        self.auc_history.append(initial_auc)
        self.auc_window.append(initial_auc)

    def _train_prediction_model(self):
        """Training Predictive Models"""
        Z_train = np.hstack([self.S_train.reshape(-1, 1), self.X2_train_scaled])
        if self.labeled_mask.sum() > 0:
            self.pred_model.fit(
                Z_train[self.labeled_mask], self.Y_train[self.labeled_mask]
            )

    def _evaluate_model(self):
        """Evaluating the performance of predictive models"""
        Z_val = np.hstack([self.S_true_val.reshape(-1, 1), self.X2_val_scaled])
        if hasattr(self.pred_model, "classes_"):
            val_proba = self.pred_model.predict_proba(Z_val)[:, 1]
            return roc_auc_score(self.Y_val, val_proba)
        return 0.5

    def _compute_uncertainty_scores(self):
        """Calculating the uncertainty score）"""
        unlabeled_mask = ~self.labeled_mask
        if not unlabeled_mask.any() or not hasattr(self.pred_model, "predict_proba"):
            return np.zeros(len(self.S_train))

        # Only unlabeled samples are processed
        Z_unlabeled = np.hstack(
            [
                self.S_train[unlabeled_mask].reshape(-1, 1),
                self.X2_train_scaled[unlabeled_mask],
            ]
        )

        proba = self.pred_model.predict_proba(Z_unlabeled)[:, 1]
        proba = np.clip(proba, 1e-6, 1 - 1e-6)
        entropy_scores = -proba * np.log(proba) - (1 - proba) * np.log(1 - proba)

        full_scores = np.zeros(len(self.S_train))
        full_scores[unlabeled_mask] = entropy_scores
        return full_scores

    def _compute_diversity_scores(self, k=5):
        """Calculating the diversity score"""
        if not hasattr(self, "selected_features") or len(self.selected_features) == 0:
            return np.random.rand(len(self.X2_train_scaled))

        unlabeled_mask = ~self.labeled_mask
        if not unlabeled_mask.any():
            return np.full(len(self.X2_train_scaled), -np.inf)

        # Accelerating Near-Neighbor Search with KD Trees
        nbrs = NearestNeighbors(
            n_neighbors=min(k, len(self.selected_features)), algorithm="kd_tree"
        )
        nbrs.fit(self.selected_features)
        distances, _ = nbrs.kneighbors(self.X2_train_scaled[unlabeled_mask])

        mean_dist = distances.mean(axis=1)
        std_dist = distances.std(axis=1)
        diversity = 0.6 * mean_dist + 0.4 * std_dist

        progress = self.labeled_mask.sum() / self.budget
        diversity *= np.exp(-2 * progress)

        full_scores = np.full(len(self.X2_train_scaled), 0)
        full_scores[unlabeled_mask] = diversity

        if np.max(full_scores) - np.min(full_scores) > 1e-8:
            full_scores = (full_scores - np.min(full_scores)) / (
                np.max(full_scores) - np.min(full_scores)
            )

        return full_scores

    def _compute_strategy_divergence(self):
        """Calculate the KL scatter between strategies (diversity measure)"""
        unlabeled_mask = ~self.labeled_mask
        if not unlabeled_mask.any():
            return 0.0

        uncertainty = self._compute_uncertainty_scores()[unlabeled_mask]
        diversity = self._compute_diversity_scores()[unlabeled_mask]
        random = np.random.rand(unlabeled_mask.sum())

        strategy_scores = np.vstack([uncertainty, diversity, random]) + 1e-8
        strategy_probs = strategy_scores / strategy_scores.sum(axis=0)

        # Calculate the average KL dispersion
        mean_probs = strategy_probs.mean(axis=1)
        divergences = []
        for i in range(strategy_probs.shape[1]):
            divergences.append(
                np.sum(mean_probs * np.log(mean_probs / strategy_probs[:, i]))
            )

        return float(np.mean(divergences))

    def _get_current_state(self):
        """Constructing the current state vector"""
        # 1-3. Strategy simulation AUC
        uncertainty_scores = self._compute_uncertainty_scores()
        diversity_scores = self._compute_diversity_scores()
        random_scores = np.random.rand(len(self.S_train))

        sim_aucs = [
            self._simulate_strategy(uncertainty_scores),
            self._simulate_strategy(diversity_scores),
            self._simulate_strategy(random_scores),
        ]

        # 4. change in AUC
        delta_auc = 0.0
        if len(self.auc_history) > 1:
            delta_auc = self.auc_history[-1] - self.auc_history[-2]

        # 5. Average uncertainty
        unlabeled_mask = ~self.labeled_mask
        avg_uncertainty = (
            uncertainty_scores[unlabeled_mask].mean() if unlabeled_mask.any() else 0.0
        )

        # 6. budget ratio
        budget_ratio = 1.0 - self.labeled_mask.sum() / self.budget

        # 7. Tactical divergence
        divergence = self._compute_strategy_divergence()

        state_vector = np.array(
            [
                sim_aucs[0],
                sim_aucs[1],
                sim_aucs[2],
                delta_auc,
                avg_uncertainty,
                budget_ratio,
                divergence,
            ]
        )

        # state normalization
        return self.state_normalizer.normalize(state_vector)

    def _simulate_strategy(self, score_vector):
        """Simulation strategy performance (optimized version)"""
        cache_key = hash(tuple(score_vector[:10]))
        if hasattr(self, "_simulation_cache") and cache_key in self._simulation_cache:
            return self._simulation_cache[cache_key]

        temp_mask = self.labeled_mask.copy()
        temp_S = self.S_train.copy()
        remaining_budget = self.budget - temp_mask.sum()
        actual_batch = min(self.batch_size, remaining_budget)

        if actual_batch <= 0:
            result = self.auc_history[-1] if self.auc_history else 0.5
            return result

        unlabeled_idx = np.where(~temp_mask)[0]
        if len(unlabeled_idx) == 0:
            result = self.auc_history[-1] if self.auc_history else 0.5
            return result

        unlabeled_scores = score_vector[unlabeled_idx]
        top_indices = np.argpartition(-unlabeled_scores, actual_batch)[:actual_batch]
        selected = unlabeled_idx[top_indices]

        # Updating labels and training models
        temp_mask[selected] = True
        temp_S[selected] = self.S_true_train[selected]

        # Accelerated training using existing models as a starting point
        temp_model = LogisticRegression(
            max_iter=200,
            random_state=self.random_state,
            warm_start=True,
        )

        Z_train = np.hstack([temp_S.reshape(-1, 1), self.X2_train_scaled])
        if temp_mask.sum() > 0:
            try:
                temp_model.fit(Z_train[temp_mask], self.Y_train[temp_mask])
            except:
                return 0.5

        # Evaluate on a validation set
        Z_val = np.hstack([self.S_true_val.reshape(-1, 1), self.X2_val_scaled])
        try:
            val_proba = temp_model.predict_proba(Z_val)[:, 1]
            auc_score = roc_auc_score(self.Y_val, val_proba)
        except:
            auc_score = 0.5

        # Cached results
        if not hasattr(self, "_simulation_cache"):
            self._simulation_cache = {}
        self._simulation_cache[cache_key] = auc_score

        return auc_score

    def _compute_long_term_reward(self, new_auc):
        # Add short-term incentive component (current AUC improvement)
        immediate_gain = new_auc - self.auc_window[-1] if self.auc_window else 0

        # long-term rewards
        future_gain = 0
        if len(self.auc_history) > 5:
            recent_auc = np.array(self.auc_history[-5:])
            x = np.arange(len(recent_auc))
            coef = np.polyfit(x, recent_auc, 1)[0]
            future_gain = coef * 3

        plateau_penalty = 0
        if len(self.auc_window) > 10:
            last_10_auc = np.array(self.auc_window)[-10:]
            if np.std(last_10_auc) < 0.005:
                plateau_penalty = -0.1

        # Mixed incentives
        total_reward = immediate_gain * 0.7 + future_gain * 0.3 + plateau_penalty

        # normalize rewards
        self.reward_stats["mean"] = 0.9 * self.reward_stats["mean"] + 0.1 * total_reward
        self.reward_stats["std"] = 0.9 * self.reward_stats["std"] + 0.1 * abs(
            total_reward
        )

        normalized_reward = (total_reward - self.reward_stats["mean"]) / (
            self.reward_stats["std"] + 1e-8
        )
        print(
            f"[REWARD DEBUG] immediate={immediate_gain:.4f}, future={future_gain:.4f}, plateau={plateau_penalty:.4f}, total={total_reward:.4f}, norm={normalized_reward:.4f}"
        )
        self.auc_window.append(new_auc)

        return normalized_reward

    def select_samples(self, strategy_weights):
        """Selection of samples (with mechanisms for diversity conservation)"""
        uncertainty = self._compute_uncertainty_scores()
        diversity = self._compute_diversity_scores()
        random = np.random.rand(len(self.S_train))

        def safe_normalize(scores):
            if not np.any(np.isfinite(scores)):
                return np.zeros_like(scores)
            min_val, max_val = np.nanmin(scores), np.nanmax(scores)
            if max_val - min_val < 1e-8:
                return np.zeros_like(scores)
            return (scores - min_val) / (max_val - min_val)

        strategies = [
            safe_normalize(uncertainty),
            safe_normalize(diversity),
            safe_normalize(random),
        ]

        # Apply strategy weights
        scores = sum(w * s for w, s in zip(strategy_weights, strategies))

        # Exclusion of labeled samples
        scores[self.labeled_mask] = -np.inf

        min_new = max(1, int(self.batch_size * 0.3))
        diversity_top = np.argsort(-diversity)[: min_new * 3]
        diversity_selected = set()
        for i in diversity_top:
            if not self.labeled_mask[i] and i not in diversity_selected:
                diversity_selected.add(i)
                if len(diversity_selected) >= min_new:
                    break

        # Remaining through main score selection
        remaining = self.batch_size - len(diversity_selected)
        if remaining > 0:
            unlabeled_idx = np.where(~self.labeled_mask)[0]
            unlabeled_scores = scores[unlabeled_idx]
            top_indices = np.argpartition(
                -unlabeled_scores, min(remaining, len(unlabeled_scores) - 1)
            )[:remaining]
            score_selected = set(unlabeled_idx[i] for i in top_indices)
        else:
            score_selected = set()

        all_selected = diversity_selected | score_selected
        return np.array(list(all_selected)[: self.batch_size])

    def update_models(self, selected_indices):
        """Updated models and labels (with sample weighting)"""
        if len(selected_indices) == 0:
            return

        # update label
        self.S_train[selected_indices] = self.S_true_train[selected_indices]
        self.labeled_mask[selected_indices] = True

        # update features
        new_features = self.X2_train_scaled[selected_indices]
        if hasattr(self, "selected_features"):
            self.selected_features = np.vstack([self.selected_features, new_features])
        else:
            self.selected_features = new_features

        sample_weights = np.ones(len(self.labeled_mask))
        sample_weights[selected_indices] = 2.0
        Z_train = np.hstack([self.S_train.reshape(-1, 1), self.X2_train_scaled])
        self.pred_model.fit(
            Z_train[self.labeled_mask],
            self.Y_train[self.labeled_mask],
            sample_weight=sample_weights[self.labeled_mask],
        )

        # Updating the phenotype model
        X1_train = self.X1[self.idx_train]
        self.pheno_model.fit(
            X1_train[self.labeled_mask],
            self.S_true_train[self.labeled_mask],
            sample_weight=sample_weights[self.labeled_mask],
        )

        # Recording History
        self.iteration += 1
        self.selection_history.append(selected_indices)
        current_auc = self._evaluate_model()
        self.auc_history.append(current_auc)
        return current_auc

    def run_al_epoch(self):
        if self.early_stop or self.labeled_mask.sum() >= self.budget:
            return None

        # Non-RL Strategies
        if self.strategy != "rl" and self.fixed_strategy is not None:
            selected = self.select_samples(self.fixed_strategy)
            old_auc = self.auc_history[-1] if self.auc_history else 0.5
            new_auc = self.update_models(selected)
            return {
                "iteration": self.iteration,
                "auc": new_auc,
                "reward": new_auc - old_auc,
                "strategy_weights": self.fixed_strategy,
                "num_labeled": self.labeled_mask.sum(),
            }

        # RL policy
        current_state = self._get_current_state()
        if np.any(np.isnan(current_state)):
            current_state = np.nan_to_num(current_state, nan=0.0)
            print(
                f"Warning: NaN status detected, replaced with 0 (iteration {self.iteration})"
            )

        strategy_weights = self.agent.select_action(current_state)
        strategy_weights = strategy_weights / (strategy_weights.sum() + 1e-8)
        if np.any(np.isnan(strategy_weights)):
            strategy_weights = np.array([0.4, 0.4, 0.2])
            print(
                f"Warning: NaN policy weights detected, use default policy (iterate {self.iteration})"
            )
        # Select Sample
        selected = self.select_samples(strategy_weights)
        current_auc = self.auc_history[-1] if self.auc_history else 0.5
        new_auc = self.update_models(selected)
        # get new state
        next_state = self._get_current_state()
        # Calculating long-term rewards
        # total_reward = new_auc - current_auc
        total_reward = self._compute_long_term_reward(new_auc)
        # Check for termination
        done = self.early_stop or (self.labeled_mask.sum() >= self.budget)

        # Storage experience
        self.agent.buffer.rewards.append(total_reward)
        self.agent.buffer.is_terminals.append(done)

        # Updating the RL strategy
        self.agent.update()
        self.iteration += 1

        # check_early_stopping
        self._check_early_stopping()

        return {
            "iteration": self.iteration,
            "auc": new_auc,
            "reward": total_reward,
            "strategy_weights": strategy_weights,
            "num_labeled": self.labeled_mask.sum(),
        }

    def _check_early_stopping(self, patience=5, min_improvement=0.001):
        """Early stop mechanism (with adaptive threshold)"""
        if len(self.auc_history) < patience + 5:
            return

        # Calculate the average improvement
        recent_auc = self.auc_history[-patience:]
        avg_improvement = (recent_auc[-1] - recent_auc[0]) / patience

        progress = self.labeled_mask.sum() / self.budget
        adaptive_threshold = min_improvement * (1.0 - 0.5 * progress)

        if avg_improvement <= adaptive_threshold:
            self.early_stop = True

    def full_training_loop(self):
        results = []
        start_time = time.time()
        print("Starting Active Learning Training...")

        while not self.early_stop and self.labeled_mask.sum() < self.budget:
            epoch_result = self.run_al_epoch()
            if epoch_result is None:
                break

            results.append(epoch_result)
            print(
                f"Iter {epoch_result['iteration']:03d}: "
                f"AUC={epoch_result['auc']:.4f} | "
                f"Reward={epoch_result['reward']:.4f} | "
                f"Strategy={epoch_result['strategy_weights']} | "
                f"Labeled={epoch_result['num_labeled']}/{self.budget}"
            )

        duration = time.time() - start_time
        print(f"Training completed in {duration:.2f} seconds")
        return results

    def visualize(self):
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(self.auc_history, "o-", color="darkred", linewidth=2)
        plt.title("Validation AUC Progress")
        plt.xlabel("AL Iterations")
        plt.ylabel("AUC")
        plt.grid(True, alpha=0.3)

        if self.strategy == "rl":
            weights_history = [r["strategy_weights"] for r in self.full_results]
            weights_array = np.array(weights_history)

            plt.subplot(1, 2, 2)
            plt.stackplot(
                range(len(weights_array)),
                weights_array.T,
                labels=["Uncertainty", "Diversity", "Random"],
            )
            plt.title("Strategy Weights Evolution")
            plt.xlabel("AL Iterations")
            plt.ylabel("Weight")
            plt.legend(loc="upper left")
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


def main():
    strategies = [
        "random",
        "uncertainty",
        "diversity",
        "hybrid_rl",
        "S_star_baseline",
        "S_true_oracle",
    ]
    strategy_auc_histories = {}
    auc_s_list = []
    auc_sstar_list = []
    auc_y_list = []

    # rl_config
    rl_config = {
        "budget": 200,
        "batch_size": 20,
        "val_size": 0.2,
        "reward_horizon": 5,
        "reward_gamma": 0.9,
        "lr_actor": 0.0001,
        "lr_critic": 0.0005,
        "K_epochs": 8,
        "entropy_coef": 0.005,
        "initial_samples": 40,
    }

    start_time = time.time()

    for strategy in strategies:
        print(f"\n== Running strategy: {strategy} ==")
        all_auc_histories = []

        for run in range(10):
            X1, X2, Y, S_true, S_star, auc_s, auc_sstar, auc_y = generate_patients_ch(
                n=1000,
                seed=run,
                beta_S_scale=1.5,
                S_star_scale=2,
                beta_Y_scale_X2=0.2,
                beta_Y_scale_S=5,
            )

            auc_s_list.append(auc_s)
            auc_sstar_list.append(auc_sstar)
            auc_y_list.append(auc_y)

            if strategy in ["S_star_baseline", "S_true_oracle"]:
                idx = np.arange(len(Y))
                idx_train, idx_val = train_test_split(
                    idx, test_size=0.2, stratify=Y, random_state=run
                )

                X2_train, X2_val = X2[idx_train], X2[idx_val]
                scaler = StandardScaler().fit(X2_train)
                X2_train = scaler.transform(X2_train)
                X2_val = scaler.transform(X2_val)

                if strategy == "S_star_baseline":
                    S_used = S_star
                else:  # S_true_oracle
                    S_used = S_true

                Z_train = np.hstack([S_used[idx_train].reshape(-1, 1), X2_train])
                Z_val = np.hstack([S_true[idx_val].reshape(-1, 1), X2_val])

                model = LogisticRegression(max_iter=1000)
                model.fit(Z_train, Y[idx_train])
                proba_val = model.predict_proba(Z_val)[:, 1]
                auc_val = roc_auc_score(Y[idx_val], proba_val)

                auc_history = [auc_val] * 20
                all_auc_histories.append(auc_history)
                continue

            data = {"X1": X1, "X2": X2, "S_true": S_true, "S_star": S_star, "Y": Y}

            if strategy == "hybrid_rl":
                config_strategy = "rl"
            else:
                config_strategy = strategy

            system = CVDALRLSystem(
                data=data,
                config={**rl_config, "strategy": config_strategy, "random_state": run},
            )

            results = system.full_training_loop()
            auc_history = [result["auc"] for result in results]

            if len(auc_history) < 20:
                last_auc = auc_history[-1] if auc_history else 0.5
                auc_history += [last_auc] * (20 - len(auc_history))

            all_auc_histories.append(auc_history)

        strategy_auc_histories[strategy] = all_auc_histories

    print(f"\nTotal experiment time: {time.time() - start_time:.2f} seconds")

    max_len = max(
        len(run) for histories in strategy_auc_histories.values() for run in histories
    )

    plt.figure(figsize=(10, 7))
    for strategy, histories in strategy_auc_histories.items():
        aligned_histories = []
        for run in histories:
            if len(run) < max_len:
                padded_run = run + [run[-1]] * (max_len - len(run))
                aligned_histories.append(padded_run)
            else:
                aligned_histories.append(run[:max_len])

        auc_array = np.array(aligned_histories)
        mean_auc = np.nanmean(auc_array, axis=0)
        std_auc = np.nanstd(auc_array, axis=0)
        ci = 1.96 * std_auc / np.sqrt(len(histories))

        if strategy == "hybrid_rl":
            plt.plot(
                mean_auc, label=strategy, linewidth=2.5, color="darkred", linestyle="-"
            )
        else:
            plt.plot(mean_auc, label=strategy, alpha=0.8)

        plt.fill_between(
            np.arange(len(mean_auc)), mean_auc - ci, mean_auc + ci, alpha=0.15
        )

    plt.title("Validation AUC vs. AL Iteration", fontsize=14)
    plt.xlabel("AL Iteration", fontsize=12)
    plt.ylabel("Validation AUC", fontsize=12)
    plt.legend(fontsize=10, loc="lower right")
    plt.grid(True, alpha=0.2)
    plt.ylim(0.5, 0.9)
    plt.tight_layout()

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig(f"auc_comparison_{timestamp}.png", dpi=300)
    plt.show()

    print("\n=== Final AUC Summary ===")
    for strategy, histories in strategy_auc_histories.items():
        final_aucs = [run[-1] for run in histories]
        mean_final_auc = np.mean(final_aucs)
        std_final_auc = np.std(final_aucs)
        print(
            f"{strategy:>18}: AUC = {mean_final_auc:.4f} ± {std_final_auc:.4f} (n={len(final_aucs)})"
        )
    print("\n=== Data Quality Summary ===")
    print(
        f"AUC(X1 → S_true)     : {np.mean(auc_s_list):.4f} ± {np.std(auc_s_list):.4f}"
    )
    print(
        f"AUC(S_star → S_true) : {np.mean(auc_sstar_list):.4f} ± {np.std(auc_sstar_list):.4f}"
    )
    print(
        f"AUC(S_true + X2 → Y) : {np.mean(auc_y_list):.4f} ± {np.std(auc_y_list):.4f}"
    )


if __name__ == "__main__":
    main()
