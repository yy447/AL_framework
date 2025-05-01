import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt


# data generation
def generate_simulated_data(n_samples=1000, seed=42):
    np.random.seed(seed)

    # 生成X1: 先整成3维的正态分布（用来代表血压这种特征
    X1 = np.random.normal(loc=0, scale=1, size=(n_samples, 3))  

    # 利用X1生成S1 (真实phenotype标签): 通过logistic模型生成
    beta_S1 = np.array([0.8, -0.5, 1.2])  # 这里设置为随便的3个系数和上面的X1维数同步
    logit_S1 = X1 @ beta_S1 + np.random.normal(0, 0.5, n_samples)  # 添加noise
    p_S1 = 1 / (1 + np.exp(-logit_S1))
    S1_true = (p_S1 > 0.5).astype(int)  # 这里给S1打标签判断是否为phenotype S1

    # 生成X2: 额外的5维EHR特征
    X2 = np.random.normal(loc=0, scale=1, size=(n_samples, 5))

    # 生成Y: 依赖S1和X2
    beta_Y = np.concatenate(
        [np.array([1.0]), np.random.randn(5) * 0.5]
    )  # S1+ X2 weight
    logit_Y = S1_true * beta_Y[0] + X2 @ beta_Y[1:]
    p_Y = 1 / (1 + np.exp(-logit_Y))
    Y_true = (p_Y > 0.5).astype(int)

    return X1, S1_true, X2, Y_true


# AL process
class ActiveLearner:
    def __init__(self, X1, X2, S1_true, S1_init, Y_true):
        """
        S1_init: 初始带moise的phenotype标签（模拟不完美的初始phenotype标签）
        Y_true: 真实的downstream CVD标签
        """
        self.X = np.concatenate([X1, X2], axis=1)
        self.S1_true = S1_true
        self.Y_true = Y_true
        self.S1_labeled = S1_init.copy()  # 初始标签（含noise）
        self.labeled_mask = np.zeros(len(X1), dtype=bool)  # 记录哪些样本已被标注

        # 初始化downstream预测模型
        self.downstream_model = LogisticRegression()

    def compute_uncertainty(self, probs, method="entropy"):
        """计算uncertainty score"""
        if method == "entropy":
            return -probs * np.log(probs) - (1 - probs) * np.log(1 - probs)
        elif method == "margin":
            return 1 - np.abs(probs - 0.5)
        else:  # 默认使用距离0.5的绝对值
            return np.abs(probs - 0.5)

    def active_learning_step(self, budget=50, uncertainty_method="entropy"):
        """一次AL迭代"""
        # 训练当前phenotype预测模型
        phenotype_model = LogisticRegression()
        phenotype_model.fit(self.X, self.S1_labeled)

        # 预测所有样本的概率
        probs = phenotype_model.predict_proba(self.X)[:, 1]

        # 计算uncertainty并选择样本
        uncertainties = self.compute_uncertainty(probs, method=uncertainty_method)

        # 选择最不确定的样本（排除已标注的）
        candidate_indices = np.where(~self.labeled_mask)[0]
        selected = np.argsort(uncertainties[candidate_indices])[-budget:]
        selected_indices = candidate_indices[selected]

        # 开始标注这些样本并且更新S1_labeled（假设标注完全正确）
        self.S1_labeled[selected_indices] = self.S1_true[selected_indices]
        self.labeled_mask[selected_indices] = True

        # 更新downstream CVD预测模型
        self.downstream_model.fit(
            np.concatenate([self.S1_labeled.reshape(-1, 1), self.X], axis=1),
            self.Y_true,
        )

        return selected_indices


# model evalation
def evaluate_performance(S1_true, S1_pred, Y_true, Y_pred_probs):
    """计算phenotype预测和downstream的评估指标"""
    # phenotype评估
    pheno_metrics = {
        "accuracy": accuracy_score(S1_true, S1_pred),
        "f1": f1_score(S1_true, S1_pred),
    }

    # downstream评估
    downstream_metrics = {
        "auc": roc_auc_score(Y_true, Y_pred_probs),
        "brier": brier_score_loss(Y_true, Y_pred_probs),
    }

    # 校准曲线
    prob_true, prob_pred = calibration_curve(Y_true, Y_pred_probs, n_bins=10)
    plt.plot(prob_pred, prob_true, marker="o", label="Current Model")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfectly Calibrated")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.legend()
    plt.close()

    return pheno_metrics, downstream_metrics, (prob_true, prob_pred)


# main
if __name__ == "__main__":
    # 生成模拟数据
    X1, S1_true, X2, Y_true = generate_simulated_data(n_samples=1000)

    # 初始化带noise的标签（模拟初始不完美标签）
    S1_init = S1_true.copy()
    noise_mask = np.random.rand(len(S1_true)) < 0.3  # 30%的标签被随机翻转
    S1_init[noise_mask] = 1 - S1_init[noise_mask]

    # 初始化AL系统
    al = ActiveLearner(X1, X2, S1_true, S1_init, Y_true)

    # 运行多轮AL
    n_rounds = 5
    budget_per_round = 50
    results = []

    for round in range(n_rounds):
        # 执行AL迭代
        selected = al.active_learning_step(budget=budget_per_round)

        # 获取当前预测结果
        current_S1_pred = al.S1_labeled.copy()
        current_Y_probs = al.downstream_model.predict_proba(
            np.concatenate([al.S1_labeled.reshape(-1, 1), al.X], axis=1)
        )[:, 1]

        # 评估性能
        pheno_metrics, downstream_metrics, calib_curve = evaluate_performance(
            al.S1_true, current_S1_pred, al.Y_true, current_Y_probs
        )

        # 记录结果
        results.append(
            {
                "round": round,
                "pheno": pheno_metrics,
                "downstream": downstream_metrics,
                "calibration": calib_curve,
            }
        )

        print(
            f"Round {round}: Pheno Accuracy={pheno_metrics['accuracy']:.3f}, AUC={downstream_metrics['auc']:.3f}"
        )

    # 画图
    plt.plot([r["round"] for r in results], [r["pheno"]["accuracy"] for r in results])
    plt.xlabel("AL Iteration")
    plt.ylabel("Phenotype Accuracy")
    plt.title("Active Learning Progress")
    plt.show()

    # 绘制AUC变化
    plt.plot([r["round"] for r in results], [r["downstream"]["auc"] for r in results])
    plt.xlabel("AL Iteration")
    plt.ylabel("CVD Prediction AUC")
    plt.show()
