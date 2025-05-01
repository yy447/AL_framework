import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt


# data generation
def generate_simulated_data(n_samples=1000, seed=42):
    np.random.seed(seed)

    # 生成X1: 先整成3维的正态分布（用来代表血压这种feature
    X1 = np.random.normal(loc=0, scale=1, size=(n_samples, 3))  

    # 利用X1生成S1 (真实phenotype标签): 通过logistic模型生成
    beta_S1 = np.array([0.8, -0.5, 1.2])  # 这里设置为3个系数和上面的X1维数同步
    logit_S1 = X1 @ beta_S1 + np.random.normal(0, 0.5, n_samples)  # 添加noise
    p_S1 = 1 / (1 + np.exp(-logit_S1))
    S1_true = (p_S1 > 0.5).astype(int)  # 这里给S1打标签判断是否为phenotype S1

    # 生成X2: 额外的5维EHR feature
    X2 = np.random.normal(loc=0, scale=1, size=(n_samples, 5))

    # 生成模拟的CUI特征
    # 假设CUI与S1有中等相关性
    cui_corr_strength = 0.7  # CUI与真实标签的相关性设为0.7
    X_cui = (S1_true.reshape(-1,1)*cui_corr_strength + np.random.normal(0, 1, (n_samples, 2)))

    # 利用S1+ X2生成Y
    beta_Y = np.concatenate(
        [np.array([1.0]), np.random.randn(5) * 0.5]
    )  
    logit_Y = S1_true * beta_Y[0] + X2 @ beta_Y[1:]
    p_Y = 1 / (1 + np.exp(-logit_Y))
    Y_true = (p_Y > 0.5).astype(int)

    return X1, S1_true, X2, X_cui, Y_true


# AL process
class ActiveLearner:
    def __init__(self, X1, X2, X_cui, S1_true, S1_init, Y_true):
        self.X_pheno = X1  
        self.X2 = X2      
        self.X_cui = X_cui 
        
        self.S1_true = S1_true
        self.Y_true = Y_true
        self.S1_labeled = S1_init.copy()
        
        # 两种lable方法
        self.labeled_mask = {
            'chart_review': np.zeros(len(X1), dtype=bool),  
            'cui': np.zeros(len(X1), dtype=bool)           
        }
        
        self.downstream_model = LogisticRegression()
        self._update_downstream_model()


    def _update_downstream_model(self):
        X_downstream = np.concatenate([
            self.S1_labeled.reshape(-1,1), 
            self.X2
        ], axis=1)
        self.downstream_model.fit(X_downstream, self.Y_true)
    
    def _simulate_cui_labeling(self, indices):
        """模拟NLP通过CUI打标签"""
        # 用CUI特征预测标签（模拟nlp）
        cui_model = LogisticRegression()
        cui_model.fit(self.X_cui, self.S1_true)
        probs = cui_model.predict_proba(self.X_cui[indices])[:,1]
        
        # 加入noise模拟nlp不完美标注
        noise_level = 0.2  # 20%的错误率
        flip_mask = np.random.rand(len(indices)) < noise_level
        labels = np.where(flip_mask, 1 - (probs > 0.5).astype(int), (probs > 0.5).astype(int))
        
        return labels
    
    def active_learning_step(self, budget=50, cui_ratio=0.4):
        """一次AL process
        - cui_ratio: 使用CUI标注的比例（剩余用chart review）
        """
        # 用X1来整出S1
        phenotype_model = LogisticRegression()
        phenotype_model.fit(self.X_pheno, self.S1_labeled)
        probs = phenotype_model.predict_proba(self.X_pheno)[:,1]
        
        # 计算uncertainty
        uncertainties = 1 - np.abs(probs - 0.5)
        
        # 选择候选的样本（排除已标注的）
        candidate_mask = ~(self.labeled_mask['chart_review'] | self.labeled_mask['cui'])
        candidates = np.where(candidate_mask)[0]
        selected = np.argsort(uncertainties[candidates])[-budget:]
        selected_indices = candidates[selected]
        
        # cost分配成两种CUI和chart_review
        n_cui = int(budget * cui_ratio)
        cui_indices = selected_indices[:n_cui]
        chart_indices = selected_indices[n_cui:]
        
        # 处理CUI标注
        if len(cui_indices) > 0:
            cui_labels = self._simulate_cui_labeling(cui_indices)
            self.S1_labeled[cui_indices] = cui_labels
            self.labeled_mask['cui'][cui_indices] = True
        
        # 处理Chart Review（这里假设直接就是完美的标注）
        if len(chart_indices) > 0:
            self.S1_labeled[chart_indices] = self.S1_true[chart_indices]
            self.labeled_mask['chart_review'][chart_indices] = True
        
        # 更新downstream model
        self._update_downstream_model()
        
        return {
            'cui': cui_indices,
            'chart': chart_indices
        }

# evaluatge performance
def evaluate_performance(S1_true, S1_labeled, Y_true, downstream_model, X2):
    # downstream model de prediction
    X_downstream = np.concatenate([S1_labeled.reshape(-1,1), X2], axis=1)
    Y_pred_probs = downstream_model.predict_proba(X_downstream)[:,1]
    
    # phenotype accuracy（统计被标注的部分）
    labeled_mask = S1_labeled != -1
    pheno_acc = accuracy_score(S1_true[labeled_mask], S1_labeled[labeled_mask])
    
    # AUC
    auc = roc_auc_score(Y_true, Y_pred_probs)
    
    return {
        'pheno_accuracy': pheno_acc,
        'downstream_auc': auc,
        'n_labeled': labeled_mask.sum()
    }


if __name__ == "__main__":
    # 生成数据
    X1, S1_true, X2, X_cui, Y_true = generate_simulated_data()
    
    S1_init = np.full_like(S1_true, -1)
    initial_label_mask = np.random.rand(len(S1_true)) < 0.1
    noise_mask = np.random.rand(initial_label_mask.sum()) < 0.3
    true_labels = S1_true[initial_label_mask]
    S1_init[initial_label_mask] = np.where(noise_mask, 1 - true_labels, true_labels)
    
    al = ActiveLearner(X1, X2, X_cui, S1_true, S1_init, Y_true)
    
    # 开始运行AL
    n_rounds = 5
    budget_per_round = 50
    results = []
    
    for round in range(n_rounds):
        # AL（40%用于CUI标注）
        selected = al.active_learning_step(budget=budget_per_round, cui_ratio=0.4)
        
        # evaluate_performance
        metrics = evaluate_performance(
            al.S1_true, al.S1_labeled, al.Y_true, al.downstream_model, al.X2
        )
        
        # 统计这些label的类型
        metrics.update({
            'chart_labeled': al.labeled_mask['chart_review'].sum(),
            'cui_labeled': al.labeled_mask['cui'].sum()
        })
        results.append(metrics)
        
        print(f"Round {round}: "
              f"Pheno Acc={metrics['pheno_accuracy']:.3f}, "
              f"AUC={metrics['downstream_auc']:.3f}, "
              f"Total Labeled={metrics['n_labeled']} "
              f"(Chart={metrics['chart_labeled']}, CUI={metrics['cui_labeled']})")
    
    # 可视化一波
    plt.figure(figsize=(12,4))
    
    plt.subplot(131)
    plt.plot([r['pheno_accuracy'] for r in results], marker='o')
    plt.title('Phenotype Accuracy')
    
    plt.subplot(132)
    plt.plot([r['downstream_auc'] for r in results], marker='o')
    plt.title('Downstream AUC')
    
    plt.subplot(133)
    plt.stackplot(
        range(n_rounds),
        [np.array([r['chart_labeled'] for r in results]), 
         np.array([r['cui_labeled'] for r in results])],
        labels=['Chart Review', 'CUI']
    )
    plt.title('Labeling Budget Allocation')
    plt.legend()
    
    plt.tight_layout()
    plt.show()