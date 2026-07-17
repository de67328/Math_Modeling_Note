"""
03_spline_fitting.py
自然三次样条拟合: 不同复杂度 (df) 下的表现 + 交叉验证选择

关键概念:
- K 个节点 → K 个基函数 → df ≈ K (自然样条)
- df 小 → 过光滑 (高偏差)
- df 大 → 过拟合 (高方差)
- 交叉验证找到最优 df
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import os

save_dir = os.path.join(os.path.dirname(__file__), "pic")
os.makedirs(save_dir, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def natural_spline_basis(x, knots):
    """自然三次样条基矩阵 (同 01, 02)。"""
    K = len(knots)
    N = np.zeros((len(x), K))
    N[:, 0] = 1.0
    N[:, 1] = x

    def d_k(k_idx):
        xi_k = knots[k_idx]
        xi_K = knots[-1]
        p1 = np.maximum(x - xi_k, 0)**3
        p2 = np.maximum(x - xi_K, 0)**3
        return (p1 - p2) / (xi_K - xi_k)

    for k in range(K - 2):
        N[:, k + 2] = d_k(k) - d_k(K - 2)
    return N


def fit_natural_spline(x, y, df):
    """
    用 df 个自由度的自然三次样条拟合。
    节点放在 x 的分位数处 (R 中 bs() 的做法)。
    """
    if df < 2:
        df = 2
    K = df  # 自然样条: K 节点 → K 基函数
    quantiles = np.linspace(0, 1, K)
    # 确保首尾节点在数据范围内
    knots = np.quantile(x, quantiles)
    # 避免重复节点
    knots = np.unique(knots)
    if len(knots) < K:
        # 如果分位数导致重复, 用等间距补充
        knots = np.linspace(x.min(), x.max(), K)
    
    B = natural_spline_basis(x, knots)
    coeff, resid, rank, s = np.linalg.lstsq(B, y, rcond=None)
    
    return knots, coeff


def predict(x_new, knots, coeff):
    B = natural_spline_basis(x_new, knots)
    return B @ coeff


if __name__ == "__main__":
    # ============================================================
    # 生成复杂函数的数据
    # ============================================================
    rng = np.random.default_rng(123)
    n = 80
    x = np.sort(rng.uniform(0, 10, n))
    # 复杂真实函数: 低频 + 中频 + 局部突起
    f_true = (np.sin(x) + 0.5 * np.cos(2*x) 
              + 0.8 * np.exp(-0.5 * ((x-5)/1.5)**2))
    y = f_true + rng.normal(0, 0.4, n)
    
    x_dense = np.linspace(-0.2, 10.2, 400)
    f_true_dense = (np.sin(x_dense) + 0.5 * np.cos(2*x_dense) 
                    + 0.8 * np.exp(-0.5 * ((x_dense-5)/1.5)**2))
    
    # ============================================================
    # 图 1: 不同 df 下的拟合效果
    # ============================================================
    df_list = [2, 3, 5, 8, 12, 20]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for idx, df in enumerate(df_list):
        ax = axes[idx // 3][idx % 3]
        
        knots, coeff = fit_natural_spline(x, y, df)
        y_fit = predict(x_dense, knots, coeff)
        
        ax.plot(x_dense, f_true_dense, 'k-', lw=2, alpha=0.25, label='真实 $f(x)$')
        ax.scatter(x, y, c='gray', s=12, alpha=0.5, zorder=5)
        ax.plot(x_dense, y_fit, 'r-', lw=2, label=f'df={df}')
        
        # 标注节点
        for xi in knots:
            ax.axvline(xi, color='red', ls=':', alpha=0.4, lw=0.8)
        
        # 标注边界
        ax.axvspan(-0.2, knots[0], color='green', alpha=0.06)
        ax.axvspan(knots[-1], 10.2, color='green', alpha=0.06)
        
        # 计算训练 MSE
        y_pred_train = predict(x, knots, coeff)
        mse_train = np.mean((y - y_pred_train)**2)
        
        ax.set_title(f'df = {df} ($K={len(knots)}$ 个节点)\n训练 MSE = {mse_train:.4f}',
                     fontsize=11)
        ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
    
    plt.suptitle('自然三次样条: 不同自由度 (df) 的拟合效果', fontsize=14, y=1.01)
    plt.tight_layout()
    save_path = os.path.join(save_dir, "03_spline_varying_df.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"图 1 已保存: {save_path}")
    
    # ============================================================
    # 图 2: K 折交叉验证选择最优 df
    # ============================================================
    df_candidates = np.arange(2, 31)  # df = 2, 3, ..., 30
    n_folds = 10
    
    cv_errors = np.zeros((n_folds, len(df_candidates)))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(x)):
        x_train, y_train = x[train_idx], y[train_idx]
        x_val, y_val = x[val_idx], y[val_idx]
        
        for j, df in enumerate(df_candidates):
            knots, coeff = fit_natural_spline(x_train, y_train, df)
            y_pred = predict(x_val, knots, coeff)
            cv_errors[fold_idx, j] = np.mean((y_val - y_pred)**2)
    
    mean_cv_error = np.mean(cv_errors, axis=0)
    se_cv_error = np.std(cv_errors, axis=0) / np.sqrt(n_folds)
    
    # 找最优 df (最小 CV 误差)
    best_idx = np.argmin(mean_cv_error)
    best_df = df_candidates[best_idx]
    
    # 1-SE 法则: 在最优值的 1 SE 内取最简单的模型
    threshold = mean_cv_error[best_idx] + se_cv_error[best_idx]
    one_se_df = df_candidates[np.where(mean_cv_error <= threshold)[0][0]]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左: CV 误差曲线
    ax = axes[0]
    ax.plot(df_candidates, mean_cv_error, 'b-o', lw=2, markersize=5,
            label=f'{n_folds}折 CV 平均 MSE')
    ax.fill_between(df_candidates,
                     mean_cv_error - se_cv_error,
                     mean_cv_error + se_cv_error,
                     color='blue', alpha=0.15, label='±1 SE')
    ax.axvline(best_df, color='red', ls='--', lw=1.5,
               label=f'最优 df = {best_df}')
    ax.axvline(one_se_df, color='green', ls='--', lw=1.5,
               label=f'1-SE 法则 df = {one_se_df}')
    ax.axhline(threshold, color='green', ls=':', lw=1, alpha=0.6)
    ax.set_xlabel('自由度 (df)', fontsize=12)
    ax.set_ylabel('交叉验证 MSE', fontsize=12)
    ax.set_title(f'{n_folds} 折交叉验证选择最优 df', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    
    # 右: 最优模型的拟合
    ax = axes[1]
    
    # 最优 df 和 1-SE df 都画出来
    for df_val, color, label in [(best_df, 'red', f'最优 df={best_df}'),
                                   (one_se_df, 'green', f'1-SE df={one_se_df}')]:
        knots_opt, coeff_opt = fit_natural_spline(x, y, df_val)
        y_opt = predict(x_dense, knots_opt, coeff_opt)
        ax.plot(x_dense, y_opt, color=color, lw=2, alpha=0.8, label=label)
    
    ax.plot(x_dense, f_true_dense, 'k-', lw=2, alpha=0.25, label='真实 $f(x)$')
    ax.scatter(x, y, c='gray', s=12, alpha=0.5, zorder=5)
    ax.set_xlabel('$x$', fontsize=12)
    ax.set_ylabel('$y$', fontsize=12)
    ax.set_title('CV 选择的最优自然三次样条', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    save_path2 = os.path.join(save_dir, "03_spline_cv_selection.png")
    fig.savefig(save_path2, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"图 2 已保存: {save_path2}")
    
    print("\n=== 核心总结 ===")
    print(f"• 最优 df = {best_df} (10 折 CV)")
    print(f"• 1-SE 法则 df = {one_se_df} (在最优 1 SE 内选最简单模型)")
    print("• df 小 → 高偏差低方差 (欠拟合/过光滑)")
    print("• df 大 → 低偏差高方差 (过拟合/过于波动)")
    print("• 自然样条的关键优势: K 个节点恰好 = K 个基函数, df 直接等于节点数")
