"""
10_kernel_ridge_regression.py
表示定理的应用: 核岭回归 (Kernel Ridge Regression)

表示定理:
    min_{f∈H_K} Σ_i (y_i - f(x_i))^2 + λ ||f||_H^2
    的解 = f(x) = Σ_i α_i K(x, x_i)

其中 α = (K + λ I)^{-1} y

对比:
- 线性岭回归: β = (X^T X + λ I)^{-1} X^T y  (p 个参数)
- 核岭回归:   α = (K + λ I)^{-1} y            (N 个参数)

关键洞察:
- 核技巧使我们可以隐式使用"无限维"特征空间
- 计算复杂度 O(N^3), 与特征维度 p 无关
- λ 控制过拟合: λ→0 插值, λ→∞ 常数

ESL §5.8
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve
from sklearn.model_selection import KFold
import os

save_dir = os.path.join(os.path.dirname(__file__), "pic")
os.makedirs(save_dir, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def gaussian_kernel(X, Y=None, gamma=1.0):
    if Y is None:
        Y = X
    X_norm = np.sum(X**2, axis=1).reshape(-1, 1)
    Y_norm = np.sum(Y**2, axis=1).reshape(1, -1)
    dist_sq = X_norm + Y_norm - 2 * X @ Y.T
    return np.exp(-gamma * dist_sq)


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    
    # ============================================================
    # 生成非线性数据
    # ============================================================
    n = 40
    X = np.sort(rng.uniform(-3, 3, n)).reshape(-1, 1)
    
    # 复杂真实函数: 低频 + 中频 + 尖峰
    f_true = (np.sin(1.5 * X.ravel())
              + 0.5 * np.cos(3 * X.ravel())
              + 1.5 * np.exp(-2 * (X.ravel() + 1)**2)
              - 0.8 * np.exp(-3 * (X.ravel() - 1.5)**2))
    
    sigma = 0.25
    y = f_true + rng.normal(0, sigma, n)
    
    X_plot = np.linspace(-3.5, 3.5, 300).reshape(-1, 1)
    f_true_plot = (np.sin(1.5 * X_plot.ravel())
                   + 0.5 * np.cos(3 * X_plot.ravel())
                   + 1.5 * np.exp(-2 * (X_plot.ravel() + 1)**2)
                   - 0.8 * np.exp(-3 * (X_plot.ravel() - 1.5)**2))
    
    # ============================================================
    # 图 1: 核岭回归 — 不同 λ 的效果
    # ============================================================
    gamma = 1.0
    lam_values = [1e-8, 0.001, 0.01, 0.1, 1.0, 10.0]
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    for idx, lam in enumerate(lam_values):
        ax = axes[idx // 3][idx % 3]
        
        K = gaussian_kernel(X, gamma=gamma)
        alpha = solve(K + lam * np.eye(n), y, assume_a='pos')
        K_plot = gaussian_kernel(X_plot, X, gamma=gamma)
        y_plot = K_plot @ alpha
        
        # 训练误差
        y_pred_train = K @ alpha
        mse_train = np.mean((y - y_pred_train)**2)
        # RKHS 范数
        norm = np.sqrt(alpha.T @ K @ alpha)
        
        ax.plot(X_plot, f_true_plot, 'k-', lw=1.5, alpha=0.2, label='真实 $f(x)$')
        ax.scatter(X, y, c='gray', s=15, alpha=0.5, zorder=5)
        ax.plot(X_plot, y_plot, 'r-', lw=2, label=f'核岭回归')
        
        ax.set_title(f'$\\lambda={lam}$: MSE={mse_train:.4f}, $\\|f\\|_H$={norm:.2f}',
                     fontsize=10)
        ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.2)
    
    plt.suptitle(f'核岭回归 (高斯核 $\\gamma={gamma}$): $\\lambda$ 的效应\n'
                 '$\\lambda \\uparrow$ → $\\|f\\|_H \\downarrow$ → 更光滑/更简单',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    save_path = os.path.join(save_dir, "10_kernel_ridge_lambda.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"图 1 已保存: {save_path}")
    
    # ============================================================
    # 图 2: CV 选择 λ 和 γ
    # ============================================================
    lam_grid = np.logspace(-5, 1, 20)
    gamma_grid = np.logspace(-1, 1.5, 15)
    n_folds = 10
    
    cv_scores = np.zeros((len(gamma_grid), len(lam_grid)))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for gi, gamma_val in enumerate(gamma_grid):
        for lj, lam_val in enumerate(lam_grid):
            fold_errors = np.zeros(n_folds)
            for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(X)):
                X_tr, y_tr = X[tr_idx], y[tr_idx]
                X_val, y_val = X[val_idx], y[val_idx]
                
                K_tr = gaussian_kernel(X_tr, gamma=gamma_val)
                try:
                    alpha = solve(K_tr + lam_val * np.eye(len(X_tr)), y_tr,
                                  assume_a='pos')
                    K_val = gaussian_kernel(X_val, X_tr, gamma=gamma_val)
                    y_pred = K_val @ alpha
                    fold_errors[fold_idx] = np.mean((y_val - y_pred)**2)
                except:
                    fold_errors[fold_idx] = np.nan
            
            cv_scores[gi, lj] = np.nanmean(fold_errors)
    
    best_idx = np.unravel_index(np.nanargmin(cv_scores), cv_scores.shape)
    best_gamma = gamma_grid[best_idx[0]]
    best_lam = lam_grid[best_idx[1]]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # 左: CV 热力图
    ax = axes[0]
    L, G = np.meshgrid(lam_grid, gamma_grid)
    im = ax.contourf(L, G, np.log10(cv_scores), levels=20, cmap='RdYlBu_r')
    ax.plot(best_lam, best_gamma, 'r*', ms=20, markeredgecolor='white',
            markeredgewidth=2, label=f'最优: $\\lambda^*={best_lam:.4f}$, $\\gamma^*={best_gamma:.2f}$')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('$\\lambda$ (正则化)', fontsize=12)
    ax.set_ylabel('$\\gamma$ (核尺度)', fontsize=12)
    ax.set_title(f'{n_folds} 折 CV: $\\log_{{10}}$(MSE)', fontsize=13)
    ax.legend(fontsize=9)
    plt.colorbar(im, ax=ax, shrink=0.8, label='$\\log_{10}$(CV MSE)')
    
    # 右: 最优拟合
    ax = axes[1]
    K_best = gaussian_kernel(X, gamma=best_gamma)
    alpha_best = solve(K_best + best_lam * np.eye(n), y, assume_a='pos')
    K_plot_best = gaussian_kernel(X_plot, X, gamma=best_gamma)
    y_plot_best = K_plot_best @ alpha_best
    
    ax.plot(X_plot, f_true_plot, 'k-', lw=2, alpha=0.25, label='真实 $f(x)$')
    ax.scatter(X, y, c='gray', s=15, alpha=0.5, zorder=5)
    ax.plot(X_plot, y_plot_best, 'r-', lw=2.5,
            label=f'CV 最优: $\\gamma={best_gamma:.2f}$, $\\lambda={best_lam:.4f}$')
    
    ax.set_xlabel('$x$', fontsize=12); ax.set_ylabel('$y$', fontsize=12)
    ax.set_title('核岭回归: CV 选择的最优拟合', fontsize=13)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    save_path2 = os.path.join(save_dir, "10_kernel_ridge_cv.png")
    fig.savefig(save_path2, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"图 2 已保存: {save_path2}")
    
    # ============================================================
    # 图 3: 表示定理的直观 —— α 系数的含义
    # ============================================================
    gamma_viz = 0.8
    lam_viz = 0.01
    K_viz = gaussian_kernel(X, gamma=gamma_viz)
    alpha_viz = solve(K_viz + lam_viz * np.eye(n), y, assume_a='pos')
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [1.5, 1]})
    
    # 上: 拟合 + 基函数权重
    ax = axes[0]
    K_plot_viz = gaussian_kernel(X_plot, X, gamma=gamma_viz)
    y_plot_viz = K_plot_viz @ alpha_viz
    
    ax.plot(X_plot, f_true_plot, 'k-', lw=1.5, alpha=0.2, label='真实 $f(x)$')
    ax.scatter(X, y, c='gray', s=20, alpha=0.5, zorder=10)
    
    # 标出每个训练点的 α 权重 (大小)
    scatter = ax.scatter(X.ravel(), y, c=alpha_viz, cmap='RdBu_r',
                         s=np.abs(alpha_viz)*200 + 30,
                         edgecolors='black', linewidth=0.5, zorder=15)
    plt.colorbar(scatter, ax=ax, shrink=0.8, label='$\\alpha_i$ (权重)')
    
    ax.plot(X_plot, y_plot_viz, 'r-', lw=2, label='$\\hat{f}(x) = \\sum_i \\alpha_i K(x, x_i)$')
    ax.set_xlabel('$x$', fontsize=12); ax.set_ylabel('$y$', fontsize=12)
    ax.set_title('表示定理: $\\hat{f}(x) = \\sum_{i=1}^{N} \\alpha_i K(x, x_i)$\n'
                 '(散点大小 ∝ |α_i|, 颜色 = 正/负)',
                 fontsize=13)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.2)
    
    # 下: 基函数分解
    ax = axes[1]
    # 选几个有代表性的训练点展示其基函数贡献
    top_indices = np.argsort(np.abs(alpha_viz))[-5:]
    
    colors_alpha = plt.cm.tab10(np.linspace(0, 1, 5))
    for idx_j, (i, c) in enumerate(zip(top_indices, colors_alpha)):
        k_i = gaussian_kernel(X_plot, X[i:i+1], gamma=gamma_viz).ravel()
        contrib = alpha_viz[i] * k_i
        ax.plot(X_plot, contrib, color=c, lw=1.5, alpha=0.7,
                label=f'$\\alpha_{{{i+1}}} K(x, x_{{{i+1}}})$ (|$\\alpha$|={abs(alpha_viz[i]):.1f})')
    
    # 总和
    ax.plot(X_plot, y_plot_viz, 'k--', lw=2, alpha=0.6, label='总和 $\\hat{f}(x)$')
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_xlabel('$x$', fontsize=12); ax.set_ylabel('贡献', fontsize=12)
    ax.set_title('基函数分解: 前 5 大权重项及其贡献', fontsize=13)
    ax.legend(fontsize=8, ncol=3); ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    save_path3 = os.path.join(save_dir, "10_representer_theorem_weights.png")
    fig.savefig(save_path3, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"图 3 已保存: {save_path3}")
    
    print("\n=== 核心总结 ===")
    print(f"• CV 选择的最优 γ = {best_gamma:.2f}, λ = {best_lam:.4f}")
    print("• 表示定理: 无穷维优化 → N 维线性系统 α = (K+λI)^{-1} y")
    print("• λ 效应: 小 → 插值 (每个点放一个窄基函数, α 大)")
    print("  λ 大 → 光滑 (基函数被压扁, α 均匀分布)")
    print("• γ 效应: 小 → 基函数宽 (全局光滑)")
    print("  γ 大 → 基函数窄 (局部灵活, 但需更多训练点支撑)")
    print("• 核岭回归 = 在 N 个以训练点为中心的高斯函数上做带 λ 惩罚的线性回归")
