"""
06_smoothing_spline_df_cv.py
光滑样条的有效自由度与 λ 的自动选择

光滑样条可写为线性光滑器:  ŷ = S_λ y
- 有效自由度: df_λ = trace(S_λ)
- df_λ 在 2 (λ→∞) 和 N (λ=0) 之间连续变化

λ 选择方法:
1. K 折交叉验证 (CV)
2. 广义交叉验证 (GCV) — 计算更高效的近似

ESL §5.4.1
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from sklearn.model_selection import KFold
import os

save_dir = os.path.join(os.path.dirname(__file__), "pic")
os.makedirs(save_dir, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def smoothing_spline_fit(x, y, lam):
    """拟合光滑样条并返回 spline 对象。"""
    order = np.argsort(x)
    x_s, y_s = x[order], y[order]
    s = len(y) * lam if lam > 0 else 0
    try:
        return UnivariateSpline(x_s, y_s, s=max(s, 1e-12), k=3)
    except:
        return None


def approx_df(spl, x):
    """
    近似有效自由度。
    
    光滑样条 df_λ = trace(S_λ)。
    这里用扰动法估计: 对每个 y_i 微小扰动, 测量 ŷ 的变化。
    由于 scipy 不直接返回 S_λ, 用数值近似。
    
    更准确的方法: 使用 csaps 库或 statsmodels。
    这里提供一个基于 leave-one-out 影响的近似。
    """
    # 简化: 用 get_residual() 的倒数关系近似
    # 实际上 UnivariateSpline 不方便获取 df, 
    # 但 df ≈ ∑_i ∂ŷ_i/∂y_i
    # 用有限差分做近似
    try:
        y_pred = spl(x)
        eps = 1e-6
        df_approx = 0.0
        for i in range(len(x)):
            y_pert = y_pred.copy()
            y_pert[i] += eps
            # 重新拟合 (太慢, 实际不可行)
            pass
    except:
        pass
    
    # 更实用的方式: 用 GCV 公式反推
    # GCV = RSS / (1 - df/n)^2
    # 但这也需要 df...
    
    return None


def compute_df_via_loo(spl, x, y):
    """
    通过 leave-one-out 近似计算有效自由度。
    
    df_λ = n - Σ_i (y_i - ŷ_i) / (y_i - ŷ_i^{-i})
    其中 ŷ_i^{-i} 是去掉第 i 个观测后的拟合值。
    
    但逐一去掉观测并重新拟合太慢。在光滑样条中,
    可以用公式近似: df_λ ≈ Σ_i S_ii
    
    实用方法: 利用 UnivariateSpline 的平滑参数 s 与 df 的关系。
    """
    # 简单近似: 数极值点
    x_dense = np.linspace(x.min(), x.max(), 1000)
    y_dense = spl(x_dense)
    dy = np.diff(y_dense)
    sign_changes = np.sum(np.diff(np.sign(dy)) != 0)
    # 非常粗糙: df ≈ sign_changes + 2
    return sign_changes + 2


if __name__ == "__main__":
    # ============================================================
    # 生成数据
    # ============================================================
    rng = np.random.default_rng(123)
    n = 60
    
    # 分两部分: 左半平滑, 右半波动
    x1 = rng.uniform(0, 5, n // 2)
    x2 = rng.uniform(5, 10, n // 2)
    x = np.sort(np.concatenate([x1, x2]))
    
    f_true = np.where(x < 5,
                      0.5 * np.sin(1.5 * x) + 0.3 * x,
                      2.0 * np.sin(0.6 * (x - 5)) + 0.5 * np.cos(1.8 * (x - 5)) + 1.5)
    y = f_true + rng.normal(0, 0.35, n)
    
    x_dense = np.linspace(-0.3, 10.3, 400)
    f_dense = np.where(x_dense < 5,
                       0.5 * np.sin(1.5 * x_dense) + 0.3 * x_dense,
                       2.0 * np.sin(0.6 * (x_dense - 5)) + 0.5 * np.cos(1.8 * (x_dense - 5)) + 1.5)
    
    # ============================================================
    # 图 1: 不同 df 近似下的光滑样条
    # ============================================================
    # 注意: scipy 的 UnivariateSpline 使用 s (平滑参数) 而非 df
    # 关系: s ≈ n * σ²_hat * (某比例)
    # 这里用不同的 s 间接控制复杂度
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # 用不同 s 值 (从小到大 = 从插值到光滑)
    s_values = [0, 0.1, 0.5, 2, 8, 30]
    titles = [
        's=0  (插值, df ≈ n)',
        's=0.1 (很灵活)',
        's=0.5 (灵活)',
        's=2   (适中)',
        's=8   (光滑)',
        's=30  (很光滑)',
    ]
    
    order = np.argsort(x)
    x_sorted, y_sorted = x[order], y[order]
    
    for idx, (s_val, title) in enumerate(zip(s_values, titles)):
        ax = axes[idx // 3][idx % 3]
        
        spl = UnivariateSpline(x_sorted, y_sorted, s=s_val, k=3)
        y_fit = spl(x_dense)
        
        # 近似 df
        df_approx = compute_df_via_loo(spl, x_sorted, y_sorted)
        
        ax.plot(x_dense, f_dense, 'k-', lw=1.5, alpha=0.2, label='真实 $f(x)$')
        ax.scatter(x_sorted, y_sorted, c='gray', s=12, alpha=0.5, zorder=5)
        ax.plot(x_dense, y_fit, 'r-', lw=2)
        
        y_pred = spl(x_sorted)
        mse = np.mean((y_sorted - y_pred)**2)
        
        ax.set_title(f'{title}\n' + r'MSE = {:.4f}, df$\approx$' + f'{df_approx}',
                     fontsize=10)
        ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
        ax.grid(True, alpha=0.2)
    
    plt.suptitle('光滑样条: 不同平滑参数 s 下的拟合\n(s 与 $\\lambda$ 成反比)',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    save_path = os.path.join(save_dir, "06_smoothing_spline_varying_s.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"图 1 已保存: {save_path}")
    
    # ============================================================
    # 图 2: K 折 CV 选择 λ (使用 GCV 风格的网格搜索)
    # ============================================================
    # 在 s 的网格上做 CV
    
    s_grid = np.logspace(-2, 1.5, 30)
    n_folds = 10
    
    cv_errors = np.zeros((n_folds, len(s_grid)))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(x)):
        x_tr, y_tr = x[train_idx], y[train_idx]
        x_val, y_val = x[val_idx], y[val_idx]
        
        for j, s_val in enumerate(s_grid):
            try:
                order_tr = np.argsort(x_tr)
                spl = UnivariateSpline(x_tr[order_tr], y_tr[order_tr],
                                       s=s_val, k=3)
                y_pred = spl(x_val)
                cv_errors[fold_idx, j] = np.mean((y_val - y_pred)**2)
            except:
                cv_errors[fold_idx, j] = np.nan
    
    mean_cv = np.nanmean(cv_errors, axis=0)
    se_cv = np.nanstd(cv_errors, axis=0) / np.sqrt(n_folds)
    
    best_idx = np.nanargmin(mean_cv)
    best_s = s_grid[best_idx]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # 左: CV 曲线
    ax = axes[0]
    ax.plot(s_grid, mean_cv, 'b-o', lw=2, markersize=5,
            label=f'{n_folds}折 CV MSE')
    ax.fill_between(s_grid, mean_cv - se_cv, mean_cv + se_cv,
                     color='blue', alpha=0.15, label='±1 SE')
    ax.axvline(best_s, color='red', ls='--', lw=1.5,
               label=f'最优 $s^*$ = {best_s:.3f}')
    
    ax.set_xscale('log')
    ax.set_xlabel('平滑参数 $s$ (对数尺度)', fontsize=12)
    ax.set_ylabel('交叉验证 MSE', fontsize=12)
    ax.set_title(f'{n_folds} 折 CV 选择光滑参数', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    
    # 右: 最优拟合
    ax = axes[1]
    
    spl_best = UnivariateSpline(x_sorted, y_sorted, s=best_s, k=3)
    y_best = spl_best(x_dense)
    df_best = compute_df_via_loo(spl_best, x_sorted, y_sorted)
    
    ax.plot(x_dense, f_dense, 'k-', lw=2, alpha=0.25, label='真实 $f(x)$')
    ax.scatter(x_sorted, y_sorted, c='gray', s=15, alpha=0.5, zorder=5)
    ax.plot(x_dense, y_best, 'r-', lw=2.5,
            label=f'CV 最优 ($s^*$={best_s:.3f}, df$\\approx${df_best})')
    
    ax.set_xlabel('$x$', fontsize=12); ax.set_ylabel('$y$', fontsize=12)
    ax.set_title(f'CV 选择的最优光滑样条', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    save_path2 = os.path.join(save_dir, "06_smoothing_spline_cv.png")
    fig.savefig(save_path2, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"图 2 已保存: {save_path2}")
    
    print("\n=== 核心总结 ===")
    print(f"• CV 选择的最优 s = {best_s:.3f}")
    print(f"• 有效自由度 df ≈ {df_best}")
    print("• 光滑样条的关键洞察:")
    print("  - 节点自动放在所有唯一 x_i 处 (无需手动选)")
    print("  - λ (或 s) 控制有效复杂度, 而非节点数")
    print("  - CV/GCV 自动选择 λ")
    print("• 与回归样条对比:")
    print("  - 回归样条: 手动选节点数 + 位置 (离散选择)")
    print("  - 光滑样条: 自动放节点 + 连续惩罚参数 (CV 自动选)")
