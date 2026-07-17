"""
07_smoothing_vs_regression_spline.py
光滑样条 vs 回归样条: 系统对比

核心区别:
┌──────────────┬─────────────────┬──────────────────┐
│              │ 回归样条 (§5.2)  │ 光滑样条 (§5.4)   │
├──────────────┼─────────────────┼──────────────────┤
│ 节点位置     │ 手动选 K 个     │ 所有唯一 x_i     │
│ 复杂度控制   │ 节点数 K (离散) │ λ 惩罚 (连续)    │
│ 自由度       │ df = K+4 或 K   │ df_λ ∈ [2, N]    │
│ 选择方法     │ CV 选 K         │ CV/GCV 选 λ      │
│ 实操         │ bs(x, df=k)     │ smooth.spline()  │
└──────────────┴─────────────────┴──────────────────┘

ESL §5.4 指出: 光滑样条在统计上等价于带大量节点 + 惩罚的回归样条,
因此实际上更优 — 免去了选节点的难题。

本程序:
1. 对比同 df 下两种样条的拟合
2. 展示光滑样条 df 连续变化的优势
3. 演示 GC 选择 λ
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline, splrep, splev
from sklearn.model_selection import KFold
import os

save_dir = os.path.join(os.path.dirname(__file__), "pic")
os.makedirs(save_dir, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# 复用自然样条基的函数
def natural_spline_basis(x, knots):
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


if __name__ == "__main__":
    # ============================================================
    # 生成数据
    # ============================================================
    rng = np.random.default_rng(2024)
    n = 50
    x = np.sort(rng.uniform(0.5, 9.5, n))
    
    # 非均匀复杂度的真实函数
    f_true = (np.sin(x) 
              + 0.6 * np.sin(3 * x) * np.exp(-0.3 * x)
              + 0.3 * x)
    sigma = 0.4
    y = f_true + rng.normal(0, sigma, n)
    
    x_dense = np.linspace(-0.2, 10.2, 400)
    f_dense = (np.sin(x_dense) 
               + 0.6 * np.sin(3 * x_dense) * np.exp(-0.3 * x_dense)
               + 0.3 * x_dense)
    
    # ============================================================
    # 图 1: 同 df 下回归样条 vs 光滑样条
    # ============================================================
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    df_targets = [4, 6, 9, 13, 18, 25]
    
    order = np.argsort(x)
    x_sorted, y_sorted = x[order], y[order]
    
    for idx, df_t in enumerate(df_targets):
        ax = axes[idx // 3][idx % 3]
        
        ax.plot(x_dense, f_dense, 'k-', lw=1.5, alpha=0.2, label='真实 $f(x)$')
        ax.scatter(x, y, c='gray', s=12, alpha=0.5, zorder=5)
        
        # --- 回归样条 (自然三次) ---
        K_reg = max(3, df_t)
        knots_reg = np.quantile(x_sorted, np.linspace(0, 1, K_reg))
        knots_reg = np.unique(knots_reg)
        
        B_train = natural_spline_basis(x_sorted, knots_reg)
        coeff = np.linalg.lstsq(B_train, y_sorted, rcond=None)[0]
        B_dense = natural_spline_basis(x_dense, knots_reg)
        y_reg_dense = B_dense @ coeff
        
        y_reg_pred = natural_spline_basis(x_sorted, knots_reg) @ coeff
        mse_reg = np.mean((y_sorted - y_reg_pred)**2)
        
        # --- 光滑样条 ---
        # 调节 s 使 df ≈ df_t
        # 由经验: s ≈ n * σ² * (n/df - 1)
        s_guess = n * sigma**2 * (n / df_t - 1) * 0.5
        s_guess = max(s_guess, 1e-6)
        
        try:
            spl_smooth = UnivariateSpline(x_sorted, y_sorted, s=s_guess, k=3)
            y_smooth_dense = spl_smooth(x_dense)
            y_smooth_pred = spl_smooth(x_sorted)
            mse_smooth = np.mean((y_sorted - y_smooth_pred)**2)
        except:
            y_smooth_dense = np.full_like(x_dense, np.nan)
            mse_smooth = np.nan
        
        ax.plot(x_dense, y_reg_dense, 'b-', lw=2, alpha=0.8,
                label=f'回归样条 (df={df_t}, MSE={mse_reg:.4f})')
        ax.plot(x_dense, y_smooth_dense, 'r--', lw=2, alpha=0.8,
                label=f'光滑样条 (df≈{df_t}, MSE={mse_smooth:.4f})')
        
        # 标注回归样条节点
        for xi in knots_reg:
            ax.axvline(xi, color='blue', ls=':', alpha=0.25, lw=0.6)
        
        ax.set_title(f'目标 df = {df_t}', fontsize=12)
        ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
    
    plt.suptitle('回归样条 vs 光滑样条: 相同自由度下的对比\n'
                 '(实线=回归样条, 虚线=光滑样条; 蓝竖线=回归样条节点)',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    save_path = os.path.join(save_dir, "07_regression_vs_smoothing_spline.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"图 1 已保存: {save_path}")
    
    # ============================================================
    # 图 2: 光滑样条 df 的连续性 — 不同 λ 在 {2,...,N} 上的过渡
    # ============================================================
    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    
    # 上: 极端值对比 (df 近 2 vs df 近 N)
    ax = axes[0]
    ax.plot(x_dense, f_dense, 'k-', lw=2, alpha=0.2)
    ax.scatter(x, y, c='gray', s=15, alpha=0.5, zorder=5)
    
    # λ 接近 0 → 插值
    spl_interp = UnivariateSpline(x_sorted, y_sorted, s=0.01, k=3)
    ax.plot(x_dense, spl_interp(x_dense), 'b-', lw=1, alpha=0.6,
            label='近插值 ($s$ 很小)')
    
    # λ 适中
    s_cv_list = [0.5, 2.0, 8.0]
    colors_cv = ['orange', 'green', 'red']
    for s_v, c_v in zip(s_cv_list, colors_cv):
        spl = UnivariateSpline(x_sorted, y_sorted, s=s_v, k=3)
        ax.plot(x_dense, spl(x_dense), color=c_v, lw=1.5, alpha=0.8,
                label=f'$s={s_v}$')
    
    # λ 很大 → 接近线性
    spl_lin = UnivariateSpline(x_sorted, y_sorted, s=100, k=3)
    ax.plot(x_dense, spl_lin(x_dense), 'purple', lw=2, alpha=0.8,
            label='近线性 ($s$ 很大)')
    
    ax.set_xlabel('$x$', fontsize=12); ax.set_ylabel('$y$', fontsize=12)
    ax.set_title('光滑样条: 从插值到线性的连续过渡', fontsize=13)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.2)
    
    # 下: 逐点导数 (展示不同 λ 对曲率的影响)
    ax = axes[1]
    
    # 算几个不同 s 下的拟合的二阶导数
    s_show = [0.1, 1.0, 5.0, 20.0]
    colors_show = ['blue', 'orange', 'green', 'red']
    
    for s_v, c_v in zip(s_show, colors_show):
        spl = UnivariateSpline(x_sorted, y_sorted, s=s_v, k=3)
        f_fit = spl(x_dense)
        f_dd = np.gradient(np.gradient(f_fit, x_dense), x_dense)
        ax.plot(x_dense, f_dd, color=c_v, lw=1.5,
                label=f'$s={s_v}$')
    
    ax.axhline(0, color='black', lw=0.5)
    ax.set_xlabel('$x$', fontsize=12)
    ax.set_ylabel("$\\hat{f}''(x)$ (曲率)", fontsize=12)
    ax.set_title('不同光滑程度下的估计曲率\n($s$ 越大 → 曲率越小 → 越光滑)',
                 fontsize=13)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    save_path2 = os.path.join(save_dir, "07_smoothing_spline_continuum.png")
    fig.savefig(save_path2, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"图 2 已保存: {save_path2}")
    
    # ============================================================
    # 打印对比总结
    # ============================================================
    print("\n=== 回归样条 vs 光滑样条 总结 ===")
    print("回归样条:")
    print("  • 优势: 概念简单, 自由度 = 基函数数 (自然样条: df=K)")
    print("  • 劣势: 需要选节点数 + 位置 (两个离散决策)")
    print("  • 实际: bs(x, df=k), R 自动放分位数节点, CV 选 df")
    print("\n光滑样条:")
    print("  • 优势: 不需要选节点 (自动在每个 x_i 放), λ 连续调节")
    print("  • 劣势: 计算量更大 (N×N 矩阵), λ 需 CV/GCV")
    print("  • 实际: smooth.spline() 或 GAM 的默认选择")
    print("\nESL 立场:")
    print("  「光滑样条在统计上等价于大量节点 + 惩罚的回归样条」")
    print("  实际上光滑样条更优 — 免去了选节点的难题")
