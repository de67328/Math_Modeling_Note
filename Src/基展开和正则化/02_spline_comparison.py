"""
02_spline_comparison.py
对比标准三次样条 vs 自然三次样条

核心区别:
- 标准三次样条: 边界行为不受约束 → 边界方差大
- 自然三次样条: 边界外强制为线性 → 边界方差小, 偏差略增

ESL 图 5.3 的精神: 展示自然样条在边界处的稳定优势
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, BSpline, splev, make_lsq_spline
import os

# ============================================================
# 0. 设置
# ============================================================
save_dir = os.path.join(os.path.dirname(__file__), "pic")
os.makedirs(save_dir, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def generate_data(n=50, noise=0.3, seed=42):
    """生成带噪声的正弦数据, 重点考察边界行为。"""
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0, 10, n))
    # 真实函数: sin(x) 在中间区域, 边界附近有弯曲
    f_true = np.sin(x) + 0.3 * np.cos(1.5 * x)
    y = f_true + rng.normal(0, noise, n)
    return x, y, f_true


if __name__ == "__main__":
    x, y, f_true = generate_data(n=60, noise=0.4, seed=42)
    x_dense = np.linspace(-0.5, 10.5, 300)
    f_true_dense = np.sin(x_dense) + 0.3 * np.cos(1.5 * x_dense)
    
    # ============================================================
    # 图 1: 单次拟合对比 —— 标准 vs 自然三次样条
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    K = 7  # 节点数
    knots = np.quantile(x, np.linspace(0, 1, K))
    
    # --- 左: 标准三次样条 ---
    ax = axes[0]
    
    # 使用 scipy 拟合标准三次样条 (边界处自然由数据决定)
    tck_std = splrep(x, y, k=3, s=0)  # s=0 → 插值; 实际上用平滑
    y_std_dense = splev(x_dense, tck_std)
    
    ax.plot(x_dense, f_true_dense, 'k-', lw=2, alpha=0.3, label='真实函数 $f(x)$')
    ax.scatter(x, y, c='gray', s=20, alpha=0.6, zorder=5, label='观测数据')
    ax.plot(x_dense, y_std_dense, 'b-', lw=2, label='标准三次样条')
    
    # 高亮边界区域
    ax.axvspan(-0.5, knots[0], color='orange', alpha=0.1)
    ax.axvspan(knots[-1], 10.5, color='orange', alpha=0.1)
    ax.text(0.0, ax.get_ylim()[1]*0.85, '边界\n区域', fontsize=9, color='orange')
    ax.text(9.5, ax.get_ylim()[1]*0.85, '边界\n区域', fontsize=9, color='orange')
    
    for xi in knots:
        ax.axvline(xi, color='gray', ls=':', alpha=0.4, lw=0.7)
    
    ax.set_title(f'标准三次样条\n($K={K}$ 个节点, 边界行为无约束)', fontsize=12)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    
    # --- 右: 自然三次样条 ---
    ax = axes[1]
    
    # 构造自然样条基
    from scipy.interpolate import BSpline
    # 使用 scipy 的 make_lsq_spline 构造自然样条
    # 自然边界条件: 二阶导数在两端点为 0
    t_nat = np.concatenate([[knots[0]]*3, knots[1:-1], [knots[-1]]*3])
    # 构造 B-spline 基
    bsp = BSpline.basis_element(t_nat)
    
    # 更简单的方法: 用 splrep 配合 s 参数做平滑, 然后用 BSpline.from_spline
    # 实际上 scipy 的 make_lsq_spline 无法直接做自然样条,
    # 我们用 statsmodels 或手动构造自然样条基 + 最小二乘
    
    # 手动自然三次样条基 (复用 01 的实现)
    def natural_spline_basis(x_data, knots):
        K = len(knots)
        N = np.zeros((len(x_data), K))
        N[:, 0] = 1.0
        N[:, 1] = x_data
        
        def d_k(k_idx):
            xi_k = knots[k_idx]
            xi_K = knots[-1]
            p1 = np.maximum(x_data - xi_k, 0)**3
            p2 = np.maximum(x_data - xi_K, 0)**3
            return (p1 - p2) / (xi_K - xi_k)
        
        for k in range(K - 2):
            N[:, k + 2] = d_k(k) - d_k(K - 2)
        return N
    
    B_train = natural_spline_basis(x, knots)
    coeff_nat = np.linalg.lstsq(B_train, y, rcond=None)[0]
    
    B_dense = natural_spline_basis(x_dense, knots)
    y_nat_dense = B_dense @ coeff_nat
    
    ax.plot(x_dense, f_true_dense, 'k-', lw=2, alpha=0.3, label='真实函数 $f(x)$')
    ax.scatter(x, y, c='gray', s=20, alpha=0.6, zorder=5, label='观测数据')
    ax.plot(x_dense, y_nat_dense, 'r-', lw=2, label='自然三次样条')
    
    ax.axvspan(-0.5, knots[0], color='green', alpha=0.1)
    ax.axvspan(knots[-1], 10.5, color='green', alpha=0.1)
    ax.text(0.0, ax.get_ylim()[1]*0.85, '线性\n外推', fontsize=9, color='green')
    ax.text(9.5, ax.get_ylim()[1]*0.85, '线性\n外推', fontsize=9, color='green')
    
    for xi in knots:
        ax.axvline(xi, color='gray', ls=':', alpha=0.4, lw=0.7)
    
    ax.set_title(f'自然三次样条\n($K={K}$ 个节点, 边界外线性约束)', fontsize=12)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "02_spline_standard_vs_natural.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"图 1 已保存: {save_path}")
    
    # ============================================================
    # 图 2: Bootstrap 模拟 —— 边界方差对比
    # ============================================================
    n_bootstrap = 200
    rng = np.random.default_rng(2024)
    
    # 固定节点
    K_bs = 6
    knots_bs = np.quantile(x, np.linspace(0, 1, K_bs))
    
    # 存储多次拟合结果
    fits_standard = np.zeros((n_bootstrap, len(x_dense)))
    fits_natural = np.zeros((n_bootstrap, len(x_dense)))
    
    for b in range(n_bootstrap):
        # 残差 Bootstrap
        # 先用自然样条得到初始拟合
        B_init = natural_spline_basis(x, knots_bs)
        coeff_init = np.linalg.lstsq(B_init, y, rcond=None)[0]
        y_hat_init = B_init @ coeff_init
        residuals = y - y_hat_init
        
        # 重抽样残差
        idx = rng.choice(len(x), len(x), replace=True)
        y_boot = y_hat_init + residuals[idx]
        
        # 标准三次样条: 插值
        tck_boot = splrep(x, y_boot, k=3, s=0.002)
        fits_standard[b] = splev(x_dense, tck_boot)
        
        # 自然三次样条
        B_boot = natural_spline_basis(x, knots_bs)
        coeff_boot = np.linalg.lstsq(B_boot, y_boot, rcond=None)[0]
        fits_natural[b] = natural_spline_basis(x_dense, knots_bs) @ coeff_boot
    
    # 计算方差和均值
    mean_std = np.mean(fits_standard, axis=0)
    std_std = np.std(fits_standard, axis=0)
    mean_nat = np.mean(fits_natural, axis=0)
    std_nat = np.std(fits_natural, axis=0)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 上左: 标准样条 Bootstrap
    ax = axes[0, 0]
    ax.plot(x_dense, f_true_dense, 'k-', lw=2.5, alpha=0.4, label='真实 $f(x)$')
    ax.scatter(x, y, c='gray', s=15, alpha=0.5, zorder=5)
    for b in range(min(n_bootstrap, 50)):
        ax.plot(x_dense, fits_standard[b], 'b-', lw=0.3, alpha=0.15)
    ax.plot(x_dense, mean_std, 'b-', lw=2, label='Bootstrap 均值')
    ax.fill_between(x_dense, mean_std - 2*std_std, mean_std + 2*std_std,
                     color='blue', alpha=0.1, label='±2 SE')
    ax.set_title('标准三次样条: Bootstrap 模拟', fontsize=13)
    ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
    
    # 上右: 自然样条 Bootstrap
    ax = axes[0, 1]
    ax.plot(x_dense, f_true_dense, 'k-', lw=2.5, alpha=0.4, label='真实 $f(x)$')
    ax.scatter(x, y, c='gray', s=15, alpha=0.5, zorder=5)
    for b in range(min(n_bootstrap, 50)):
        ax.plot(x_dense, fits_natural[b], 'r-', lw=0.3, alpha=0.15)
    ax.plot(x_dense, mean_nat, 'r-', lw=2, label='Bootstrap 均值')
    ax.fill_between(x_dense, mean_nat - 2*std_nat, mean_nat + 2*std_nat,
                     color='red', alpha=0.1, label='±2 SE')
    ax.set_title('自然三次样条: Bootstrap 模拟', fontsize=13)
    ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
    
    # 下: 逐点标准差对比 (核心!)
    ax = axes[1, 0]
    ax.plot(x_dense, std_std, 'b-', lw=2, label='标准三次样条')
    ax.plot(x_dense, std_nat, 'r-', lw=2, label='自然三次样条')
    ax.axvspan(-0.5, knots_bs[0], color='orange', alpha=0.08)
    ax.axvspan(knots_bs[-1], 10.5, color='orange', alpha=0.08)
    ax.text(0.2, np.max(std_std)*0.9, '边界\n区域', fontsize=9, color='orange')
    ax.text(9.3, np.max(std_std)*0.9, '边界\n区域', fontsize=9, color='orange')
    ax.set_xlabel('$x$', fontsize=12)
    ax.set_ylabel('逐点标准差 SE$(\\hat{f}(x))$', fontsize=12)
    ax.set_title('逐点标准差对比: 自然样条在边界处方差大幅降低', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    
    # 下右: 方差比值
    ax = axes[1, 1]
    ratio = std_nat / (std_std + 1e-10)
    ax.plot(x_dense, ratio, 'purple', lw=2)
    ax.axhline(1.0, color='gray', ls='--', lw=1)
    ax.axhline(0.5, color='gray', ls=':', lw=0.8)
    ax.axvspan(-0.5, knots_bs[0], color='orange', alpha=0.08)
    ax.axvspan(knots_bs[-1], 10.5, color='orange', alpha=0.08)
    ax.set_xlabel('$x$', fontsize=12)
    ax.set_ylabel('方差比: Var(自然) / Var(标准)', fontsize=12)
    ax.set_title('方差比值 (越小 = 自然样条越稳定)', fontsize=13)
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    save_path2 = os.path.join(save_dir, "02_spline_boundary_variance_bootstrap.png")
    fig.savefig(save_path2, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"图 2 已保存: {save_path2}")
    
    print("\n=== 核心总结 ===")
    print("• 标准三次样条: 边界区域不受约束 → 方差爆增")
    print("• 自然三次样条: 边界外强制线性 → 边界方差大幅降低")
    print("• 代价: 边界附近偏差略有增加 (假设边界处近似线性通常是合理的)")
    print("• 净收益: 释放 4 个自由度 → 更多节点可放在内部 → 整体拟合更好")
