"""
01_natural_spline_basis.py
演示自然三次样条的基函数形态

关键概念:
- 标准三次样条: K 个节点 → 4+K 个基函数 (截断幂基)
- 自然三次样条: K 个节点 → K 个基函数
  (添加边界线性约束, 释放 4 个自由度)
- 每个自然样条基函数在边界之外具有零二阶和三阶导数
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
import os

# ============================================================
# 0. 设置
# ============================================================
save_dir = os.path.join(os.path.dirname(__file__), "pic")
os.makedirs(save_dir, exist_ok=True)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 构造自然三次样条基函数（按 ESL 公式 (5.4)）
# ============================================================

def natural_cubic_spline_basis(x, knots):
    """
    构造自然三次样条基函数 N_1(x), ..., N_K(x).
    
    基函数定义 (ESL 习题 5.4):
        N_1(X) = 1
        N_2(X) = X
        N_{k+2}(X) = d_k(X) - d_{K-1}(X),   k = 1,...,K-2
    
    其中 d_k(X) = [(X-ξ_k)_+^3 - (X-ξ_K)_+^3] / (ξ_K - ξ_k)
    
    参数:
        x: 输入点 (1D array)
        knots: 节点序列 [ξ_1, ξ_2, ..., ξ_K] (1D array)
    
    返回:
        basis: shape (len(x), K) 的基矩阵
    """
    x = np.asarray(x, dtype=float)
    knots = np.asarray(knots, dtype=float)
    K = len(knots)
    
    N = np.zeros((len(x), K))
    
    # N_1(x) = 1
    N[:, 0] = 1.0
    
    # N_2(x) = x
    N[:, 1] = x
    
    # 计算 d_k(x) for k = 1, ..., K-1
    def d_k(k):
        """d_k(X) = [(X-ξ_k)_+^3 - (X-ξ_K)_+^3] / (ξ_K - ξ_k)"""
        xi_k = knots[k]       # 0-indexed: k 从 0 到 K-2
        xi_K = knots[-1]      # 最后一个节点
        pos1 = np.maximum(x - xi_k, 0) ** 3
        pos2 = np.maximum(x - xi_K, 0) ** 3
        return (pos1 - pos2) / (xi_K - xi_k)
    
    # N_{k+2}(X) = d_k(X) - d_{K-1}(X), k = 1,...,K-2
    for k in range(K - 2):  # k = 0, 1, ..., K-3
        N[:, k + 2] = d_k(k) - d_k(K - 2)  # d_k - d_{K-1}
    
    return N


def check_boundary_linearity(x, basis, knots):
    """
    验证自然三次样条基函数在边界之外是线性的。
    即: 对于 x > ξ_K, 二阶导数 = 0 且三阶导数 = 0。
    """
    beyond_right = x > knots[-1]
    if not np.any(beyond_right):
        return
    
    # 在右侧边界外取密集点, 检查是否线性
    x_right = np.linspace(knots[-1], knots[-1] + 3, 100)
    basis_right = natural_cubic_spline_basis(x_right, knots)
    
    # 线性函数的一阶差分为常数, 二阶差分为 0
    for j in range(basis_right.shape[1]):
        vals = basis_right[:, j]
        first_diff = np.diff(vals) / np.diff(x_right)
        second_diff = np.diff(first_diff) / np.diff(x_right[:-1])
        linear_error = np.max(np.abs(second_diff))
        if linear_error > 1e-6:
            print(f"  基函数 N_{j+1}: 边界外非线性, 误差={linear_error:.2e}")
    
    return x_right, basis_right


# ============================================================
# 2. 主程序
# ============================================================
if __name__ == "__main__":
    # --- 参数 ---
    K = 5                                    # 节点数
    knots = np.array([0.0, 1.0, 2.0, 3.0, 4.0])  # 均匀节点
    
    x_plot = np.linspace(-1, 6, 500)         # 扩展示意图 (包含边界外)
    
    # 构造基矩阵
    basis = natural_cubic_spline_basis(x_plot, knots)
    
    # ============================================================
    # 图 1: K 个自然三次样条基函数
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = plt.cm.tab10(np.linspace(0, 1, K))
    
    # 左: 所有基函数
    ax = axes[0]
    for j in range(K):
        ax.plot(x_plot, basis[:, j], color=colors[j], lw=2,
                label=f'$N_{j+1}(x)$')
    
    # 标注节点
    for i, xi in enumerate(knots):
        ax.axvline(xi, color='gray', ls='--', alpha=0.4, lw=0.8)
    
    ax.axvspan(knots[-1], x_plot[-1], color='green', alpha=0.06)
    ax.text(knots[-1] + 0.3, ax.get_ylim()[1]*0.9, '边界外\n(线性)',
            fontsize=9, color='green')
    ax.axvspan(x_plot[0], knots[0], color='green', alpha=0.06)
    ax.text(knots[0] - 0.8, ax.get_ylim()[1]*0.9, '边界外\n(线性)',
            fontsize=9, color='green')
    
    ax.set_xlabel('$x$', fontsize=12)
    ax.set_ylabel('基函数值', fontsize=12)
    ax.set_title(f'自然三次样条的 {K} 个基函数\n($K={K}$ 个节点)', fontsize=13)
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.2)
    
    # 右: 逐个基函数展示 + 二阶导数
    ax = axes[1]
    ax2 = ax.twinx()
    
    # 选取一个代表性的基函数 (如 N_3)
    j_show = 2
    ax.plot(x_plot, basis[:, j_show], 'b-', lw=2.5,
            label=f'$N_{j_show+1}(x)$')
    
    # 数值二阶导数
    dx = x_plot[1] - x_plot[0]
    second_deriv = np.gradient(np.gradient(basis[:, j_show], dx), dx)
    ax2.plot(x_plot, second_deriv, 'r--', lw=1.5, alpha=0.7,
             label=f"$N_{j_show+1}''(x)$")
    
    for i, xi in enumerate(knots):
        ax.axvline(xi, color='gray', ls='--', alpha=0.3, lw=0.8)
    
    ax.axhline(0, color='black', lw=0.5)
    ax2.axhline(0, color='red', ls=':', lw=0.5, alpha=0.5)
    
    # 边界外二阶导数应为 0
    ax2.axvspan(knots[-1], x_plot[-1], color='green', alpha=0.08)
    ax2.axvspan(x_plot[0], knots[0], color='green', alpha=0.08)
    
    ax.set_xlabel('$x$', fontsize=12)
    ax.set_ylabel('基函数值', fontsize=12, color='blue')
    ax2.set_ylabel('二阶导数', fontsize=12, color='red')
    ax.set_title(f'$N_{j_show+1}(x)$ 及其二阶导数\n'
                 f'(边界外二阶导数为零 → 线性)', fontsize=13)
    
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1+lines2, labels1+labels2, loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "01_natural_spline_basis_functions.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"图 1 已保存: {save_path}")
    
    # 验证边界线性性
    print("\n=== 验证自然三次样条边界线性性 ===")
    x_right, basis_right = check_boundary_linearity(x_plot, basis, knots)
    print("  (若上方无报错, 则所有基函数在边界外均为线性)")
    
    # ============================================================
    # 图 2: 对比不同 K 值下的基函数数量与自由度
    # ============================================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    for idx, K_val in enumerate([3, 5, 7, 9]):
        ax = axes[idx // 2][idx % 2]
        knots_val = np.linspace(0, 5, K_val)
        x_val = np.linspace(-1, 7, 400)
        basis_val = natural_cubic_spline_basis(x_val, knots_val)
        
        for j in range(K_val):
            ax.plot(x_val, basis_val[:, j], lw=1.5, alpha=0.8)
        
        for xi in knots_val:
            ax.axvline(xi, color='gray', ls=':', alpha=0.3, lw=0.6)
        
        ax.axvline(0, color='black', lw=1.2)
        ax.axhline(0, color='black', lw=0.5)
        ax.set_title(f'$K={K_val}$ 个节点 → $K={K_val}$ 个基函数', fontsize=12)
        ax.set_xlabel('$x$')
        ax.set_ylabel('基函数值')
        ax.grid(True, alpha=0.2)
    
    plt.suptitle('自然三次样条: 不同节点数下的基函数族', fontsize=14, y=1.01)
    plt.tight_layout()
    save_path2 = os.path.join(save_dir, "01_natural_spline_basis_varying_K.png")
    fig.savefig(save_path2, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"图 2 已保存: {save_path2}")
    
    print("\n=== 核心总结 ===")
    print(f"• K={K} 个节点 → K 个基函数 (标准三次样条需要 K+4 个)")
    print("• 边界约束释放 4 个自由度, 重新分配到内部区域")
    print("• 边界外二阶/三阶导数 = 0 → 线性外推")
    print("• 基函数 N_1=1 (截距), N_2=x (线性趋势), N_3~N_K 提供非线性弯曲")
