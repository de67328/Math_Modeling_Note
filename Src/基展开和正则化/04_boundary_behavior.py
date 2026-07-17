"""
04_boundary_behavior.py
重点演示自然三次样条的边界行为

ESL §5.2 核心论点:
- 标准三次样条在边界区域方差爆增 (ESL 图 5.3)
- 自然三次样条在边界外强制线性 → 牺牲少量偏差, 大幅降低方差
- 释放 4 个自由度, 可在内部放更多节点

本程序:
1. 展示标准样条边界方差 vs 自然样条边界方差
2. 量化自由度分配
3. 展示残差分布对比
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

save_dir = os.path.join(os.path.dirname(__file__), "pic")
os.makedirs(save_dir, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


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


def truncated_power_basis(x, knots, degree=3):
    """
    截断幂基 (标准三次样条):
    h_1(x)=1, h_2(x)=x, h_3(x)=x^2, h_4(x)=x^3,
    h_{4+j}(x) = (x - ξ_j)_+^3, j=1,...,K
    """
    K = len(knots)
    M = degree + 1 + K  # 4 + K 个基函数
    H = np.zeros((len(x), M))
    H[:, 0] = 1
    H[:, 1] = x
    H[:, 2] = x**2
    H[:, 3] = x**3
    for j in range(K):
        H[:, 4 + j] = np.maximum(x - knots[j], 0)**3
    return H


if __name__ == "__main__":
    # ============================================================
    # 模拟设置: 多次重复拟合, 量化边界方差
    # ============================================================
    rng = np.random.default_rng(42)
    n_sim = 500        # 模拟次数
    n_data = 60        # 每次模拟的数据量
    
    # 真实函数 (简单的正弦, 边界处有弯曲)
    def f_true(x):
        return np.sin(x) + 0.3 * np.cos(2*x)
    
    x_dense = np.linspace(-0.3, 10.3, 300)
    f_dense = f_true(x_dense)
    
    # 边界标记
    boundary_left = 0.0
    boundary_right = 10.0
    
    K = 6
    knots_uniform = np.linspace(1, 9, K)   # 离边界留距离, 让边界效应更明显
    
    # ============================================================
    # 多次模拟
    # ============================================================
    fits_standard = np.zeros((n_sim, len(x_dense)))
    fits_natural = np.zeros((n_sim, len(x_dense)))
    
    for sim in range(n_sim):
        x_sim = np.sort(rng.uniform(0.2, 9.8, n_data))
        y_sim = f_true(x_sim) + rng.normal(0, 0.35, n_data)
        
        # 标准三次样条 (截断幂基)
        H_train = truncated_power_basis(x_sim, knots_uniform, degree=3)
        coeff_std = np.linalg.lstsq(H_train, y_sim, rcond=None)[0]
        H_dense = truncated_power_basis(x_dense, knots_uniform, degree=3)
        fits_standard[sim] = H_dense @ coeff_std
        
        # 自然三次样条
        B_train = natural_spline_basis(x_sim, knots_uniform)
        coeff_nat = np.linalg.lstsq(B_train, y_sim, rcond=None)[0]
        B_dense = natural_spline_basis(x_dense, knots_uniform)
        fits_natural[sim] = B_dense @ coeff_nat
    
    mean_std = np.mean(fits_standard, axis=0)
    std_std = np.std(fits_standard, axis=0)
    bias_std = mean_std - f_dense
    
    mean_nat = np.mean(fits_natural, axis=0)
    std_nat = np.std(fits_natural, axis=0)
    bias_nat = mean_nat - f_dense
    
    # ============================================================
    # 主图: 四面板综合对比
    # ============================================================
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig,
                            width_ratios=[1, 1, 0.7],
                            hspace=0.35, wspace=0.35)
    
    # ---- 面板 A: 标准样条 50 次拟合叠加 ----
    ax = fig.add_subplot(gs[0, 0])
    for sim in range(min(n_sim, 50)):
        ax.plot(x_dense, fits_standard[sim], 'b-', lw=0.3, alpha=0.15)
    ax.plot(x_dense, f_dense, 'k-', lw=2.5, label='真实 $f(x)$')
    ax.plot(x_dense, mean_std, 'blue', lw=2, ls='--', label='模拟均值')
    ax.axvline(boundary_left, color='orange', lw=1.2, alpha=0.7)
    ax.axvline(boundary_right, color='orange', lw=1.2, alpha=0.7)
    ax.axvspan(-0.3, boundary_left, color='orange', alpha=0.08)
    ax.axvspan(boundary_right, 10.3, color='orange', alpha=0.08)
    ax.set_title(f'A. 标准三次样条\n({K}+4={K+4} 个基函数)', fontsize=12)
    ax.set_xlabel('$x$'); ax.set_ylabel('$\\hat{f}(x)$')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
    ax.set_ylim(-2.5, 2.5)
    
    # ---- 面板 B: 自然样条 50 次拟合叠加 ----
    ax = fig.add_subplot(gs[0, 1])
    for sim in range(min(n_sim, 50)):
        ax.plot(x_dense, fits_natural[sim], 'r-', lw=0.3, alpha=0.15)
    ax.plot(x_dense, f_dense, 'k-', lw=2.5, label='真实 $f(x)$')
    ax.plot(x_dense, mean_nat, 'red', lw=2, ls='--', label='模拟均值')
    ax.axvline(boundary_left, color='green', lw=1.2, alpha=0.7)
    ax.axvline(boundary_right, color='green', lw=1.2, alpha=0.7)
    ax.axvspan(-0.3, boundary_left, color='green', alpha=0.08)
    ax.axvspan(boundary_right, 10.3, color='green', alpha=0.08)
    ax.set_title(f'B. 自然三次样条\n($K={K}$ 个基函数)', fontsize=12)
    ax.set_xlabel('$x$'); ax.set_ylabel('$\\hat{f}(x)$')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
    ax.set_ylim(-2.5, 2.5)
    
    # ---- 面板 C: 逐点标准差对比 (图 5.3 的精神) ----
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(x_dense, std_std, 'b-', lw=2, label='标准三次样条')
    ax.plot(x_dense, std_nat, 'r-', lw=2, label='自然三次样条')
    ax.fill_between(x_dense, 0, std_std, color='blue', alpha=0.08)
    ax.fill_between(x_dense, 0, std_nat, color='red', alpha=0.08)
    ax.axvline(boundary_left, color='gray', lw=0.8, ls='--')
    ax.axvline(boundary_right, color='gray', lw=0.8, ls='--')
    
    # 标注边界区域
    y_max_std = np.max(std_std) * 1.05
    ax.annotate('边界方差爆增!', xy=(0.2, np.max(std_std[x_dense < 1.5])),
                xytext=(1.5, y_max_std*0.85),
                arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
                fontsize=10, color='blue')
    ax.annotate('方差稳定', xy=(0.2, np.max(std_nat[x_dense < 1.5])),
                xytext=(1.5, y_max_std*0.5),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=10, color='red')
    
    ax.set_xlabel('$x$', fontsize=12)
    ax.set_ylabel('SE$(\\hat{f}(x))$', fontsize=12)
    ax.set_title('C. 逐点标准差: 自然样条消除边界方差爆增', fontsize=13)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.2)
    
    # ---- 面板 D: 偏差平方对比 ----
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(x_dense, bias_std**2, 'b-', lw=2, label='标准样条偏差²')
    ax.plot(x_dense, bias_nat**2, 'r-', lw=2, label='自然样条偏差²')
    ax.axvline(boundary_left, color='gray', lw=0.8, ls='--')
    ax.axvline(boundary_right, color='gray', lw=0.8, ls='--')
    ax.set_xlabel('$x$', fontsize=12)
    ax.set_ylabel('Bias^2(f_hat(x))', fontsize=12)
    ax.set_title('D. 偏差平方: 自然样条边界偏差略增', fontsize=13)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.2)
    
    # ---- 面板 E: 综合指标对比表 ----
    ax = fig.add_subplot(gs[:, 2])
    ax.axis('off')
    
    # 分区域统计
    regions = {
        '左边界 x < 1': (x_dense < 1.0),
        '内部 1 ≤ x ≤ 9': ((x_dense >= 1.0) & (x_dense <= 9.0)),
        '右边界 x > 9': (x_dense > 9.0),
        '整体': np.ones(len(x_dense), dtype=bool),
    }
    
    table_data = []
    for name, mask in regions.items():
        mse_std = np.mean((mean_std[mask] - f_dense[mask])**2 + std_std[mask]**2)
        mse_nat = np.mean((mean_nat[mask] - f_dense[mask])**2 + std_nat[mask]**2)
        var_std_avg = np.mean(std_std[mask]**2)
        var_nat_avg = np.mean(std_nat[mask]**2)
        table_data.append([
            name,
            f'{var_std_avg:.4f}',
            f'{var_nat_avg:.4f}',
            f'{var_nat_avg/var_std_avg*100:.0f}%',
            f'{mse_std:.4f}',
            f'{mse_nat:.4f}',
        ])
    
    col_labels = ['区域', 'Var(标准)', 'Var(自然)', '比值', 'MSE(标准)', 'MSE(自然)']
    
    table = ax.table(cellText=table_data,
                     colLabels=col_labels,
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.15, 0.12, 0.12, 0.09, 0.12, 0.12])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.8)
    
    # 给标题行和结果行着色
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#40466e')
        table[0, j].set_text_props(color='white', fontweight='bold')
    
    for i in range(1, len(table_data) + 1):
        for j in range(len(col_labels)):
            if i == len(table_data):  # 整体行
                table[i, j].set_facecolor('#e8e8e8')
    
    ax.set_title('E. 分区域方差与 MSE 对比\n'
                 f'($K={K}$, {n_sim} 次模拟)',
                 fontsize=11, y=0.85)
    
    plt.suptitle('自然三次样条的边界行为: 方差-偏差权衡',
                 fontsize=14, y=0.98)
    
    save_path = os.path.join(save_dir, "04_boundary_behavior_comprehensive.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"图已保存: {save_path}")
    
    # ============================================================
    # 报告核心数字
    # ============================================================
    mask_boundaries = (x_dense < 1.0) | (x_dense > 9.0)
    mask_interior = (x_dense >= 1.0) & (x_dense <= 9.0)
    
    var_reduction_boundary = (1 - np.mean(std_nat[mask_boundaries]**2) 
                              / np.mean(std_std[mask_boundaries]**2)) * 100
    var_reduction_interior = (1 - np.mean(std_nat[mask_interior]**2)
                              / np.mean(std_std[mask_interior]**2)) * 100
    
    print(f"\n=== 边界方差降低 ===")
    print(f"• 边界区域: 方差降低 {var_reduction_boundary:.1f}%")
    print(f"• 内部区域: 方差降低 {var_reduction_interior:.1f}%")
    print(f"\n=== 自由度分配 ===")
    print(f"• 标准三次样条: K+4 = {K+4} 个基函数 (4 个浪费在边界)")
    print(f"• 自然三次样条: K = {K} 个基函数 (4 个重分配到内部)")
    print(f"• 等价于: 相同自由度下, 自然样条可在内部多放节点 → 更好拟合信号")
