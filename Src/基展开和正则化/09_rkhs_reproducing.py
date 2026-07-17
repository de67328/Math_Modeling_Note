"""
09_rkhs_reproducing.py
RKHS 的再生性与特征映射

关键概念:
1. 再生性: f(x) = <f, K(·, x)>_H  — 求值 = 内积
2. 有限维特征映射 φ(x) 可以显式构造 (对某些核)
3. 核技巧: 内积 φ(x)^T φ(y) = K(x, y) — 只需算核, 不需算 φ
4. RKHS 范数的含义: 偏好"简单"函数

ESL §5.8
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh, solve
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


def polynomial_kernel(X, Y=None, degree=2):
    if Y is None:
        Y = X
    return (X @ Y.T + 1.0) ** degree


# ============================================================
# 1. 构造多项式核的显式特征映射
# ============================================================
def poly_feature_map_2d(X, degree=2):
    """
    二维输入 x = (x1, x2) → d=2 多项式核的显式特征。
    
    K(x, y) = (x1*y1 + x2*y2 + 1)^2
            = 1 + 2x1*y1 + 2x2*y2 + x1^2*y1^2 + x2^2*y2^2 + 2x1*x2*y1*y2
            = φ(x)^T φ(y)
    
    其中 φ(x) = [1, √2 x1, √2 x2, x1^2, x2^2, √2 x1 x2]^T
    """
    x1, x2 = X[:, 0], X[:, 1]
    return np.column_stack([
        np.ones(len(X)),
        np.sqrt(2) * x1,
        np.sqrt(2) * x2,
        x1**2,
        x2**2,
        np.sqrt(2) * x1 * x2,
    ])


# ============================================================
# 2. RKHS 范数的近似估计 (对高斯核)
# ============================================================
def approx_rkhs_norm(alpha, K):
    """
    给定核岭回归解 f(x) = Σ α_i K(x, x_i),
    其 RKHS 范数为 ||f||_H^2 = α^T K α
    """
    return np.sqrt(alpha.T @ K @ alpha)


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    rng = np.random.default_rng(123)
    
    # ============================================================
    # 图 1: 显式特征映射 vs 核技巧 (多项式核 d=2)
    # ============================================================
    n = 6
    X_small = rng.uniform(-2, 2, (n, 2))
    
    # 方法 A: 显式特征映射
    Phi = poly_feature_map_2d(X_small, degree=2)
    K_explicit = Phi @ Phi.T
    
    # 方法 B: 核函数直接算
    K_direct = polynomial_kernel(X_small, degree=2)
    
    diff = np.max(np.abs(K_explicit - K_direct))
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    # 左: 原始数据
    ax = axes[0]
    ax.scatter(X_small[:, 0], X_small[:, 1], c='blue', s=80, zorder=5)
    for i in range(n):
        ax.annotate(f'$x_{i+1}$', (X_small[i, 0], X_small[i, 1]),
                    textcoords="offset points", xytext=(8, 5), fontsize=9)
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
    ax.set_title(f'原始输入 (n={n}, d=2)\n在 $\\mathbb{{R}}^2$ 中线性不可分', fontsize=12)
    ax.grid(True, alpha=0.2)
    ax.set_aspect('equal')
    
    # 中: 特征空间示意
    ax = axes[1]
    # 只可视化前 3 个特征维度
    ax.scatter(Phi[:, 1], Phi[:, 2], c='red', s=80, zorder=5)
    for i in range(n):
        ax.annotate(f'$x_{i+1}$', (Phi[i, 1], Phi[i, 2]),
                    textcoords="offset points", xytext=(8, 5), fontsize=9)
    
    # 画一条可能的分隔线
    xx = np.linspace(Phi[:, 1].min()-1, Phi[:, 1].max()+1, 50)
    ax.plot(xx, 2 * xx - 1, 'g--', lw=2, alpha=0.6, label='可能的线性分界')
    ax.set_xlabel('$\\phi_1 = \\sqrt{2}x_1$', fontsize=11)
    ax.set_ylabel('$\\phi_2 = \\sqrt{2}x_2$', fontsize=11)
    ax.set_title(f'多项式核 (d=2) 的特征空间\n'
                 f'(高维中可线性分离)', fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    
    # 右: 两种方法对比
    ax = axes[2]
    ax.scatter(K_direct.ravel(), K_explicit.ravel(), c='purple', s=30, alpha=0.7)
    ax.plot([0, K_direct.max()], [0, K_direct.max()], 'k--', lw=1)
    ax.set_xlabel('核函数 $K(x_i, x_j)$', fontsize=11)
    ax.set_ylabel('特征内积 $\\phi(x_i)^T\\phi(y_j)$', fontsize=11)
    ax.set_title(f'核技巧验证: 最大误差 = {diff:.1e}\n'
                 f'(核计算 = 特征映射内积)', fontsize=12)
    ax.grid(True, alpha=0.2)
    
    plt.suptitle('显式特征映射 vs 核技巧\n'
                 f'$K(x,y) = (x^T y + 1)^2 = \\phi(x)^T\\phi(y)$',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    save_path = os.path.join(save_dir, "09_feature_map_vs_kernel.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"图 1 已保存: {save_path}")
    print(f"  核技巧验证: ||K_explicit - K_direct||_∞ = {diff:.2e}")
    
    # ============================================================
    # 图 2: RKHS 范数 — 不同复杂度函数的惩罚
    # ============================================================
    x_train = np.linspace(-3, 3, 30).reshape(-1, 1)
    y_smooth = np.sin(x_train).ravel()
    y_rough = (np.sin(4 * x_train) + 0.5 * np.cos(7 * x_train)).ravel()
    y_noisy = y_smooth + 0.3 * rng.normal(0, 1, len(x_train))
    
    # 用几个不同 gamma 拟合这几个数据集
    gammas = [0.1, 0.5, 2.0]
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    for row, (y_data, y_name) in enumerate([
        (y_smooth, '光滑信号 $\\sin(x)$'),
        (y_rough, '崎岖信号 $\\sin(4x)+0.5\\cos(7x)$'),
        (y_noisy, '噪声信号 $\\sin(x)+\\varepsilon$'),
    ]):
        for col, gamma in enumerate(gammas):
            ax = axes[row][col]
            
            # 核岭回归
            K = gaussian_kernel(x_train, gamma=gamma)
            lam = 0.01
            alpha = solve(K + lam * np.eye(len(x_train)), y_data, assume_a='pos')
            
            # 预测
            x_plot = np.linspace(-3.5, 3.5, 300).reshape(-1, 1)
            K_plot = gaussian_kernel(x_plot, x_train, gamma=gamma)
            y_plot = K_plot @ alpha
            
            # RKHS 范数
            norm_f = approx_rkhs_norm(alpha, K)
            
            ax.scatter(x_train, y_data, c='gray', s=12, alpha=0.5, zorder=5)
            ax.plot(x_plot, y_plot, 'r-', lw=2)
            ax.set_title(f'$\\gamma={gamma}$: $\\|f\\|_H$ = {norm_f:.2f}',
                         fontsize=10)
            ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
            ax.grid(True, alpha=0.2)
        
        # 行标签
        axes[row][0].set_ylabel(y_name, fontsize=11)
    
    # 列标签
    for col, gamma in enumerate(gammas):
        axes[0][col].set_title(f'$\\gamma={gamma}$', fontsize=11, y=1.05)
    
    plt.suptitle('RKHS 范数的含义: $\\gamma$ 和信号复杂度对 $\\|f\\|_H$ 的影响\n'
                 '(范数越大 → 函数越"不光滑" → 正则化惩罚越重)',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    save_path2 = os.path.join(save_dir, "09_rkhs_norm_penalty.png")
    fig.savefig(save_path2, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"图 2 已保存: {save_path2}")
    
    # ============================================================
    # 图 3: 演示再生性 f(x) = <f, K(·, x)>_H
    # ============================================================
    # 用离散近似: 在密集格点上验证
    x_grid = np.linspace(-2, 2, 200).reshape(-1, 1)
    gamma_demo = 1.5
    
    # 取一个已知函数: f(x) = sin(2x) + 0.5 cos(3x), 用核岭回归学习
    y_demo = np.sin(2 * x_grid.ravel()) + 0.5 * np.cos(3 * x_grid.ravel())
    
    # 核岭回归拟合
    K_grid = gaussian_kernel(x_grid, gamma=gamma_demo)
    lam_demo = 1e-4
    alpha_demo = solve(K_grid + lam_demo * np.eye(len(x_grid)), y_demo,
                       assume_a='pos')
    
    # 验证再生性于 3 个测试点
    test_points = np.array([[-1.5], [0.0], [1.5]])
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # 上: 拟合的函数
    ax = axes[0]
    ax.plot(x_grid, y_demo, 'k-', lw=1, alpha=0.3, label='目标 $f(x)$')
    f_hat = K_grid @ alpha_demo
    ax.plot(x_grid, f_hat, 'b-', lw=2, label='核岭回归 $\\hat{f}(x)$')
    
    for i, x_test in enumerate(test_points):
        # 再生性: <f, K(·, x_test)>_H = Σ_i α_i K(x_i, x_test) = f(x_test)
        # 因为 K @ alpha = f_hat(x_train), 在新点 x_test:
        # f_hat(x_test) = Σ_j K(x_test, x_j) α_j = K_test^T α
        k_test = gaussian_kernel(x_test.reshape(1, -1), x_grid, gamma=gamma_demo)
        f_at_test = (k_test @ alpha_demo)[0]
        
        ax.plot(x_test[0], f_at_test, 'ro', ms=10, zorder=10)
        ax.annotate(f'$\\hat{{f}}({x_test[0]:.1f})={f_at_test:.3f}$',
                    (x_test[0], f_at_test),
                    textcoords="offset points", xytext=(15, 15),
                    fontsize=9, color='red',
                    arrowprops=dict(arrowstyle='->', color='red'))
    
    ax.set_xlabel('$x$', fontsize=12); ax.set_ylabel('$f(x)$', fontsize=12)
    ax.set_title('再生性验证: $\\hat{f}(x_{test}) = \\langle \\hat{f}, K(\\cdot, x_{test})\\rangle_{\\mathcal{H}}$',
                 fontsize=13)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.2)
    
    # 下: 内积侧 — 展示 K(·, x_test) 和 α
    ax = axes[1]
    
    colors_test = ['#e74c3c', '#2ecc71', '#3498db']
    for i, (x_test, c) in enumerate(zip(test_points, colors_test)):
        k_vec = gaussian_kernel(x_grid, x_test.reshape(1, -1), gamma=gamma_demo)
        ax.plot(x_grid, k_vec.ravel(), color=c, lw=2, alpha=0.7,
                label=f'$K(\\cdot, x_{{test}}={x_test[0]:.1f})$')
        # 加权
        weighted = k_vec.ravel() * alpha_demo.ravel()
        ax.fill_between(x_grid.ravel(), 0, weighted, color=c, alpha=0.08)
    
    ax.set_xlabel('$x$', fontsize=12)
    ax.set_ylabel('核函数值', fontsize=12)
    ax.set_title('再生性的直观: $\\hat{f}(x_{test}) = \\sum_j \\alpha_j K(x_j, x_{test})$\n'
                 '(阴影面积 = $\\alpha_j$ 加权后的贡献)', fontsize=13)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    save_path3 = os.path.join(save_dir, "09_reproducing_property.png")
    fig.savefig(save_path3, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"图 3 已保存: {save_path3}")
    
    print("\n=== 核心总结 ===")
    print("• 核技巧: K(x,y) = φ(x)^T φ(y) — 隐式使用高维/无限维特征")
    print("• 多项式核 (d=2): 2D → 6D 显式映射, 核计算 = 内积 (无误)")
    print("• RKHS 范数: ||f||_H = sqrt(α^T K α), 大 → 函数复杂 → 惩罚重")
    print("• γ 越大: 特征值衰减越慢, 可表示更复杂的函数, 范数更大")
    print("• 再生性: 求值 = α 与核的加权和, 表示定理的几何根基")
