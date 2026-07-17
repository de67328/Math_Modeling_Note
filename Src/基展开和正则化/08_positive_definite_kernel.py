"""
08_positive_definite_kernel.py
正定核：验证、对比与可视化

关键概念：
1. 正定性：Gram 矩阵 K_ij = K(x_i, x_j) 对所有点集半正定
2. 不同核定义不同的"相似度"
3. 高斯核的 γ (或 ν) 控制局部性

ESL §5.8
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import os

save_dir = os.path.join(os.path.dirname(__file__), "pic")
os.makedirs(save_dir, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 核函数定义
# ============================================================

def linear_kernel(X, Y=None):
    """K(x,y) = x^T y"""
    if Y is None:
        Y = X
    return X @ Y.T


def polynomial_kernel(X, Y=None, degree=3, gamma=1.0, coef0=1.0):
    """K(x,y) = (γ x^T y + c)^d"""
    if Y is None:
        Y = X
    return (gamma * X @ Y.T + coef0) ** degree


def gaussian_kernel(X, Y=None, gamma=1.0):
    """K(x,y) = exp(-γ ||x-y||^2)"""
    if Y is None:
        Y = X
    # ||x-y||^2 = ||x||^2 + ||y||^2 - 2 x^T y
    X_norm = np.sum(X**2, axis=1).reshape(-1, 1)
    Y_norm = np.sum(Y**2, axis=1).reshape(1, -1)
    dist_sq = X_norm + Y_norm - 2 * X @ Y.T
    return np.exp(-gamma * dist_sq)


def laplacian_kernel(X, Y=None, gamma=1.0):
    """K(x,y) = exp(-γ ||x-y||_1)"""
    if Y is None:
        Y = X
    K = np.zeros((X.shape[0], Y.shape[0]))
    for i in range(X.shape[0]):
        K[i, :] = np.exp(-gamma * np.sum(np.abs(X[i] - Y), axis=1))
    return K


def check_positive_definiteness(K, name, tol=1e-10):
    """验证 Gram 矩阵半正定性。"""
    eigvals = eigh(K, eigvals_only=True)
    min_eig = np.min(eigvals)
    is_psd = min_eig > -tol
    print(f"  {name}: 最小特征值 = {min_eig:.2e}, 半正定 = {is_psd}")
    return eigvals


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 8
    X = rng.uniform(-3, 3, (n, 2))
    
    # ============================================================
    # 图 1: 验证正定性 + 可视化 Gram 矩阵
    # ============================================================
    kernels = {
        '线性核': lambda: linear_kernel(X),
        '多项式核 (d=3)': lambda: polynomial_kernel(X, degree=3),
        '高斯核 (γ=0.1)': lambda: gaussian_kernel(X, gamma=0.1),
        '高斯核 (γ=1.0)': lambda: gaussian_kernel(X, gamma=1.0),
        '高斯核 (γ=10)': lambda: gaussian_kernel(X, gamma=10.0),
        'Laplacian 核': lambda: laplacian_kernel(X, gamma=1.0),
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    print("=== 正定性验证 (所有 Gram 矩阵应半正定) ===")
    
    for idx, (name, k_func) in enumerate(kernels.items()):
        ax = axes[idx // 3][idx % 3]
        K = k_func()
        eigvals = check_positive_definiteness(K, name)
        
        im = ax.imshow(K, cmap='RdYlBu_r', vmin=0, vmax=K.max())
        ax.set_title(f'{name}\n(特征值范围: [{eigvals[0]:.1e}, {eigvals[-1]:.1f}])',
                     fontsize=10)
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels([f'$x_{i+1}$' for i in range(n)], fontsize=7)
        ax.set_yticklabels([f'$x_{i+1}$' for i in range(n)], fontsize=7)
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    plt.suptitle(r'不同核函数的 Gram 矩阵 $\mathbf{K}_{ij} = K(x_i, x_j)$' + '\n'
                 r'(半正定性: 所有特征值 $\geq 0$)', fontsize=14, y=1.01)
    plt.tight_layout()
    save_path = os.path.join(save_dir, "08_kernel_gram_matrices.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"图 1 已保存: {save_path}")
    
    # ============================================================
    # 图 2: 高斯核的 γ 参数 — 局部性的控制
    # ============================================================
    x_1d = np.linspace(-4, 4, 200).reshape(-1, 1)
    x_centers = np.array([-2, 0, 2]).reshape(-1, 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    gamma_values = [0.1, 1.0, 10.0]
    for idx, gamma in enumerate(gamma_values):
        ax = axes[idx]
        K_mat = gaussian_kernel(x_1d, x_centers, gamma=gamma)
        
        for j in range(3):
            ax.plot(x_1d, K_mat[:, j], lw=2,
                    label=f'$K(x, x_c={x_centers[j,0]:.0f})$')
        
        ax.set_title(f'高斯核 $\\gamma={gamma}$\n'
                     f'($\\gamma \\uparrow$ → 越局部, 有效维数越高)', fontsize=12)
        ax.set_xlabel('$x$'); ax.set_ylabel('$K(x, x_c)$')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.set_ylim(-0.05, 1.1)
    
    plt.suptitle('高斯核的尺度参数 $\\gamma$: 控制「局部性」', fontsize=14)
    plt.tight_layout()
    save_path2 = os.path.join(save_dir, "08_gaussian_kernel_gamma.png")
    fig.savefig(save_path2, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"图 2 已保存: {save_path2}")
    
    # ============================================================
    # 图 3: 特征值衰减 — 核的有效维数
    # ============================================================
    # 在更多点上评估，看特征值谱
    n_large = 100
    X_large = rng.uniform(-3, 3, (n_large, 2))
    
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    
    for name, gamma in [('$\\gamma=0.1$ (全局)', 0.1),
                          ('$\\gamma=1.0$ (中等)', 1.0),
                          ('$\\gamma=10$ (局部)', 10.0)]:
        K_large = gaussian_kernel(X_large, gamma=gamma)
        eigvals = np.sort(eigh(K_large, eigvals_only=True))[::-1]
        eigvals_pos = eigvals[eigvals > 1e-10]
        # 归一化
        eigvals_norm = eigvals_pos / eigvals_pos[0]
        ax.semilogy(range(1, len(eigvals_norm)+1), eigvals_norm,
                    'o-', lw=1.5, markersize=3, label=name)
    
    # 95% 能量线
    ax.axhline(0.05, color='gray', ls='--', lw=0.8, alpha=0.5)
    ax.text(1, 0.06, '5% 能量线', fontsize=8, color='gray')
    
    ax.set_xlabel('特征值序号 (降序)', fontsize=12)
    ax.set_ylabel('归一化特征值 (对数尺度)', fontsize=12)
    ax.set_title('高斯核的特征值谱\n'
                 '($\\gamma$ 越大 → 衰减越慢 → 有效维数越高 → 函数空间越大)',
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    save_path3 = os.path.join(save_dir, "08_kernel_eigenvalue_decay.png")
    fig.savefig(save_path3, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"图 3 已保存: {save_path3}")
    
    print("\n=== 核心总结 ===")
    print("• 正定性: Gram 矩阵的半正定性 = 核作为内积的几何自洽性")
    print("• γ 控制局部性: γ 小 → 核覆盖面广 → 平滑函数 (低有效维度)")
    print("  γ 大 → 核极局部 → 崎岖函数 (高有效维度)")
    print("• 特征值衰减速度 = 核空间的「有效大小」")
    print("• RKHS 范数天然偏好特征值大的方向 (低频) → 正则化生效的机理")
