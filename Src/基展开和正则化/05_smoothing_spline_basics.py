"""
05_smoothing_spline_basics.py
光滑样条基础: λ 的角色

光滑样条最小化:
    RSS + λ ∫[f''(x)]² dx

关键性质:
- 解是节点位于所有唯一 x_i 处的自然三次样条
- λ = 0   → 插值 (过拟合, df = N)
- λ → ∞  → 线性回归 (欠拟合, df = 2)
- λ 控制了偏差-方差权衡

ESL §5.4, 图 5.6 的精神
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import os

save_dir = os.path.join(os.path.dirname(__file__), "pic")
os.makedirs(save_dir, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def smoothing_spline(x, y, lam, x_eval=None):
    """
    用 scipy 的 UnivariateSpline 拟合光滑样条。
    
    scipy 使用平滑参数 s (而非 λ):
        s = Σ w_i (y_i - f(x_i))^2  (目标残差平方和)
    
    关系: λ ∝ 1/s (s 大 → λ 小 → 更灵活)
    
    为了直观, 我们使用 s 参数:
        s = len(y) * σ²_hat  → 大致相当于 GCV 选择的平滑量
    
    本函数封装的 lam 参数映射为:
        s = len(y) * lam  (lam 是"允许的残差放大倍数")
    """
    if x_eval is None:
        x_eval = np.linspace(x.min(), x.max(), 300)
    
    # 排序
    order = np.argsort(x)
    x_sorted, y_sorted = x[order], y[order]
    
    # scipy 的 s 参数
    s = len(y) * lam if lam > 0 else 0
    
    if lam < 1e-10:
        # 插值
        spl = UnivariateSpline(x_sorted, y_sorted, s=0, k=3)
    else:
        spl = UnivariateSpline(x_sorted, y_sorted, s=s, k=3)
    
    return x_eval, spl(x_eval), spl


def effective_df(spl):
    """近似有效自由度 (仅适用于 UnivariateSpline 的 s>0 情形)。"""
    # scipy 不直接返回 df, 用平滑样条的近似: df ≈ trace(S_λ)
    # 这里用 get_residual() 推算
    try:
        res = spl.get_residual()
        # 粗糙近似
        return None
    except:
        return None


if __name__ == "__main__":
    # ============================================================
    # 生成数据
    # ============================================================
    rng = np.random.default_rng(42)
    n = 40
    x = np.sort(rng.uniform(0, 10, n))
    
    # 复杂的真实函数: 低频 + 中频 + 局部波动
    f_true = (np.sin(0.8 * x) 
              + 0.4 * np.cos(2.5 * x) 
              + 0.6 * np.exp(-0.5 * ((x - 5) / 1.2)**2))
    sigma = 0.3
    y = f_true + rng.normal(0, sigma, n)
    
    x_dense = np.linspace(-0.3, 10.3, 400)
    f_dense = (np.sin(0.8 * x_dense) 
               + 0.4 * np.cos(2.5 * x_dense) 
               + 0.6 * np.exp(-0.5 * ((x_dense - 5) / 1.2)**2))
    
    # ============================================================
    # 图 1: 不同 λ 下的光滑样条 (核心!)
    # ============================================================
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    lambda_values = [
        (1e-8,  '插值'),
        (0.01,  '很灵活'),
        (0.05,  '较灵活'),
        (0.2,   '适中'),
        (1.0,   '光滑'),
        (10.0,  '很光滑'),
    ]
    
    for idx, (lam, label) in enumerate(lambda_values):
        ax = axes[idx // 3][idx % 3]
        
        x_s, y_s, spl = smoothing_spline(x, y, lam, x_dense)
        
        ax.plot(x_dense, f_dense, 'k-', lw=1.5, alpha=0.2, label='真实 $f(x)$')
        ax.scatter(x, y, c='gray', s=15, alpha=0.5, zorder=5)
        ax.plot(x_s, y_s, 'r-', lw=2, label=f'光滑样条')
        
        # 计算训练 MSE
        y_pred_train = spl(x)
        mse_train = np.mean((y - y_pred_train)**2)
        
        # 显示有效自由度 (scipy 的 UnivariateSpline 不易直接获取 df,
        # 但可以估算: df ≈ 在唯一 x_i 处的自然样条, 复杂度由 s 隐含)
        
        ax.set_title(f'$\\lambda$ = {lam}  ({label})\n训练 MSE = {mse_train:.4f}',
                     fontsize=11)
        ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
        # ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
    
    plt.suptitle('光滑样条: $\\lambda$ 控制光滑程度\n'
                 '$\\lambda$ 小 → 插值 (过拟合); $\\lambda$ 大 → 线性 (欠拟合)',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    save_path = os.path.join(save_dir, "05_smoothing_spline_lambda.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"图 1 已保存: {save_path}")
    
    # ============================================================
    # 图 2: 一条曲线上叠加多个 λ 的对比
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # 左: 多条曲线在同一图上
    ax = axes[0]
    ax.plot(x_dense, f_dense, 'k-', lw=2.5, alpha=0.3, label='真实 $f(x)$')
    ax.scatter(x, y, c='gray', s=15, alpha=0.5, zorder=5)
    
    colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, 5))
    lam_show = [1e-8, 0.02, 0.1, 0.5, 5.0]
    labels_show = ['插值 ($\\lambda \\approx 0$)', 
                   '$\\lambda=0.02$', '$\\lambda=0.1$', 
                   '$\\lambda=0.5$', '线性 ($\\lambda=5$)']
    lw_show = [0.8, 1.2, 1.8, 1.8, 2.5]
    
    for lam, label, c, lw in zip(lam_show, labels_show, colors, lw_show):
        x_s, y_s, _ = smoothing_spline(x, y, lam, x_dense)
        ax.plot(x_s, y_s, color=c, lw=lw, alpha=0.85, label=label)
    
    ax.set_xlabel('$x$', fontsize=12)
    ax.set_ylabel('$y$', fontsize=12)
    ax.set_title('光滑样条: $\\lambda$ 连续调节光滑度', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    
    # 右: 残差平方和 vs 曲率惩罚的 trade-off
    ax = axes[1]
    
    lam_range = np.logspace(-4, 2, 50)
    rss_vals = []
    penalty_vals = []
    
    # 排序 x
    order = np.argsort(x)
    x_sorted, y_sorted = x[order], y[order]
    
    for lam in lam_range:
        s_val = len(y) * lam if lam > 0 else 0
        try:
            spl = UnivariateSpline(x_sorted, y_sorted, s=max(s_val, 1e-10), k=3)
            y_fit = spl(x_sorted)
            rss = np.sum((y_sorted - y_fit)**2)
            
            # 数值近似曲率惩罚 ∫[f'']² dx
            x_fine = np.linspace(x_sorted.min(), x_sorted.max(), 500)
            f_fine = spl(x_fine)
            f_dd = np.gradient(np.gradient(f_fine, x_fine), x_fine)
            penalty = np.trapz(f_dd**2, x_fine)
            
            rss_vals.append(rss)
            penalty_vals.append(penalty)
        except:
            rss_vals.append(np.nan)
            penalty_vals.append(np.nan)
    
    rss_vals = np.array(rss_vals)
    penalty_vals = np.array(penalty_vals)
    
    # 归一化到 [0,1] 以便比较
    valid = ~np.isnan(rss_vals)
    rss_norm = rss_vals[valid] / np.max(rss_vals[valid])
    penalty_norm = penalty_vals[valid] / np.max(penalty_vals[valid])
    lam_valid = lam_range[valid]
    
    ax.plot(lam_valid, rss_norm, 'b-', lw=2, label='残差平方和 RSS (归一化)')
    ax.plot(lam_valid, penalty_norm, 'r-', lw=2, label='曲率惩罚 $\\int(f'')^2$ (归一化)')
    ax.plot(lam_valid, rss_norm + penalty_norm * np.max(rss_vals[valid]) / np.max(penalty_vals[valid]),
            'purple', lw=2, ls='--', alpha=0.5, label='RSS + $\\lambda \\times$ 惩罚')
    
    ax.set_xscale('log')
    ax.set_xlabel('$\\lambda$ (对数尺度)', fontsize=12)
    ax.set_ylabel('归一化值', fontsize=12)
    ax.set_title('偏差-方差权衡的量化\n$\\lambda \\uparrow$: RSS $\\uparrow$, 惩罚 $\\downarrow$', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    save_path2 = os.path.join(save_dir, "05_smoothing_spline_tradeoff.png")
    fig.savefig(save_path2, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"图 2 已保存: {save_path2}")
    
    print("\n=== 核心总结 ===")
    print("• 光滑样条 = 所有唯一 x_i 处放节点 + λ 惩罚曲率")
    print("• λ = 0: 插值 (df = N), 零偏差/最大方差")
    print("• λ → ∞: 线性回归 (df = 2), 最大偏差/零方差")
    print("• λ 的选择: 通常用 GCV 或 CV (§5.4.1)")
