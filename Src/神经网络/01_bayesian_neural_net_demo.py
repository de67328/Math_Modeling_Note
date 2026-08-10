# -*- coding: utf-8 -*-
"""
贝叶斯神经网络 (BNN) 实践演示：手写 HMC 采样权重后验 + 预测分布可视化

对应笔记 §11.9（贝叶斯神经网络与 NIPS 2003 挑战）的 BNN 四步流程：
    建模 → 推断 → 采样（HMC）→ 预测（后验平均）

模型（单隐层回归网络，隐藏层 tanh 激活）：
    a_m(x)  = alpha0_m + alpha_m * x,        m = 1..M
    h_m(x)  = tanh(a_m(x))
    f(x;θ)  = beta0 + Σ_m beta_m * h_m(x)
    似然    Y | x, θ ~ N(f(x;θ), σ²)         (σ 固定)

先验（弥散高斯先验, 见笔记）：
    θ ~ N(0, σ0² I),  σ0 取大 → 弱信息、让数据主导

采样（Hybrid Monte Carlo / HMC, 见笔记）：
    势能 U(θ) = -log π(θ|data)，动能 K(p) = ½ pᵀp
    leapfrog 积分 + Metropolis 接受/拒绝 → 得到后验样本 θ_1..θ_L

可视化：
    1. 后验预测分布：均值曲线 + 90% 不确定性带
    2. 多个采样网络的预测曲线（模型平均的直观展示）
    3. 权重 trace 与后验直方图（验证采样）
    4. 与普通 NN（点估计）对比：BNN 给出不确定性
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- 中文字体 ----------
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).parent / "pic"
OUT.mkdir(exist_ok=True)

rng = np.random.default_rng(0)

# ============================================================
# 0. 数据生成：Y = sin(πX) + 噪声（输出标准化，使 σ² 尺度正常）
# ============================================================
N = 60
X = rng.uniform(-1, 1, size=N)
true_f = np.sin(np.pi * X)           # 单周期，tanh 网络易拟合
sigma = 0.2                          # 原始噪声标准差
Y_raw = true_f + sigma * rng.standard_normal(N)

# 标准化输出：均值 0、标准差 1（残差尺度 O(1)，数值稳定）
Y_mean, Y_std = Y_raw.mean(), Y_raw.std()
Y = (Y_raw - Y_mean) / Y_std
sigma = 1.0                          # 标准化后的似然噪声

X_grid = np.linspace(-1, 1, 200)     # 预测网格
true_grid = np.sin(np.pi * X_grid)

# ============================================================
# 1. 网络结构与梯度（解析反向传播）
# ============================================================
M = 6                                # 隐藏单元数

def unpack(theta):
    """theta: [alpha0(M), alpha(M), beta0(1), beta(M)]"""
    alpha0 = theta[:M]
    alpha = theta[M:2 * M]
    beta0 = theta[2 * M]
    beta = theta[2 * M + 1:]
    return alpha0, alpha, beta0, beta

def forward(theta, x):
    """前向传播 f(x;θ)，返回 (f, h, a)"""
    alpha0, alpha, beta0, beta = unpack(theta)
    a = alpha0[:, None] + alpha[:, None] * x[None, :]     # (M, N)
    h = np.tanh(a)                                        # (M, N)
    f = beta0 + beta @ h                                  # (N,)
    return f, h, a

def log_prior(theta, sigma0=1.0):
    """先验：θ ~ N(0, σ0² I)（σ0=1 相对权重尺度 0.1 仍属弱信息）"""
    return -0.5 * np.sum(theta ** 2) / sigma0 ** 2

def log_lik(theta):
    """高斯似然：Σ_i log N(y_i | f(x_i;θ), σ²)"""
    f, _, _ = forward(theta, X)
    return -0.5 * np.sum((Y - f) ** 2) / sigma ** 2

def log_target(theta):
    """未归一化后验的对数"""
    return log_lik(theta) + log_prior(theta)

def grad_log_target(theta, sigma0=1.0):
    """∇θ log π(θ|data)（解析反向传播）"""
    f, h, a = forward(theta, X)
    alpha0, alpha, beta0, beta = unpack(theta)

    # 损失残差 r_i = (y_i - f_i)/σ²
    r = (Y - f) / sigma ** 2

    # ∂f/∂h = beta（输出层）
    # ∂h/∂a = 1 - tanh²(a)；∂a/∂alpha0 = 1, ∂a/∂alpha = x
    g = np.tanh(a)  # 已用；d_tanh = 1 - g²
    dtanh = 1 - g ** 2                                    # (M, N)

    # ∂/∂beta_m = Σ_i r_i * h_mi
    grad_beta = r @ h.T                                   # (M,)
    # ∂/∂beta0 = Σ_i r_i
    grad_beta0 = np.sum(r)
    # ∂/∂alpha_m = Σ_i r_i * beta_m * dtanh_mi * x_i
    grad_alpha = (r[None, :] * beta[:, None] * dtanh * X[None, :]).sum(axis=1)
    # ∂/∂alpha0_m = Σ_i r_i * beta_m * dtanh_mi
    grad_alpha0 = (r[None, :] * beta[:, None] * dtanh).sum(axis=1)

    grad_theta = np.concatenate([grad_alpha0, grad_alpha,
                                 [grad_beta0], grad_beta])
    grad_theta -= theta / sigma0 ** 2                     # 先验梯度
    return grad_theta

# ============================================================
# 2. HMC 采样器（leapfrog + Metropolis）
# ============================================================
def leapfrog(theta, p, eps, L):
    """蛙跳积分：步长 eps、L 步（对应笔记 eq:11_20a）"""
    theta = theta.copy()
    p = p.copy()
    # 半步动量
    p = p - 0.5 * eps * grad_log_target(theta)
    for _ in range(L):
        theta = theta + eps * p                      # 全步位置
        if _ < L - 1:
            p = p - eps * grad_log_target(theta)     # 全步动量（除最后）
    p = p - 0.5 * eps * grad_log_target(theta)       # 半步动量收尾
    return theta, p

def hmc_sampler(theta0, n_samples, eps=0.005, L=30, burnin=300):
    """HMC：从后验采样 n_samples 个 θ（含 burn-in 预热）"""
    theta = np.array(theta0, dtype=float)
    samples = []
    accepted = 0
    n_total = burnin + n_samples
    for it in range(n_total):
        # 重新采样动量 p ~ N(0, I)
        p = rng.standard_normal(theta.size)
        # 势能
        H0 = -log_target(theta) + 0.5 * (p @ p)
        # leapfrog
        theta_star, p_star = leapfrog(theta, p, eps, L)
        H_star = -log_target(theta_star) + 0.5 * (p_star @ p_star)
        # 数值安全：发散（NaN/inf）则拒绝
        if not (np.isfinite(H_star) and np.isfinite(theta_star).all()):
            continue
        # Metropolis 接受/拒绝
        if rng.uniform() < np.exp(min(0, H0 - H_star)):
            theta = theta_star
            accepted += 1
        if it >= burnin:
            samples.append(theta.copy())
    return np.array(samples), accepted / n_total

# ============================================================
# 3. 运行：MAP 预热 → HMC 采样 → 预测
# ============================================================
# 起始值：小随机权重（对应笔记 §11.5.1）
n_params = 2 * M + 1 + M
theta0 = 0.3 * rng.standard_normal(n_params)

# 普通 NN 点估计（MAP，即带先验的梯度上升；同时作为 HMC 预热起点）
def fit_map(theta_init, steps=2000, lr=0.02, n_restarts=8):
    """MAP 点估计：多随机起点 + 梯度裁剪，取 log 后验最高的（应对多重极小）"""
    best_th, best_l = None, -np.inf
    for r in range(n_restarts):
        th = (theta_init.copy() if r == 0 else 0.3 * rng.standard_normal(th.size))
        for _ in range(steps):
            g = grad_log_target(th)
            if not np.isfinite(g).all() or not np.isfinite(th).all():
                th = 0.3 * rng.standard_normal(th.size)   # 发散则重新随机
                continue
            g_norm = np.linalg.norm(g)
            if g_norm > 5.0:
                g = g * (5.0 / g_norm)
            th = th + lr * g
        l = log_target(th)
        if l > best_l:
            best_l, best_th = l, th.copy()
    return best_th

theta_map = fit_map(theta0)
print("MAP 预热完成，从后验众数附近开始 HMC")

print("HMC 采样中...")
samples, acc_rate = hmc_sampler(theta_map, n_samples=400, eps=0.005, L=30, burnin=300)
print(f"接受率 = {acc_rate:.2%}, 有效样本 = {len(samples)}")

# 后验预测：对每个 θℓ 计算 f(x_grid; θℓ)
f_samps = np.array([forward(th, X_grid)[0] for th in samples])   # (L, G)
f_mean_std = f_samps.mean(axis=0)
f_lo_std, f_hi_std = np.percentile(f_samps, [5, 95], axis=0)    # 90% 带
f_map_std = forward(theta_map, X_grid)[0]

# 反标准化回原始尺度（仅用于绘图）
f_mean = f_mean_std * Y_std + Y_mean
f_lo = f_lo_std * Y_std + Y_mean
f_hi = f_hi_std * Y_std + Y_mean
f_map = f_map_std * Y_std + Y_mean
true_grid = np.sin(np.pi * X_grid)
f_samps_orig = f_samps * Y_std + Y_mean

# ============================================================
# 4. 可视化
# ============================================================
# ---- 图 1：后验预测分布（均值 + 90% 带）----
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.scatter(X, Y_raw, s=14, color="gray", alpha=0.6, label="训练数据")
ax.plot(X_grid, true_grid, "k--", lw=1.2, label="真值 sin(2πX)")
ax.plot(X_grid, f_mean, "b-", lw=2, label="BNN 后验均值")
ax.fill_between(X_grid, f_lo, f_hi, color="b", alpha=0.2, label="90% 预测带")
ax.plot(X_grid, f_map, "r-", lw=1.5, alpha=0.8, label="MAP 点估计")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("BNN 后验预测分布（HMC 采样后验平均）")
ax.legend(loc="upper right", fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "fig1_predictive.png", dpi=150)

# ---- 图 2：多个采样网络的预测曲线（模型平均）----
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.scatter(X, Y_raw, s=14, color="gray", alpha=0.5, label="训练数据")
idx = np.linspace(0, len(f_samps_orig) - 1, 20).astype(int)
for i in idx:
    ax.plot(X_grid, f_samps_orig[i], color="b", alpha=0.25, lw=0.8)
ax.plot(X_grid, f_mean, "b-", lw=2.5, label="后验平均")
ax.plot(X_grid, true_grid, "k--", lw=1.2, label="真值 sin(πx)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title(r"20 个采样网络的预测曲线：模型平均（$\frac{1}{L}\sum_\ell f(x;\theta_\ell)$）")
ax.legend(loc="upper right", fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "fig2_ensemble.png", dpi=150)

# ---- 图 3：权重 trace 与后验直方图 ----
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
# trace（取前 3 个参数）
for k in range(3):
    axes[0].plot(samples[:, k], lw=0.7, label=f"θ[{k}]")
axes[0].set_xlabel("采样迭代")
axes[0].set_ylabel("参数值")
axes[0].set_title("HMC 权重 trace")
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3)
# 后验直方图（β0 与第一个 β_m）
axes[1].hist(samples[:, 2 * M], bins=30, alpha=0.6, label=r"$\beta_0$ 后验")
axes[1].hist(samples[:, 2 * M + 1], bins=30, alpha=0.6, label=r"$\beta_1$ 后验")
axes[1].axvline(theta_map[2 * M], color="r", ls="--", lw=1, label=r"MAP $\beta_0$")
axes[1].set_xlabel("参数值")
axes[1].set_ylabel("频数")
axes[1].set_title("权重后验分布（HMC 采样）")
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "fig3_posterior.png", dpi=150)

print("图片已保存到:", OUT)
