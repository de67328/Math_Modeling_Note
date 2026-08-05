# -*- coding: utf-8 -*-
"""
MCMC 实践演示：Metropolis-Hastings 从非标准后验抽样 + 收敛诊断可视化

对应笔记 §8.6 MCMC（含小节「MCMC 的实践：有限步收敛与诊断」）。

模型（非共轭，后验无解析形式，故用 MCMC）：
    观测 y ~ N(theta, 1)
    先验 theta ~ Laplace(0, b)
    后验 ∝ exp(-(y-theta)^2/2) * exp(-|theta|/b)     # 无闭合形式

诊断可视化：
    1. 多链 trace plot（burn-in 分界）
    2. 后验直方图 vs 网格"真值"（验证抽样正确性）
    3. 自相关图 + 有效样本量 ESS
    4. Gelman-Rubin R-hat 随迭代次数的收敛曲线
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

# ============ 0. 模型设定 ============
y_obs = 3.0       # 观测值
b = 2.0           # Laplace 先验尺度
sigma_prop = 1.0  # 随机游走提议分布的标准差（步长）

def log_prior(theta):
    """theta ~ Laplace(0, b)：log p(theta) = -|theta|/b + const"""
    return -np.abs(theta) / b

def log_lik(theta):
    """单个观测 y ~ N(theta, 1)：log p(y|theta) = -(y-theta)^2/2 + const"""
    return -0.5 * (y_obs - theta) ** 2

def log_target(theta):
    """未归一化后验的对数：log [p(y|theta) * p(theta)]"""
    return log_lik(theta) + log_prior(theta)

# ---- 网格法计算"精确"后验（数值归一化，作真值对照）----
grid = np.linspace(-10, 14, 2001)
log_g = log_target(grid)
density = np.exp(log_g - log_g.max())
post_true = density / np.trapezoid(density, grid)

dx = grid[1] - grid[0]
cdf_true = np.cumsum(post_true * dx)
mean_true = np.trapezoid(grid * post_true, grid)

def grid_quantile(q):
    return grid[np.searchsorted(cdf_true, q)]

# ============ 1. Metropolis-Hastings ============
rng = np.random.default_rng(42)

def metropolis_hastings(theta0, n_iter, sigma=sigma_prop):
    """随机游走 Metropolis-Hastings，返回样本序列与接受率"""
    thetas = np.empty(n_iter)
    theta = theta0
    n_acc = 0
    for t in range(n_iter):
        prop = theta + rng.normal(0, sigma)
        log_acc = log_target(prop) - log_target(theta)
        if np.log(rng.uniform()) < log_acc:
            theta = prop
            n_acc += 1
        thetas[t] = theta
    return thetas, n_acc / n_iter

# ---- 4 条链、不同起点（体现"忘记起点"）----
starts = [-8.0, -2.0, 4.0, 9.0]
n_iter = 12000
burnin = 2000

chains, acc_rates = [], []
for s in starts:
    ch, ar = metropolis_hastings(s, n_iter)
    chains.append(ch)
    acc_rates.append(ar)
chains = np.array(chains)  # shape (n_chains, n_iter)
print("各链接受率：", np.round(acc_rates, 3))

# ============ 2. 工具函数：诊断量 ============
def autocorr(x, max_lag=100):
    """自相关函数 rho_k（滞后 k）"""
    x = x - x.mean()
    n = len(x)
    s2 = np.sum(x ** 2)
    return np.array([np.sum(x[:n-k] * x[k:]) / s2 for k in range(max_lag + 1)])

def effective_sample_size(x):
    """有效样本量 ESS = n / tau，tau = 1 + 2*sum(rho_k)，截断到首个负自相关"""
    ac = autocorr(x, max_lag=len(x) - 1)
    rho = ac[1:]
    neg = np.where(rho < 0)[0]
    cut = neg[0] if len(neg) else len(rho)
    tau = 1.0 + 2.0 * np.sum(rho[:cut])
    return len(x) / tau, tau

def rhat_by_iteration(chains, t_start=2000, step=200):
    """Gelman-Rubin R-hat 随前缀长度 t 的变化"""
    n_ch, n_tot = chains.shape
    ts = np.arange(t_start, n_tot + 1, step)
    rhats = []
    for t in ts:
        x = chains[:, :t]                     # (n_ch, t)
        m = x.mean(axis=1)                    # 每条链的均值
        W = x.var(axis=1, ddof=1).mean()      # 链内方差
        B = t / (n_ch - 1) * np.sum((m - x.mean()) ** 2)  # 链间方差
        var_est = (t - 1) / t * W + B / t
        rhats.append(np.sqrt(var_est / W))
    return ts, np.array(rhats)

# ============ 3. 计算诊断量 ============
post = chains[:, burnin:].ravel()             # 丢弃 burn-in 后的合并样本
ess_val, tau_val = effective_sample_size(chains[0, burnin:])

mcmc_mean = post.mean()
mcmc_lo, mcmc_hi = np.quantile(post, [0.025, 0.975])
ts_r, rhats = rhat_by_iteration(chains)

print(f"网格真值  后验均值 = {mean_true:.3f}   95% CI = [{grid_quantile(0.025):.3f}, {grid_quantile(0.975):.3f}]")
print(f"MCMC 估计 后验均值 = {mcmc_mean:.3f}   95% CI = [{mcmc_lo:.3f}, {mcmc_hi:.3f}]")
print(f"最终 R-hat = {rhats[-1]:.3f}（< 1.1 认为收敛）")
print(f"自相关时间 tau = {tau_val:.1f}，有效样本量 ESS = {ess_val:.0f}（样本数 {len(post)}）")

# ============ 4. 图 1：多链 trace plot ============
fig, ax = plt.subplots(figsize=(10, 5))
colors = plt.cm.tab10(np.linspace(0, 1, len(chains)))[:len(chains)]
for j, ch in enumerate(chains):
    ax.plot(ch, lw=0.7, color=colors[j], label=f"链 {j+1}（起点 {starts[j]:.0f}）")
ax.axvline(burnin, color="red", ls="--", lw=1.2, label=f"burn-in = {burnin}")
ax.axhline(mean_true, color="black", ls=":", lw=1.2, label=f"后验真值均值 {mean_true:.2f}")
ax.set_xlabel("迭代次数 t")
ax.set_ylabel(r"$\theta^{(t)}$")
ax.set_title("多链 trace plot：burn-in 前段受起点影响，之后各链围绕后验中心振荡")
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "fig1_trace.png", dpi=150)
plt.close(fig)

# ============ 5. 图 2：后验验证（直方图 vs 真值）============
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(post, bins=80, density=True, alpha=0.55, color="steelblue",
        label="MCMC 样本（丢弃 burn-in）")
ax.plot(grid, post_true, "r-", lw=2, label="网格法真值后验")
ax.axvline(mcmc_mean, color="steelblue", ls="--", lw=1.2, label=f"MCMC 均值 {mcmc_mean:.2f}")
ax.axvline(mean_true, color="red", ls="--", lw=1.2, label=f"真值均值 {mean_true:.2f}")
ax.axvspan(mcmc_lo, mcmc_hi, color="steelblue", alpha=0.12,
           label=f"95% 可信区间 [{mcmc_lo:.2f}, {mcmc_hi:.2f}]")
ax.set_xlabel(r"$\theta$")
ax.set_ylabel("密度")
ax.set_title("后验抽样验证：MCMC 直方图与数值真值几乎重合")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "fig2_posterior.png", dpi=150)
plt.close(fig)

# ============ 6. 图 3：自相关图 + ESS ============
ac = autocorr(chains[0, burnin:], max_lag=100)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(len(ac)), ac, "o-", ms=4, color="darkorange", label="自相关函数")
ax.axhline(0, color="gray", lw=0.8)
ax.axhline(np.exp(-1), color="gray", ls="--", lw=0.8, label=r"$e^{-1}$ 阈值")
ax.set_xlabel("滞后 k")
ax.set_ylabel(r"自相关 $\rho_k$")
ax.set_title(f"自相关与有效样本量：$\\tau={tau_val:.1f}$，ESS $= {ess_val:.0f}$（样本数 {len(post)}）")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "fig3_autocorr_ess.png", dpi=150)
plt.close(fig)

# ============ 7. 图 4：Gelman-Rubin R-hat 收敛 ============
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(ts_r, rhats, "o-", ms=4, color="seagreen", label=r"$\hat{R}(t)$")
ax.axhline(1.0, color="black", ls="--", lw=1.0, label=r"$\hat{R}=1$（理想）")
ax.axhline(1.1, color="red", ls="--", lw=1.2, label=r"$\hat{R}=1.1$（常用阈值）")
ax.set_xlabel("链长 t（丢弃 burn-in 后）")
ax.set_ylabel(r"Gelman–Rubin $\hat{R}$")
ax.set_title(rf"收敛诊断：$\hat{{R}}$ 随链长下降并趋近 1（最终 $\hat{{R}}={rhats[-1]:.3f}$）")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "fig4_rhat.png", dpi=150)
plt.close(fig)

print("\n图片已保存到：", OUT)
print("fig1_trace.png / fig2_posterior.png / fig3_autocorr_ess.png / fig4_rhat.png")
