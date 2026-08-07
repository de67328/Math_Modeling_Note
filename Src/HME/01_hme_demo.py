# -*- coding: utf-8 -*-
"""
HME 演示：两层专家分层混合（Hierarchical Mixtures of Experts），EM 训练

对应笔记 §9.5。

模型：
  顶层门控 g_j(x)（softmax，K=2）→ 第二层门控 g_{l|j}(x)（softmax，K=2）
  → 4 个专家，每个专家是线性回归 Y = beta^T x + eps（高斯）
  总概率：
      Pr(y|x, Psi) = sum_j g_j(x) * sum_l g_{l|j}(x) * N(y; beta_{jl}^T x, sigma^2)

训练：EM（软分配，不是分开训练专家）
  E 步：给定当前参数，计算每个观测到各专家的后验概率（软分配）
  M 步：专家参数 = 加权最小二乘（权重=后验概率）；
        门控参数 = softmax 回归（用后验概率作目标，梯度上升）；
        sigma = 加权残差估计

输出图（pic/）：
  fig1_hme_data.png      训练数据（按真实分区着色）
  fig2_hme_prediction.png HME 预测面（网格）
  fig3_hme_assignment.png 软分配（每个观测按最大后验专家着色）
  fig4_hme_loglik.png     对数似然收敛曲线
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

rng = np.random.default_rng(1)

# ============ 1. 生成数据：4 个真实分区，各一个线性模型 ============
N = 400
x1 = rng.uniform(-2, 2, N)
x2 = rng.uniform(-2, 2, N)
X = np.column_stack([np.ones(N), x1, x2])      # 含截距，shape (N,3)
true_idx_j = (x1 >= 0).astype(int)             # 顶部分支：x1 的符号
true_idx_l = (x2 >= 0).astype(int)             # 第二层分支：x2 的符号

# 四个专家的真实系数（每个区域一个线性模型）
true_beta = {
    (0, 0): np.array([0.0, 1.0, 1.0]),     # y = x1 + x2
    (0, 1): np.array([1.0, 1.0, -1.0]),    # y = 1 + x1 - x2
    (1, 0): np.array([-1.0, -1.0, 1.0]),   # y = -1 - x1 + x2
    (1, 1): np.array([0.0, -1.0, -1.0]),   # y = -x1 - x2
}
sigma_true = 0.2

y = np.zeros(N)
for j in (0, 1):
    for l in (0, 1):
        mask = (true_idx_j == j) & (true_idx_l == l)
        y[mask] = X[mask] @ true_beta[(j, l)] + rng.normal(0, sigma_true, mask.sum())

# ============ 2. HME 前向（对数域，避免下溢） ============
K = 2

def gate_logit(w, x):
    """二分支门控：分支 1 的 logit = w@x，分支 2 的 logit = 0"""
    return w @ x

def log_gate_probs(w, X):
    """返回 (log g_1, log g_2) —— softmax 对数概率"""
    a = X @ w                       # 分支 1 的 logit
    m = np.maximum(a, 0)
    log_g1 = a - np.logaddexp(a, 0)
    log_g2 = 0 - np.logaddexp(a, 0)  # 分支 2 logit = 0
    return log_g1, log_g2

def log_likelihood(X, y, w_top, w_g1, w_g2, betas, sig2):
    """log Pr(y|X, Psi)：log-sum-exp 稳定计算"""
    n = len(y)
    lg1, lg2 = log_gate_probs(w_top, X)            # 顶层
    lg11, lg12 = log_gate_probs(w_g1, X)           # 分支 1 内
    lg21, lg22 = log_gate_probs(w_g2, X)           # 分支 2 内
    ll = np.zeros(n)
    for j in (0, 1):
        for l in (0, 1):
            # 观测到专家 (j,l) 的路径对数概率
            lg_path = (lg1 if j == 0 else lg2) + (lg11 if (j, l) == (0, 0)
                      else lg12 if (j, l) == (0, 1)
                      else lg21 if (j, l) == (1, 0) else lg22)
            mu = X @ betas[(j, l)]
            lgauss = -0.5 * np.log(2 * np.pi * sig2) - (y - mu) ** 2 / (2 * sig2)
            ll = np.logaddexp(ll, lg_path + lgauss)
    return ll.sum()

# ============ 3. EM 训练 ============
def weighted_ls(X, y, w):
    """加权最小二乘：beta = (X^T W X)^{-1} X^T W y"""
    W = w[:, None] * np.ones_like(X)               # 每行权重（列广播）
    XtWX = X.T @ (w[:, None] * X)
    XtWy = X.T @ (w * y)
    return np.linalg.solve(XtWX + 1e-9 * np.eye(X.shape[1]), XtWy)

# ---- 初始化 ----
w_top = np.zeros(3)      # 顶层门控（分支 1）
w_g1 = np.zeros(3)       # 分支 1 内门控（专家 1）
w_g2 = np.zeros(3)       # 分支 2 内门控（专家 1）
beta0 = np.linalg.lstsq(X, y, rcond=None)[0]       # 全局线性回归
betas = {(j, l): beta0.copy() for j in (0, 1) for l in (0, 1)}
sig2 = np.var(y)

max_iter = 60
tol = 1e-6
ll_hist = []
for it in range(max_iter):
    # ---- E 步：软分配（后验概率 r_{i,jl}） ----
    lg1, lg2 = log_gate_probs(w_top, X)
    lg11, lg12 = log_gate_probs(w_g1, X)
    lg21, lg22 = log_gate_probs(w_g2, X)
    lg_path = {}
    for j in (0, 1):
        for l in (0, 1):
            lg_path[(j, l)] = (lg1 if j == 0 else lg2) + \
                (lg11 if (j, l) == (0, 0) else lg12 if (j, l) == (0, 1)
                 else lg21 if (j, l) == (1, 0) else lg22)
    lgauss = {}
    for key in lg_path:
        mu = X @ betas[key]
        lgauss[key] = -0.5 * np.log(2 * np.pi * sig2) - (y - mu) ** 2 / (2 * sig2)
    # 对每个观测归一化（log-sum-exp 技巧）
    log_all = np.column_stack([lg_path[k] + lgauss[k] for k in lg_path])
    logsum = np.logaddexp.reduce(log_all, axis=1)
    r = np.exp(log_all - logsum[:, None])          # shape (N, 4)，每行和为 1

    # ---- M 步：专家参数（加权最小二乘）+ sigma ----
    keys = [(0, 0), (0, 1), (1, 0), (1, 1)]
    total_w = np.zeros(N)
    for idx, k in enumerate(keys):
        w_i = r[:, idx]
        total_w += w_i
        betas[k] = weighted_ls(X, y, w_i)
    sig2 = np.sum(r * ((y[:, None] - np.column_stack([X @ betas[k] for k in keys])) ** 2)) / r.sum()

    # ---- M 步：门控参数（softmax 回归，梯度上升，几步） ----
    lr = 0.05
    for _ in range(10):
        # 顶层：目标 t_j = sum_l r_{i,(j,l)}
        t1 = r[:, 0] + r[:, 1]
        g1 = 1 / (1 + np.exp(-(X @ w_top)))
        w_top += lr * (X.T @ (t1 - g1))
        # 分支 1 内：目标 t_{1|1} = r_{i,(0,0)} / (r_{i,(0,0)}+r_{i,(0,1)})
        denom1 = np.maximum(r[:, 0] + r[:, 1], 1e-9)
        t11 = r[:, 0] / denom1
        g11 = 1 / (1 + np.exp(-(X @ w_g1)))
        w_g1 += lr * (X.T @ (t11 - g11))
        # 分支 2 内：目标 t_{1|2} = r_{i,(1,0)} / (r_{i,(1,0)}+r_{i,(1,1)})
        denom2 = np.maximum(r[:, 2] + r[:, 3], 1e-9)
        t21 = r[:, 2] / denom2
        g21 = 1 / (1 + np.exp(-(X @ w_g2)))
        w_g2 += lr * (X.T @ (t21 - g21))

    ll = log_likelihood(X, y, w_top, w_g1, w_g2, betas, sig2)
    ll_hist.append(ll)
    if it > 2 and abs(ll_hist[-1] - ll_hist[-2]) < tol * abs(ll_hist[-1]):
        break

print(f"EM 收敛于第 {len(ll_hist)} 次迭代，对数似然 = {ll_hist[-1]:.1f}")

# 预测函数（均值）
def predict_mean(Xg, w_top, w_g1, w_g2, betas):
    lg1, lg2 = log_gate_probs(w_top, Xg)
    lg11, lg12 = log_gate_probs(w_g1, Xg)
    lg21, lg22 = log_gate_probs(w_g2, Xg)
    prob = {}
    for j in (0, 1):
        for l in (0, 1):
            prob[(j, l)] = np.exp((lg1 if j == 0 else lg2) +
                                  (lg11 if (j, l) == (0, 0) else lg12 if (j, l) == (0, 1)
                                   else lg21 if (j, l) == (1, 0) else lg22))
    out = np.zeros(Xg.shape[0])
    for k in prob:
        out += prob[k] * (Xg @ betas[k])
    return out

# ============ 4. 可视化 ============
# 图 1：训练数据（按真实分区着色）
fig, ax = plt.subplots(figsize=(6, 5))
sc = ax.scatter(x1, x2, c=y, cmap="viridis", s=18, alpha=0.8)
ax.axvline(0, color="gray", ls="--", lw=1)
ax.axhline(0, color="gray", ls="--", lw=1)
ax.set_xlabel("$X_1$")
ax.set_ylabel("$X_2$")
ax.set_title("训练数据：颜色 = 响应 $y$（真实分区在 0 处）")
fig.colorbar(sc)
fig.tight_layout()
fig.savefig(OUT / "fig1_hme_data.png", dpi=150)
plt.close(fig)

# 图 2：HME 预测面（网格）
g = np.linspace(-2, 2, 100)
G1, G2 = np.meshgrid(g, g)
Xg = np.column_stack([np.ones(G1.size), G1.ravel(), G2.ravel()])
pred = predict_mean(Xg, w_top, w_g1, w_g2, betas).reshape(G1.shape)
fig, ax = plt.subplots(figsize=(6, 5))
pc = ax.pcolormesh(G1, G2, pred, shading="auto", cmap="viridis")
ax.set_xlabel("$X_1$")
ax.set_ylabel("$X_2$")
ax.set_title("HME 预测面（EM 训练后）")
fig.colorbar(pc)
fig.tight_layout()
fig.savefig(OUT / "fig2_hme_prediction.png", dpi=150)
plt.close(fig)

# 图 3：软分配（每个观测按最大后验专家着色）
assign = np.argmax(r, axis=1)
colors = np.array(["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(x1, x2, c=colors[assign], s=18, alpha=0.8)
ax.axvline(0, color="gray", ls="--", lw=1)
ax.axhline(0, color="gray", ls="--", lw=1)
ax.set_xlabel("$X_1$")
ax.set_ylabel("$X_2$")
ax.set_title("软分配：每个观测的最大后验专家\n（0=蓝 1=橙 2=绿 3=红）")
fig.tight_layout()
fig.savefig(OUT / "fig3_hme_assignment.png", dpi=150)
plt.close(fig)

# 图 4：对数似然收敛
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.plot(range(1, len(ll_hist) + 1), ll_hist, "o-", color="seagreen")
ax.set_xlabel("EM 迭代")
ax.set_ylabel("对数似然")
ax.set_title("EM 训练：对数似然单调上升并收敛")
ax.grid(True, ls="--", alpha=0.4)
fig.tight_layout()
fig.savefig(OUT / "fig4_hme_loglik.png", dpi=150)
plt.close(fig)

print("图片已保存到：", OUT)
print("fig1_hme_data / fig2_hme_prediction / fig3_hme_assignment / fig4_hme_loglik")
