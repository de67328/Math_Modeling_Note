# -*- coding: utf-8 -*-
"""
Backfitting 演示：拟合加性模型

对应笔记 §9.1.1（Algorithm 9.1）。

模型（加性）：
    Y = alpha + f1(X1) + f2(X2) + eps
    f1(x) = sin(2*pi*x)           非线性
    f2(x) = 4*(x-0.5)^2 - 0.5     非线性（二次，非对称）

平滑算子：三次平滑样条（scipy.interpolate.UnivariateSpline）
Backfitting：交替一元光滑——固定其他函数，对部分残差
    r_i = y_i - alpha - sum_{k != j} f_k(x_ik)
    做平滑，更新 f_j，并重中心化（减均值，保证 sum f_j = 0）。

输出图：
    fig1_backfitting_fit.png   真值 vs 估计（每个变量）
    fig2_convergence.png       收敛曲线（各函数最大变化量随迭代）
    fig3_additive_check.png    加性检查：部分残差 vs 拟合
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from pathlib import Path

# ---------- 中文字体 ----------
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).parent / "pic"
OUT.mkdir(exist_ok=True)

rng = np.random.default_rng(7)

# ============ 1. 生成加性数据 ============
N = 300
x1 = rng.uniform(0, 1, N)
x2 = rng.uniform(0, 1, N)

alpha_true = 0.5
def f1_true(x): return np.sin(2 * np.pi * x)
def f2_true(x): return 4.0 * (x - 0.5) ** 2 - 0.5

y = alpha_true + f1_true(x1) + f2_true(x2) + rng.normal(0, 0.15, N)

X = np.column_stack([x1, x2])
f_true = [f1_true, f2_true]
names = ["$f_1(X_1)$", "$f_2(X_2)$"]

# ============ 2. Backfitting ============
smoothing = 0.05 * N   # UnivariateSpline 的光滑参数（越大越光滑）

def smooth(x, r, s=smoothing):
    """三次平滑样条：对 (x, r) 拟合（UnivariateSpline 要求 x 递增，先排序），
    返回在原始 x 处的光滑值"""
    order = np.argsort(x)
    xs, rs = x[order], r[order]
    spl = UnivariateSpline(xs, rs, k=3, s=s)
    return spl(x)

alpha = np.mean(y)
f = [np.zeros(N), np.zeros(N)]     # 各函数在训练点的估计值

max_iter = 50
tol = 1e-5
history = []                        # 每次迭代各函数的最大变化量

for it in range(max_iter):
    change = 0.0
    for j in range(2):
        r = y - alpha - sum(f[k] for k in range(2) if k != j)   # 部分残差
        f_new = smooth(X[:, j], r)
        f_new = f_new - np.mean(f_new)                          # 重中心化
        change = max(change, np.max(np.abs(f_new - f[j])))
        f[j] = f_new
    history.append(change)
    if change < tol:
        break

n_iter = len(history)
print(f"收敛于第 {n_iter} 次迭代，最终变化量 = {history[-1]:.2e}")

# ============ 3. 图 1：真值 vs 估计 ============
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
xs = np.linspace(0, 1, 300)
for j, ax in enumerate(axes):
    ax.scatter(X[:, j], y - alpha - sum(f[k] for k in range(2) if k != j),
               s=12, alpha=0.35, color="lightgray", label="部分残差")
    # 真值居中：可识别性约束使常数被 α 吸收，这里展示居中版本与估计可比
    ft = f_true[j](xs) - np.trapezoid(f_true[j](xs), xs) / (xs[-1] - xs[0])
    ax.plot(xs, ft, "r-", lw=2.2, label="真值 $f_j$（居中）")
    order = np.argsort(X[:, j])
    ax.plot(X[order, j], f[j][order], "b--", lw=2.0,
            label="Backfitting 估计 $\\hat{f}_j$")
    ax.set_xlabel(f"$X_{j+1}$")
    ax.set_ylabel(names[j])
    ax.set_title(f"{names[j]}：真值 vs 估计")
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "fig1_backfitting_fit.png", dpi=150)
plt.close(fig)

# ============ 4. 图 2：收敛曲线 ============
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(range(1, n_iter + 1), history, "o-", color="seagreen")
ax.set_xlabel("迭代次数")
ax.set_ylabel("各函数最大变化量 $\\max_j \\max_i |\\Delta \\hat{f}_j(x_{ij})|$")
ax.set_yscale("log")
ax.set_title(f"Backfitting 收敛曲线（{n_iter} 次迭代到 $10^{{-5}}$）")
ax.grid(True, which="both", ls="--", alpha=0.4)
fig.tight_layout()
fig.savefig(OUT / "fig2_convergence.png", dpi=150)
plt.close(fig)

# ============ 5. 图 3：加性检查（拟合面 vs 观测） ============
yhat = alpha + f[0] + f[1]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].scatter(y, yhat, s=14, alpha=0.5, color="steelblue")
lims = [min(y.min(), yhat.min()), max(y.max(), yhat.max())]
axes[0].plot(lims, lims, "r--", lw=1.5)
axes[0].set_xlabel("观测 $y$")
axes[0].set_ylabel("拟合 $\\hat{y} = \\hat\\alpha + \\hat{f}_1 + \\hat{f}_2$")
axes[0].set_title("加性拟合 vs 观测")
axes[0].set_aspect("equal")

axes[1].hist(y - yhat, bins=40, color="darkorange", alpha=0.7)
axes[1].set_xlabel("残差 $y - \\hat{y}$")
axes[1].set_ylabel("频数")
axes[1].set_title(f"残差分布（均值 {np.mean(y-yhat):.3f}，标准差 {np.std(y-yhat):.3f}）")
fig.tight_layout()
fig.savefig(OUT / "fig3_additive_check.png", dpi=150)
plt.close(fig)

# ============ 6. 控制台输出 ============
print(f"alpha 估计 = {alpha:.3f}（真值 0.500；差异来自重中心化把 f2 的常数偏移吸收进 α）")
print(f"残差标准差 = {np.std(y - yhat):.3f}（噪声真值 0.15）")
print("图片已保存到：", OUT)
print("fig1_backfitting_fit.png / fig2_convergence.png / fig3_additive_check.png")
