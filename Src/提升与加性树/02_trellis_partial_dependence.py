# -*- coding: utf-8 -*-
"""
提升树的解释（续）：交互效应、trellis 图 与 部分依赖的对比

对应笔记 §10.13.2（ESL 第10章 部分依赖图）。

演示目标：
1. 构造带「交互」的模拟数据
       Y = 2*X1 + 3*X1*X2 + 0.5*sin(2*pi*X2) + eps
   X1 的效应随 X2 变化（斜率 = 2 + 3*X2）——即 X1×X2 交互。
2. 训练梯度提升树（GBM, J=3 允许二阶交互）。
3. trellis 图：固定 X2 在低/中/高三个值，分别画 X1 的效应曲线，
   排列成 1x3 网格——「条件化切片」（相当于固定其他变量后逐片查看）。
4. 部分依赖图：把 X2（及其他变量）平均掉，画 X1 的边际平均效应曲线。
5. 对比：trellis 的三条「条件曲线」斜率随 X2 变化（交互可见），
   而「部分依赖」是一条平均曲线（斜率 = 2 + 3*E[X2]）。

输出图（pic/）：
    fig4_trellis_1d.png         trellis 图：X1 效应在 X2 = 低/中/高 下的切片
    fig5_2d_interaction.png     X1 × X2 二元部分依赖热力图（交互的全景）
    fig6_conditional_vs_marginal.png  条件曲线 vs 部分依赖曲线对比
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

rng = np.random.default_rng(23)

# ============ 1. 生成带交互的数据 ============
N, P = 3000, 6
X = rng.uniform(-1, 1, size=(N, P))       # X1,X2 真实变量；X3..X6 噪声
# 交互：X1 的斜率 = 2 + 3*X2
Y = 2.0 * X[:, 0] + 3.0 * X[:, 0] * X[:, 1] + 0.5 * np.sin(2 * np.pi * X[:, 1]) \
    + rng.normal(0, 0.15, size=N)

# ============ 2. 训练梯度提升树（J=3 允许二阶交互） ============
from sklearn.ensemble import GradientBoostingRegressor

gb = GradientBoostingRegressor(
    n_estimators=400, max_depth=3, learning_rate=0.1,
    loss="squared_error", random_state=0,
)
gb.fit(X, Y)

def predict_fixed(x1, x2, others_fill=0.0):
    """在 (X1,X2) 指定值下，其余变量固定为 others_fill 的预测"""
    x = np.ones(P) * others_fill
    x[0], x[1] = x1, x2
    return gb.predict(x.reshape(1, -1))[0]

def predict_average(x1, x2):
    """在 (X1,X2) 指定值下，其余变量取其训练值做平均（部分依赖 eq:10.48）"""
    Xm = X.copy()
    Xm[:, 0] = x1
    Xm[:, 1] = x2
    return gb.predict(Xm).mean()

x1_grid = np.linspace(-1, 1, 200)
x2_levels = [-0.7, 0.0, 0.7]                        # 低 / 中 / 高
x2_labels = ["$X_2$ = 低 ($-0.7$)", "$X_2$ = 中 ($0$)", "$X_2$ = 高 ($0.7$)"]

# ---------- 图 4：trellis 图（条件切片） ----------
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
for a, (x2v, lab) in enumerate(zip(x2_levels, x2_labels)):
    y_slice = [predict_fixed(x1, x2v) for x1 in x1_grid]
    # 真值：斜率 2+3*x2v 的直线（用于对照）
    axes[a].plot(x1_grid, y_slice, "b-", lw=2, label="拟合（条件）")
    axes[a].plot(x1_grid, (2 + 3 * x2v) * x1_grid + 0.5 * np.sin(2 * np.pi * x2v),
                 "r--", lw=1.2, label="真值")
    axes[a].set_title(lab, fontsize=11)
    axes[a].set_xlabel("$X_1$")
    axes[a].grid(alpha=0.3)
    axes[a].legend(fontsize=8, loc="upper left")
axes[0].set_ylabel("条件效应 $f(X_1, X_2)$")
fig.suptitle("Trellis 图：固定 $X_2$ 的三个取值，分别看 $X_1$ 的效应（斜率随 $X_2$ 变化 = 交互）", y=1.02)
fig.tight_layout()
fig.savefig(OUT / "fig4_trellis_1d.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------- 图 5：二元部分依赖热力图（交互全景） ----------
grid = 60
g1 = np.linspace(-1, 1, grid)
g2 = np.linspace(-1, 1, grid)
Z = np.empty((grid, grid))
for i, x1 in enumerate(g1):
    for j, x2 in enumerate(g2):
        Z[i, j] = predict_average(x1, x2)

fig, ax = plt.subplots(figsize=(6.5, 5.5))
cf = ax.contourf(g1, g2, Z.T, levels=30, cmap="viridis")
cs = ax.contour(g1, g2, Z.T, levels=12, colors="k", linewidths=0.4, alpha=0.4)
ax.clabel(cs, inline=True, fontsize=7, fmt="%.1f")
ax.set_xlabel("$X_1$")
ax.set_ylabel("$X_2$")
ax.set_title("二元部分依赖 $f(X_1, X_2)$（等高线近乎平行 = 近似加性乘积交互）")
fig.colorbar(cf, ax=ax, label="部分依赖值")
fig.tight_layout()
fig.savefig(OUT / "fig5_2d_interaction.png", dpi=150)
plt.close(fig)

# ---------- 图 6：条件曲线 vs 部分依赖曲线 ----------
fig, ax = plt.subplots(figsize=(7, 4.8))
# 三条条件曲线
for x2v, lab, c in zip(x2_levels, x2_labels, ["#C44E52", "#55A868", "#4C72B0"]):
    y_slice = [predict_fixed(x1, x2v) for x1 in x1_grid]
    ax.plot(x1_grid, y_slice, color=c, lw=1.8, label=lab)
# 部分依赖（平均掉 X2 及其他变量）
y_pd = [predict_average(x1, 0.0) for x1 in x1_grid]     # 平均在 predict_average 内完成
ax.plot(x1_grid, y_pd, "k-", lw=3, label="部分依赖 $f_{X_1}(X_1)$（平均掉 $X_2$）")
ax.set_xlabel("$X_1$")
ax.set_ylabel("效应")
ax.set_title("条件曲线（trellis 切片） vs 部分依赖（边际平均）")
ax.grid(alpha=0.3)
ax.legend(fontsize=9, loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "fig6_conditional_vs_marginal.png", dpi=150)
plt.close(fig)

print("图片已保存到:", OUT)
