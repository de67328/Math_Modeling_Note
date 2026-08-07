# -*- coding: utf-8 -*-
"""
部分依赖图的形态 → 交互类型 可视化

对应笔记 §10.13.2（部分依赖图）与 §10.11（交互阶数）。

演示目标：同样的 X1,X2，用四种不同的「真值结构」各训练一个 GBM，
看它们二维部分依赖图（等高线）与 trellis 条件切片的形态差异：

    1. 加性（无交互）   Y = 2X1 + X2            → 等高线平行、等距
    2. 乘积交互         Y = 2X1 + X2 + 3X1X2    → 双曲线 / 鞍形
    3. 阈值交互         Y = 2X1 + X2 + 3X1·I(X2>0) → 在 X2=0 处斜率突变
    4. 纯交互（无主效应）Y = 3X1X2              → 中心鞍点，无平行带

输出图（pic/）：
    fig7_interaction_2d.png    2×2 网格：四类交互的二维部分依赖等高线
    fig8_interaction_trellis.png 2×2 网格：四类交互的 trellis 条件切片
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

rng = np.random.default_rng(31)

# ============ 数据生成：四种结构 ============
N = 5000
X12 = rng.uniform(-1, 1, size=(N, 2))
X1, X2 = X12[:, 0], X12[:, 1]
eps = rng.normal(0, 0.05, size=N)   # 小噪声，保证形态清晰

structures = [
    ("加性（无交互）",  2 * X1 + X2 + eps),
    ("乘积交互 $X_1 X_2$", 2 * X1 + X2 + 3.0 * X1 * X2 + eps),
    ("阈值交互 $X_1 I(X_2>0)$", 2 * X1 + X2 + 3.0 * X1 * (X2 > 0) + eps),
    ("纯交互（无主效应）", 3.0 * X1 * X2 + eps),
]

# ============ 训练 GBM 并计算部分依赖 ============
from sklearn.ensemble import GradientBoostingRegressor

def fit_and_partial2d(Y, grid=55):
    """训练 GBM（J=3，允许二阶交互），返回 X1,X2 网格上的二元部分依赖"""
    X = np.column_stack([X12, rng.uniform(-1, 1, size=(N, 2))])   # 加 2 个噪声特征
    gb = GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.1,
        loss="squared_error", random_state=0,
    )
    gb.fit(X, Y)
    g1 = np.linspace(-1, 1, grid)
    g2 = np.linspace(-1, 1, grid)
    Z = np.empty((grid, grid))
    Xm = X.copy()
    for i, x1 in enumerate(g1):
        for j, x2 in enumerate(g2):
            Xm[:, 0] = x1
            Xm[:, 1] = x2
            Z[i, j] = gb.predict(Xm).mean()       # eq:10.48 部分依赖
    return g1, g2, Z

def conditional_slice(x2_fixed, x1_grid, Y):
    """trellis 条件切片：固定 X2=x2_fixed，看 X1 的效应（GBM 预测，其余特征取均值）"""
    X = np.column_stack([X12, rng.uniform(-1, 1, size=(N, 2))])
    gb = GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.1, random_state=0,
    )
    gb.fit(X, Y)
    Xm = X.copy()
    out = np.empty(len(x1_grid))
    for i, x1 in enumerate(x1_grid):
        Xm[:, 0] = x1
        Xm[:, 1] = x2_fixed
        out[i] = gb.predict(Xm).mean()
    return out

x1g = np.linspace(-1, 1, 200)
x2_levels = [-0.6, 0.0, 0.6]
x2_labels = ["$X_2$=低", "$X_2$=中", "$X_2$=高"]
colors = ["#C44E52", "#55A868", "#4C72B0"]

# ---------- 图 7：2×2 二维部分依赖等高线 ----------
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
for ax, (name, Y) in zip(axes.ravel(), structures):
    g1, g2, Z = fit_and_partial2d(Y)
    cf = ax.contourf(g1, g2, Z.T, levels=30, cmap="viridis")
    cs = ax.contour(g1, g2, Z.T, levels=12, colors="k", linewidths=0.4, alpha=0.5)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.1f")
    ax.set_title(name, fontsize=12)
    ax.set_xlabel("$X_1$")
    ax.set_ylabel("$X_2$")
    fig.colorbar(cf, ax=ax, shrink=0.85)
fig.suptitle("二维部分依赖图的形态 → 交互类型（等高线是否平行？是否弯曲/鞍形？）", y=1.01, fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "fig7_interaction_2d.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------- 图 8：2×2 trellis 条件切片 ----------
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, (name, Y) in zip(axes.ravel(), structures):
    for x2v, lab, c in zip(x2_levels, x2_labels, colors):
        y_slice = conditional_slice(x2v, x1g, Y)
        ax.plot(x1g, y_slice, color=c, lw=1.8, label=lab)
    ax.set_title(name, fontsize=12)
    ax.set_xlabel("$X_1$")
    ax.set_ylabel("条件效应 $f(X_1, X_2)$")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
fig.suptitle("Trellis 条件切片：固定 $X_2$ 三水平看 $X_1$ 效应（切片形状是否一致？）", y=1.01, fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "fig8_interaction_trellis.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("图片已保存到:", OUT)
print("判断要点：")
print("  加性：等高线平行等距；trellis 切片只平移、形状相同")
print("  乘积交互：等高线双曲/鞍形；trellis 斜率随 X2 变化")
print("  阈值交互：X2=0 处斜率突变；trellis 切片在阈值两侧形状突变")
print("  纯交互：中心鞍点；trellis 切片在 X2=0 两侧斜率反号")
