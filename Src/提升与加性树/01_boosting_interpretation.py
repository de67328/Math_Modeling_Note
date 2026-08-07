# -*- coding: utf-8 -*-
"""
提升树的解释：预测变量的相对重要性 + 部分依赖图

对应笔记 §10.13（ESL 第10章 10.13 Interpretation）。

内容：
1. 生成带「主效应 + 交互」的模拟数据
       Y = f1(X1) + f2(X2) + X3*X4 + eps
   f1 线性、f2 非线性、X3*X4 二阶交互——使 X1..X4 重要，其余为噪声。
2. 训练梯度提升树（GradientBoostingRegressor，小树 J=2 允许交互）。
3. 手动实现教材 eq:10.42 / eq:10.43 的「相对重要性」：
       I_l^2(T) = sum_{t=1}^{J-1} i_t^2 * I(v(t)=l)     （单棵树：累加分裂改进）
       I_l^2     = (1/M) sum_m I_l^2(T_m)               （对 M 棵树平均）
   并归一化使最大者为 100（教材习惯）。
4. 手动实现 eq:10.47 / eq:10.48 的「部分依赖」：
       f_S(X_S)   = E_{X_C} f(X_S, X_C)
       bar f_S    = (1/N) sum_i f(X_S, x_iC)            （其他变量取训练值平均）
   画一元部分依赖图与二元（X3 × X4 交互）热力图。

输出图（pic/）：
    fig1_relative_importance.png   相对重要性条形图
    fig2_partial_dependence_1d.png 一元部分依赖（X1..X4）
    fig3_partial_dependence_2d.png 二元部分依赖热力图（X3 × X4 交互）
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

rng = np.random.default_rng(11)

# ============ 1. 生成模拟数据（主效应 + 交互） ============
N, P = 2000, 8
X = rng.uniform(-1, 1, size=(N, P))          # 8 个特征，X1..X8
f1 = lambda x: 3.0 * x                        # X1 线性主效应
f2 = lambda x: 2.0 * np.sin(2 * np.pi * x)    # X2 非线性主效应
Y = f1(X[:, 0]) + f2(X[:, 1]) + 4.0 * X[:, 2] * X[:, 3] + rng.normal(0, 0.3, size=N)
# X3*X4 是二阶交互；X5..X8 是纯噪声（应重要性低）

# ============ 2. 训练梯度提升树 ============
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

# J = 2 终端节点（stump 对）允许二阶交互；学习率收缩 + 中等迭代数
M = 300
gb = GradientBoostingRegressor(
    n_estimators=M, max_depth=2, learning_rate=0.1,
    loss="squared_error", random_state=0,
)
gb.fit(X, Y)

# ============ 3. 手动实现相对重要性 eq:10.42 / eq:10.43 ============
def tree_importance_single(tree, P):
    """单棵树 eq:10.42：I_l^2(T) = sum_t i_t^2 * I(v(t)=l)

    与 sklearn CART 一致：分裂改进 = 节点加权不纯度 - 子节点加权不纯度
    （即 impurity * n_samples 之差，不除以样本数）。
    """
    tr = tree.tree_
    imp = np.zeros(P)
    children_left, children_right = tr.children_left, tr.children_right
    feature, impurity = tr.feature, tr.impurity
    ns = tr.weighted_n_node_samples
    for t in range(tr.node_count):
        if children_left[t] != children_right[t]:            # 内部节点
            l, r = children_left[t], children_right[t]
            # 加权不纯度减少（sklearn 特征重要性的内部定义）
            gain = ns[t] * impurity[t] - ns[l] * impurity[l] - ns[r] * impurity[r]
            imp[feature[t]] += gain
    return imp

def relative_importance(models, P):
    """eq:10.43：I_l^2 = (1/M) sum_m I_l^2(T_m)，再归一化使最大者为 100"""
    acc = np.zeros(P)
    for m in models:
        acc += tree_importance_single(m, P)
    acc /= len(models)
    acc = np.sqrt(np.maximum(acc, 0))        # 平方重要性开根 -> 实际重要性
    acc = 100.0 * acc / acc.max()            # 归一化：最大 = 100
    return acc

# estimators_ 是 (n_estimators,) 的回归树数组
importance = relative_importance(gb.estimators_[:, 0], P)

# 与 sklearn 内置对比（应大致一致，方向相同）
sk_imp = 100.0 * gb.feature_importances_ / gb.feature_importances_.max()
print("手动实现 重要性:", np.round(importance, 1))
print("sklearn 内置重要性:", np.round(sk_imp, 1))

# ---------- 图 1：相对重要性条形图 ----------
fig, ax = plt.subplots(figsize=(7, 4.5))
xpos = np.arange(P)
ax.bar(xpos, importance, color="#4C72B0", edgecolor="black", linewidth=0.5)
ax.set_xticks(xpos)
ax.set_xticklabels([f"$X_{j+1}$" for j in range(P)])
ax.set_ylabel("相对重要性")
ax.set_title("预测变量的相对重要性（eq:10.43，最大者 = 100）")
ax.grid(axis="y", alpha=0.3)
# 标注真值中重要变量
for j in [0, 1, 2, 3]:
    ax.get_xticklabels()[j].set_color("red")
fig.tight_layout()
fig.savefig(OUT / "fig1_relative_importance.png", dpi=150)
plt.close(fig)

# ============ 4. 手动实现部分依赖 eq:10.47 / eq:10.48 ============
def partial_dependence_1d(model, X, j, grid=50):
    """一元部分依赖 eq:10.48：bar f_S(x) = (1/N) sum_i f(x, x_iC)"""
    xgrid = np.linspace(X[:, j].min(), X[:, j].max(), grid)
    vals = np.empty(grid)
    Xm = X.copy()
    for k, xv in enumerate(xgrid):
        Xm[:, j] = xv
        vals[k] = model.predict(Xm).mean()
    return xgrid, vals

def partial_dependence_2d(model, X, j1, j2, grid=40):
    """二元部分依赖热力图：固定 (X_{j1}, X_{j2})，其余取训练值平均"""
    g1 = np.linspace(X[:, j1].min(), X[:, j1].max(), grid)
    g2 = np.linspace(X[:, j2].min(), X[:, j2].max(), grid)
    Z = np.empty((grid, grid))
    Xm = X.copy()
    for a, x1 in enumerate(g1):
        for b, x2 in enumerate(g2):
            Xm[:, j1] = x1
            Xm[:, j2] = x2
            Z[a, b] = model.predict(Xm).mean()
    return g1, g2, Z

# ---------- 图 2：一元部分依赖（X1..X4） ----------
fig, axes = plt.subplots(1, 4, figsize=(15, 4))
for a, j in enumerate([0, 1, 2, 3]):
    xg, pd = partial_dependence_1d(gb, X, j)
    axes[a].plot(xg, pd, "b-", lw=2)
    axes[a].set_title(f"$X_{j+1}$ 的部分依赖")
    axes[a].set_xlabel(f"$X_{j+1}$")
    axes[a].grid(alpha=0.3)
axes[0].set_ylabel("部分依赖 $f_S(X_S)$")
fig.suptitle("一元部分依赖图（eq:10.48，其余变量取训练值平均）", y=1.02)
fig.tight_layout()
fig.savefig(OUT / "fig2_partial_dependence_1d.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------- 图 3：二元部分依赖热力图（X3 × X4 交互） ----------
g1, g2, Z = partial_dependence_2d(gb, X, 2, 3)
fig, ax = plt.subplots(figsize=(6.5, 5.5))
cf = ax.contourf(g1, g2, Z.T, levels=30, cmap="viridis")
ax.contour(g1, g2, Z.T, levels=12, colors="k", linewidths=0.4, alpha=0.4)
ax.set_xlabel("$X_3$")
ax.set_ylabel("$X_4$")
ax.set_title("二元部分依赖：$X_3 \\times X_4$ 交互")
fig.colorbar(cf, ax=ax, label="部分依赖值")
fig.tight_layout()
fig.savefig(OUT / "fig3_partial_dependence_2d.png", dpi=150)
plt.close(fig)

print("图片已保存到:", OUT)
