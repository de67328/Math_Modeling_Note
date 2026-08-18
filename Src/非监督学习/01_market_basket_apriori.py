# -*- coding: utf-8 -*-
"""
ESL 第14章 §14.2 关联规则：市场篮子分析与 Apriori 算法演示

对应笔记：Note/Chapters/ch_非监督学习.tex §14.2.1~§14.2.2。

本程序在「市场篮子」数据上完整走一遍 Apriori 流程：
1. 生成模拟购物数据（10 种商品，注入若干「捆绑购买」模式）；
2. 手写实现 Apriori：单项集支持度 → 逐层候选生成（自连接 + 剪枝）→ 扫描统计；
   演示「反单调性 / 向下闭包」剪枝：含非频繁子集的候选直接丢弃、无需扫描数据；
3. 从频繁项集分裂生成关联规则 A⇒B，计算 支持度 / 置信度 / 提升度（eq:14.7~14.8）；
4. 可视化：
   - fig1_apriori_support.png    频繁项集支持度（按 1/2/3-项集 分组条形图）
   - fig2_rules_conf_lift.png    关联规则散点（x=置信度, y=提升度, 点大小/颜色=支持度）
   - fig3_threshold_effect.png   支持度阈值 t 对频繁项集数量的影响（对数坐标，指数下降）

运行：python 01_market_basket_apriori.py
输出图保存于 pic/。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import combinations
from pathlib import Path

# ---------- 中文字体 ----------
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).parent / "pic"
OUT.mkdir(exist_ok=True)

rng = np.random.default_rng(42)

# ============ 1. 生成市场篮子数据 ============
P = 10
ITEM_NAMES = ["牛奶", "面包", "鸡蛋", "黄油", "咖啡",
              "茶", "啤酒", "尿布", "薯片", "纸巾"]
BASE_P = np.array([0.20, 0.20, 0.15, 0.12, 0.14, 0.10,
                   0.12, 0.11, 0.13, 0.16])


def gen_basket(N=2000):
    """生成 N 笔交易；每笔是商品编号的 frozenset（含捆绑注入）。"""
    baskets = []
    for _ in range(N):
        items = set(np.where(rng.random(P) < BASE_P)[0].tolist())
        if rng.random() < 0.30:      # 捆绑 {牛奶, 面包}
            items |= {0, 1}
        if rng.random() < 0.25:      # 捆绑 {啤酒, 尿布}（经典模式）
            items |= {6, 7}
        if rng.random() < 0.18:      # 捆绑 {鸡蛋, 黄油, 咖啡}
            items |= {2, 3, 4}
        baskets.append(frozenset(items))
    return baskets


# ============ 2. 手写 Apriori ============
def generate_candidates(Fk, k):
    """由频繁 k-项集生成候选 (k+1)-项集。

    自连接：前 k-1 个元素相同的两个项集合并；
    剪枝（反单调性 / 向下闭包）：新候选的每个 k-子集都必须已在 Fk 中，
    否则该候选及其一切超集不可能频繁，直接丢弃。
    """
    Fk_list = sorted(Fk)
    Fk_set = set(Fk)
    cand = set()
    for i in range(len(Fk_list)):
        for j in range(i + 1, len(Fk_list)):
            a, b = Fk_list[i], Fk_list[j]
            if a[:-1] == b[:-1]:                       # 前 k-1 个元素相同
                new = a + (b[-1],)
                if all(tuple(c) in Fk_set for c in combinations(new, k)):
                    cand.add(new)
    return cand


def apriori(baskets, min_sup):
    """Apriori：返回 {频繁项集(升序元组) : 支持度} 字典。"""
    N = len(baskets)
    freq = {}

    # 频繁 1-项集
    items = {i for b in baskets for i in b}
    Fk = set()
    for i in sorted(items):
        s = sum(1 for b in baskets if i in b) / N
        if s >= min_sup:
            freq[(i,)] = s
            Fk.add((i,))

    # 逐层扩展：k-项集 -> (k+1)-项集
    k = 1
    while Fk:
        C = generate_candidates(Fk, k)
        Fk = set()
        for c in C:
            cs = frozenset(c)
            s = sum(1 for b in baskets if cs.issubset(b)) / N
            if s >= min_sup:
                freq[tuple(c)] = s
                Fk.add(tuple(c))
        k += 1
    return freq


# ============ 3. 生成关联规则 ============
def gen_rules(freq, min_conf):
    """对每个频繁项集 K，枚举分裂 A⇒B（B = K\\A），计算三指标并筛置信度。

    注意：对大小为 |K| 的项集，形如 A⇒(K-A) 的规则共有 2^{|K|-1}-1 条。
    """
    rules = []
    for K, supK in freq.items():
        if len(K) < 2:
            continue
        for r in range(1, len(K)):
            for A in combinations(K, r):
                B = tuple(sorted(set(K) - set(A)))
                supA = freq[tuple(sorted(A))]
                conf = supK / supA                       # eq:14.8
                lift = conf / freq[B]                    # conf / T(B)
                rules.append((tuple(sorted(A)), B, supK, conf, lift))
    return [r for r in rules if r[3] >= min_conf]


def fmt(items):
    """项集 -> 可读中文串。"""
    return " + ".join(ITEM_NAMES[i] for i in items)


# ============ 主流程 ============
baskets = gen_basket(N=2000)
N = len(baskets)
print(f"交易数 N = {N}，商品数 P = {P}")

MIN_SUP = 0.06
MIN_CONF = 0.30
freq = apriori(baskets, min_sup=MIN_SUP)
print(f"min_sup={MIN_SUP} 时频繁项集个数 = {len(freq)}")
print("Top 5 频繁项集（按支持度）：")
for k, s in sorted(freq.items(), key=lambda kv: -kv[1])[:5]:
    print(f"   {{ {fmt(k)} }}  sup = {s:.3f}")

rules = gen_rules(freq, min_conf=MIN_CONF)
print(f"min_conf={MIN_CONF} 时关联规则条数 = {len(rules)}")
print("Top 6 规则（按提升度）：")
for A, B, sup, conf, lift in sorted(rules, key=lambda r: -r[4])[:6]:
    print(f"   {{ {fmt(A)} }} => {{ {fmt(B)} }}  "
          f"sup={sup:.3f} conf={conf:.3f} lift={lift:.3f}")

# ============ 图1：频繁项集支持度（按大小分组） ============
sizes = sorted(set(len(k) for k in freq))
fig, ax = plt.subplots(figsize=(9, 5))
cmap = plt.cm.viridis(np.linspace(0.2, 0.9, len(sizes)))
for si, sz in enumerate(sizes):
    fsz = sorted(((k, s) for k, s in freq.items() if len(k) == sz),
                 key=lambda kv: -kv[1])
    xs = np.arange(len(fsz))
    ax.bar(xs, [s for _, s in fsz], color=cmap[si], alpha=0.85,
           label=f"{sz}-项集（{len(fsz)} 个）")
    ax.set_xticks([])
ax.set_xlabel("频繁项集（按支持度降序，分组为 1/2/3-项集）")
ax.set_ylabel("支持度")
ax.set_title(f"Apriori 挖掘出的频繁项集及其支持度（min_sup={MIN_SUP}）")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "fig1_apriori_support.png", dpi=150)
plt.close(fig)

# ============ 图2：规则散点（置信度 vs 提升度） ============
fig, ax = plt.subplots(figsize=(8, 6))
sup = np.array([r[2] for r in rules])
conf = np.array([r[3] for r in rules])
lift = np.array([r[4] for r in rules])
sc = ax.scatter(conf, lift, s=200 * sup / sup.max() + 20, c=sup,
                cmap="plasma", alpha=0.8, edgecolors="k", linewidths=0.4)
plt.colorbar(sc, ax=ax, label="支持度")
ax.set_xlabel("置信度 conf")
ax.set_ylabel("提升度 lift")
ax.set_title(f"关联规则：置信度 vs 提升度（共 {len(rules)} 条，min_conf={MIN_CONF}）")
ax.axvline(1.0, color="gray", ls="--", lw=0.8)   # lift>1 才有正向关联
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "fig2_rules_conf_lift.png", dpi=150)
plt.close(fig)

# ============ 图3：支持度阈值 t 对频繁项集数量的影响 ============
thresholds = np.linspace(0.03, 0.30, 28)
counts = [len(apriori(baskets, min_sup=t)) for t in thresholds]
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(thresholds, counts, "o-", color="C0")
ax.set_xlabel("支持度阈值 t")
ax.set_ylabel("频繁项集个数")
ax.set_yscale("log")
ax.set_title("支持度阈值 t 对频繁项集数量的影响（对数坐标，指数下降）")
ax.grid(alpha=0.3, which="both")
plt.tight_layout()
plt.savefig(OUT / "fig3_threshold_effect.png", dpi=150)
plt.close(fig)

print("\n已输出图片：")
for f in sorted(OUT.glob("fig*.png")):
    print("  ", f.name)
