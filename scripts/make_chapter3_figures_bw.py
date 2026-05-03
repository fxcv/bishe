from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# =========================
# 路径
# =========================
PROJECT_DIR = Path(r"D:\Users\30126\Desktop\project")
INPUT_FILE = PROJECT_DIR / "analysis_data_for_thesis.xlsx"
FIG_DIR = PROJECT_DIR / "chapter3_figures_bw"
FIG_DIR.mkdir(exist_ok=True)

# =========================
# 中文与负号显示
# =========================
rcParams["font.sans-serif"] = [
    "SimSun",
    "SimHei",
    "Microsoft YaHei",
    "Arial Unicode MS",
    "Noto Sans CJK SC",
]
rcParams["axes.unicode_minus"] = False

# 统一字体大小，贴近论文插图风格
plt.rcParams["font.size"] = 11
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10

# =========================
# 读取数据
# =========================
metrics_df = pd.read_excel(INPUT_FILE, sheet_name="metrics_summary")
loss_df = pd.read_excel(INPUT_FILE, sheet_name="loss_summary")

# =========================
# 名称映射
# =========================
dataset_name_map = {
    "ChnSentiCorp": "ChnSentiCorp",
    "OnlineShopping10Cats": "OnlineShopping10Cats",
    "WeiboSenti100k": "WeiboSenti100k",
}

model_name_map = {
    "bert": "BERT",
    "macbert": "MacBERT",
    "roberta": "RoBERTa",
    "bert_bigru": "BERT+BiGRU",
    "bert_lora": "BERT+LoRA",
}

metrics_df["dataset_name"] = metrics_df["dataset_name"].map(dataset_name_map).fillna(metrics_df["dataset_name"])
metrics_df["model_key"] = metrics_df["model_key"].map(model_name_map).fillna(metrics_df["model_key"])

loss_df["dataset_name"] = loss_df["dataset_name"].map(dataset_name_map).fillna(loss_df["dataset_name"])
loss_df["model_key"] = loss_df["model_key"].map(model_name_map).fillna(loss_df["model_key"])

dataset_order = ["ChnSentiCorp", "OnlineShopping10Cats", "WeiboSenti100k"]
model_order = ["BERT", "MacBERT", "RoBERTa", "BERT+BiGRU", "BERT+LoRA"]

# =========================
# 图3-1：F1柱状图（模仿论文风格）
# =========================
plot_df = metrics_df[["dataset_name", "model_key", "f1"]].copy()
plot_df = plot_df[plot_df["dataset_name"].isin(dataset_order)]
plot_df = plot_df[plot_df["model_key"].isin(model_order)]

pivot_f1 = plot_df.pivot(index="dataset_name", columns="model_key", values="f1")
pivot_f1 = pivot_f1.reindex(index=dataset_order, columns=model_order)

x = np.arange(len(dataset_order))
width = 0.14

# 模仿论文中黑白纹理柱状图
hatches = ["", "//", "xx", "..", "\\\\"]

fig, ax = plt.subplots(figsize=(10, 5.8))

for i, model in enumerate(model_order):
    bars = ax.bar(
        x + (i - 2) * width,
        pivot_f1[model] * 100,
        width=width,
        label=model,
        facecolor="white",
        edgecolor="black",
        linewidth=1.0,
    )
    for bar in bars:
        bar.set_hatch(hatches[i])

# y轴不从0开始，突出差异
all_f1 = (pivot_f1.values.flatten() * 100)
all_f1 = all_f1[~np.isnan(all_f1)]
ymin = np.floor(all_f1.min() - 1)
ymax = np.ceil(all_f1.max() + 1)
ax.set_ylim(ymin, ymax)

ax.set_ylabel("F1值/%")
ax.set_xticks(x)
ax.set_xticklabels(dataset_order)

# 不加主标题，模仿论文图
# 图例放在图内上方，类似你给的示例
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 0.995),
    ncol=3,
    frameon=False,
    handlelength=1.6,
    columnspacing=1.2,
)

# 只保留左下坐标轴，更像论文图
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# y轴淡淡辅助线
ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.5)
ax.set_axisbelow(True)

plt.tight_layout()

f1_png = FIG_DIR / "fig3_1_f1_compare_bw.png"
f1_svg = FIG_DIR / "fig3_1_f1_compare_bw.svg"
plt.savefig(f1_png, dpi=600, bbox_inches="tight", facecolor="white")
plt.savefig(f1_svg, bbox_inches="tight", facecolor="white")
plt.close()

# =========================
# 图3-2：Loss折线图（模仿论文风格）
# =========================
# 图3-2：Loss折线图（模仿论文风格）
# =========================
plot_loss_df = loss_df[["dataset_name", "model_key", "min_eval_loss"]].copy()
plot_loss_df = plot_loss_df[plot_loss_df["dataset_name"].isin(dataset_order)]
plot_loss_df = plot_loss_df[plot_loss_df["model_key"].isin(model_order)]

pivot_loss = plot_loss_df.pivot(index="model_key", columns="dataset_name", values="min_eval_loss")
pivot_loss = pivot_loss.reindex(index=model_order, columns=dataset_order)

fig, ax = plt.subplots(figsize=(10, 5.8))

# 黑白折线 + 不同点型
line_styles = ["-", "--", "-."]
markers = ["^", "o", "s"]

for i, dataset in enumerate(dataset_order):
    ax.plot(
        model_order,
        pivot_loss[dataset],
        linestyle=line_styles[i],
        marker=markers[i],
        color="black",
        linewidth=1.2,
        markersize=5.5,
        label=dataset,
    )

ax.set_ylabel("Loss值")
ax.set_xticks(range(len(model_order)))
ax.set_xticklabels(model_order)

# 图例移到图外上方，避免挡住曲线
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 0.98),
    ncol=3,
    frameon=False,
    handlelength=2.2,
    columnspacing=1.5,
)
# 只保留下边和左边框线
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.grid(False)
plt.tight_layout()

loss_png = FIG_DIR / "fig3_2_loss_compare_bw.png"
loss_svg = FIG_DIR / "fig3_2_loss_compare_bw.svg"
plt.savefig(loss_png, dpi=600, bbox_inches="tight", facecolor="white")
plt.savefig(loss_svg, bbox_inches="tight", facecolor="white")
plt.close()

print("论文风格黑白图已生成：")
print(f1_png)
print(f1_svg)
print(loss_png)
print(loss_svg)