from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =========================
# 路径
# =========================
PROJECT_DIR = Path(r"D:\Users\30126\Desktop\project")
INPUT_FILE = PROJECT_DIR / "analysis_data_for_thesis.xlsx"
FIG_DIR = PROJECT_DIR / "chapter3_figures"
FIG_DIR.mkdir(exist_ok=True)

# =========================
# 读取数据
# =========================
metrics_df = pd.read_excel(INPUT_FILE, sheet_name="metrics_summary")
loss_df = pd.read_excel(INPUT_FILE, sheet_name="loss_summary")

# =========================
# 统一名称
# =========================
dataset_name_map = {
    "ChnSentiCorp": "ChnSentiCorp",
    "OnlineShopping10Cats": "OnlineShopping10Cats",
    "WeiboSenti100k": "WeiboSenti100k",
}

model_name_map = {
    "bert": "BERT",
    "macbert": "MacBERT",
    "roberta": "RoBERTa-wwm-ext",
    "bert_bigru": "BERT+BiGRU",
    "bert_lora": "BERT+LoRA",
}

metrics_df["dataset_name"] = metrics_df["dataset_name"].map(dataset_name_map).fillna(metrics_df["dataset_name"])
metrics_df["model_key"] = metrics_df["model_key"].map(model_name_map).fillna(metrics_df["model_key"])

loss_df["dataset_name"] = loss_df["dataset_name"].map(dataset_name_map).fillna(loss_df["dataset_name"])
loss_df["model_key"] = loss_df["model_key"].map(model_name_map).fillna(loss_df["model_key"])

dataset_order = ["ChnSentiCorp", "OnlineShopping10Cats", "WeiboSenti100k"]
model_order = ["BERT", "MacBERT", "RoBERTa-wwm-ext", "BERT+BiGRU", "BERT+LoRA"]

# =========================
# 图3-1：F1 对比图
# =========================
plot_df = metrics_df[["dataset_name", "model_key", "f1"]].copy()
plot_df = plot_df[plot_df["dataset_name"].isin(dataset_order)]
plot_df = plot_df[plot_df["model_key"].isin(model_order)]

pivot_f1 = plot_df.pivot(index="dataset_name", columns="model_key", values="f1")
pivot_f1 = pivot_f1.reindex(index=dataset_order, columns=model_order)

x = np.arange(len(dataset_order))
width = 0.15

fig, ax = plt.subplots(figsize=(12, 6))

for i, model in enumerate(model_order):
    ax.bar(x + (i - 2) * width, pivot_f1[model] * 100, width=width, label=model)

ax.set_xticks(x)
ax.set_xticklabels(dataset_order)
ax.set_ylabel("F1 (%)")
ax.set_xlabel("数据集")
ax.set_title("五种模型在三个数据集上的F1值对比")
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
plt.tight_layout()

f1_png = FIG_DIR / "fig3_1_f1_compare.png"
f1_svg = FIG_DIR / "fig3_1_f1_compare.svg"
plt.savefig(f1_png, dpi=300, bbox_inches="tight")
plt.savefig(f1_svg, bbox_inches="tight")
plt.close()

# =========================
# 图3-2：最小验证损失图
# =========================
plot_loss_df = loss_df[["dataset_name", "model_key", "min_eval_loss"]].copy()
plot_loss_df = plot_loss_df[plot_loss_df["dataset_name"].isin(dataset_order)]
plot_loss_df = plot_loss_df[plot_loss_df["model_key"].isin(model_order)]

pivot_loss = plot_loss_df.pivot(index="model_key", columns="dataset_name", values="min_eval_loss")
pivot_loss = pivot_loss.reindex(index=model_order, columns=dataset_order)

fig, ax = plt.subplots(figsize=(12, 6))

for dataset in dataset_order:
    ax.plot(model_order, pivot_loss[dataset], marker="o", label=dataset)

ax.set_ylabel("Loss")
ax.set_xlabel("模型")
ax.set_title("五种模型在三个数据集上的最小验证损失对比")
ax.legend()
plt.xticks(rotation=15)
plt.tight_layout()

loss_png = FIG_DIR / "fig3_2_loss_compare.png"
loss_svg = FIG_DIR / "fig3_2_loss_compare.svg"
plt.savefig(loss_png, dpi=300, bbox_inches="tight")
plt.savefig(loss_svg, bbox_inches="tight")
plt.close()

print("图已生成：")
print(f1_png)
print(f1_svg)
print(loss_png)
print(loss_svg)