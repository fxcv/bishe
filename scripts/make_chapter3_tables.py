from pathlib import Path
import pandas as pd

# =========================
# 路径
# =========================
PROJECT_DIR = Path(r"D:\Users\30126\Desktop\project")
INPUT_FILE = PROJECT_DIR / "analysis_data_for_thesis.xlsx"
OUTPUT_FILE = PROJECT_DIR / "chapter3_tables.xlsx"

# =========================
# 读取数据
# =========================
metrics_df = pd.read_excel(INPUT_FILE, sheet_name="metrics_summary")

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
    "roberta": "RoBERTa-wwm-ext",
    "bert_bigru": "BERT+BiGRU",
    "bert_lora": "BERT+LoRA",
}

metrics_df["dataset_name"] = metrics_df["dataset_name"].map(dataset_name_map).fillna(metrics_df["dataset_name"])
metrics_df["model_key"] = metrics_df["model_key"].map(model_name_map).fillna(metrics_df["model_key"])

dataset_order = ["ChnSentiCorp", "OnlineShopping10Cats", "WeiboSenti100k"]
model_order = ["BERT", "MacBERT", "RoBERTa-wwm-ext", "BERT+BiGRU", "BERT+LoRA"]

metrics_df["dataset_name"] = pd.Categorical(metrics_df["dataset_name"], categories=dataset_order, ordered=True)
metrics_df["model_key"] = pd.Categorical(metrics_df["model_key"], categories=model_order, ordered=True)
metrics_df = metrics_df.sort_values(["dataset_name", "model_key"]).copy()

# =========================
# 表3-4：总结果表
# =========================
table_3_4 = metrics_df[
    ["dataset_name", "model_key", "accuracy", "precision", "recall", "f1"]
].copy()

table_3_4.columns = ["数据集", "模型", "Accuracy", "Precision", "Recall", "F1-score"]

for col in ["Accuracy", "Precision", "Recall", "F1-score"]:
    table_3_4[col] = table_3_4[col].round(4)

# =========================
# 表3-5：BERT vs BERT+BiGRU
# =========================
table_3_5 = metrics_df[
    metrics_df["model_key"].isin(["BERT", "BERT+BiGRU"])
][["dataset_name", "model_key", "f1", "train_seconds"]].copy()

table_3_5.columns = ["数据集", "模型", "F1-score", "Train Seconds"]
table_3_5["F1-score"] = table_3_5["F1-score"].round(4)
table_3_5["Train Seconds"] = table_3_5["Train Seconds"].round(2)

# =========================
# 表3-6：BERT vs BERT+LoRA
# =========================
table_3_6 = metrics_df[
    metrics_df["model_key"].isin(["BERT", "BERT+LoRA"])
][["dataset_name", "model_key", "f1", "train_seconds"]].copy()

table_3_6.columns = ["数据集", "模型", "F1-score", "Train Seconds"]
table_3_6["F1-score"] = table_3_6["F1-score"].round(4)
table_3_6["Train Seconds"] = table_3_6["Train Seconds"].round(2)

# =========================
# 导出
# =========================
with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    table_3_4.to_excel(writer, sheet_name="表3-4_总结果表", index=False)
    table_3_5.to_excel(writer, sheet_name="表3-5_BERT_vs_BiGRU", index=False)
    table_3_6.to_excel(writer, sheet_name="表3-6_BERT_vs_LoRA", index=False)

print("第三章表格已生成：")
print(OUTPUT_FILE)