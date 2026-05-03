import pandas as pd
from pathlib import Path

# ========= 这里改成你自己的文件路径 =========

TRAIN_FILE = r"D:\Users\30126\Desktop\project\data_clean\weibo_train.csv"
VAL_FILE = r"D:\Users\30126\Desktop\project\data_clean\weibo_valid.csv"
TEST_FILE  = r"D:\Users\30126\Desktop\project\data_clean\weibo_test.csv"

# 标签列名
LABEL_COL = "label"

# 如果你的标签是数字，写映射；如果已经是中文标签，可以删掉这个映射
LABEL_MAP = {
    1: "正向",
    0: "负向"
    # 如果你是三分类就改成：
    # 1: "积极",
    # 0: "中立",
    # -1: "消极"
}
# =========================================


def read_file(file_path: str) -> pd.DataFrame:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    elif path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t")
    elif path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    else:
        raise ValueError(f"暂不支持的文件格式: {path.suffix}")


def count_labels(df: pd.DataFrame, label_col: str) -> pd.Series:
    if label_col not in df.columns:
        raise KeyError(f"数据中找不到标签列: {label_col}，当前列名: {list(df.columns)}")
    return df[label_col].value_counts().sort_index()


def convert_label_name(label_value):
    return LABEL_MAP.get(label_value, str(label_value))


def main():
    train_df = read_file(TRAIN_FILE)
    val_df = read_file(VAL_FILE)
    test_df = read_file(TEST_FILE)

    train_counts = count_labels(train_df, LABEL_COL)
    val_counts = count_labels(val_df, LABEL_COL)
    test_counts = count_labels(test_df, LABEL_COL)

    all_labels = sorted(set(train_counts.index) | set(val_counts.index) | set(test_counts.index))

    rows = []
    for label in all_labels:
        rows.append({
            "情感标签": convert_label_name(label),
            "训练集": int(train_counts.get(label, 0)),
            "验证集": int(val_counts.get(label, 0)),
            "测试集": int(test_counts.get(label, 0)),
        })

    result_df = pd.DataFrame(rows)

    total_row = {
        "情感标签": "合计",
        "训练集": int(result_df["训练集"].sum()),
        "验证集": int(result_df["验证集"].sum()),
        "测试集": int(result_df["测试集"].sum()),
    }
    result_df = pd.concat([result_df, pd.DataFrame([total_row])], ignore_index=True)

    print("\n===== 数据集划分统计结果 =====")
    print(result_df.to_string(index=False))

    save_path = Path("dataset_split_stats.xlsx")
    result_df.to_excel(save_path, index=False)
    print(f"\n统计结果已保存到: {save_path.resolve()}")


if __name__ == "__main__":
    main()