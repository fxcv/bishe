from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_DIR = PROJECT_ROOT / "data_clean"

files = [
    "chn_senti_corp_train.csv",
    "chn_senti_corp_valid.csv",
    "chn_senti_corp_test.csv",
    "online_shopping_10_cats_train.csv",
    "online_shopping_10_cats_valid.csv",
    "online_shopping_10_cats_test.csv",
    "weibo_train.csv",
    "weibo_valid.csv",
    "weibo_test.csv",
]

for file_name in files:
    file_path = CLEAN_DIR / file_name
    print("\n" + "=" * 60)
    print(f"检查文件: {file_name}")

    df = pd.read_csv(file_path)

    print("行数, 列数:", df.shape)
    print("列名:", list(df.columns))
    print("前5行:")
    print(df.head())

    print("label 分布:")
    print(df["label"].value_counts(dropna=False).sort_index())

    print("是否有空值:")
    print(df[["text", "label"]].isnull().sum())