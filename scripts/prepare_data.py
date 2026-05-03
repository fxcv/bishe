from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CLEAN_DIR = PROJECT_ROOT / "data_clean"
CLEAN_DIR.mkdir(exist_ok=True)

def process_chnsenticorp_from_tsv():
    print("\n===== 处理 ChnSentiCorp（train/dev/test.tsv）=====")

    train_df = pd.read_csv(DATA_DIR / "train.tsv", sep="\t")
    dev_df = pd.read_csv(DATA_DIR / "dev.tsv", sep="\t")
    test_df = pd.read_csv(DATA_DIR / "test.tsv", sep="\t")

    print("train.tsv 列名:", list(train_df.columns))
    print("dev.tsv 列名:", list(dev_df.columns))
    print("test.tsv 列名:", list(test_df.columns))

    # ChnSentiCorp 的 tsv 格式是: label, text_a
    train_df = train_df[["text_a", "label"]].copy()
    dev_df = dev_df[["text_a", "label"]].copy()
    test_df = test_df[["text_a", "label"]].copy()

    train_df.columns = ["text", "label"]
    dev_df.columns = ["text", "label"]
    test_df.columns = ["text", "label"]

    train_df["text"] = train_df["text"].astype(str).str.strip()
    dev_df["text"] = dev_df["text"].astype(str).str.strip()
    test_df["text"] = test_df["text"].astype(str).str.strip()

    train_df = train_df.dropna(subset=["text", "label"])
    dev_df = dev_df.dropna(subset=["text", "label"])
    test_df = test_df.dropna(subset=["text", "label"])

    train_df = train_df[train_df["text"] != ""]
    dev_df = dev_df[dev_df["text"] != ""]
    test_df = test_df[test_df["text"] != ""]

    train_df.to_csv(CLEAN_DIR / "chn_senti_corp_train.csv", index=False, encoding="utf-8-sig")
    dev_df.to_csv(CLEAN_DIR / "chn_senti_corp_valid.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(CLEAN_DIR / "chn_senti_corp_test.csv", index=False, encoding="utf-8-sig")

    print("已生成：")
    print("data_clean/chn_senti_corp_train.csv")
    print("data_clean/chn_senti_corp_valid.csv")
    print("data_clean/chn_senti_corp_test.csv")


def process_online_shopping():
    print("\n===== 处理 Online Shopping 10 Cats =====")

    df = pd.read_csv(DATA_DIR / "online_shopping_10_cats.csv")
    print("online_shopping_10_cats.csv 列名:", list(df.columns))

    # online_shopping_10_cats.csv 格式是: cat, label, review
    df = df[["review", "label"]].copy()
    df.columns = ["text", "label"]

    df["text"] = df["text"].astype(str).str.strip()
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"] != ""]

    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    valid_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=temp_df["label"]
    )

    train_df.to_csv(CLEAN_DIR / "online_shopping_10_cats_train.csv", index=False, encoding="utf-8-sig")
    valid_df.to_csv(CLEAN_DIR / "online_shopping_10_cats_valid.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(CLEAN_DIR / "online_shopping_10_cats_test.csv", index=False, encoding="utf-8-sig")

    print("已生成：")
    print("data_clean/online_shopping_10_cats_train.csv")
    print("data_clean/online_shopping_10_cats_valid.csv")
    print("data_clean/online_shopping_10_cats_test.csv")


def process_weibo_senti_100k():
    print("\n===== 处理 Weibo Senti 100k =====")

    df = pd.read_csv(DATA_DIR / "weibo_senti_100k.csv")
    print("weibo_senti_100k.csv 列名:", list(df.columns))

    # 自动识别文本列和标签列
    text_col = None
    label_col = None

    for c in df.columns:
        c_low = str(c).lower()
        if text_col is None and c_low in ["text", "content", "review", "comment"]:
            text_col = c
        if label_col is None and c_low in ["label", "sentiment", "class", "target"]:
            label_col = c

    # 如果没识别出来，默认第一列文本，第二列标签
    if text_col is None:
        text_col = df.columns[0]
    if label_col is None:
        label_col = df.columns[1]

    df = df[[text_col, label_col]].copy()
    df.columns = ["text", "label"]

    df["text"] = df["text"].astype(str).str.strip()
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"] != ""]

    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    valid_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=temp_df["label"]
    )

    train_df.to_csv(CLEAN_DIR / "weibo_train.csv", index=False, encoding="utf-8-sig")
    valid_df.to_csv(CLEAN_DIR / "weibo_valid.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(CLEAN_DIR / "weibo_test.csv", index=False, encoding="utf-8-sig")

    print("已生成：")
    print("data_clean/weibo_train.csv")
    print("data_clean/weibo_valid.csv")
    print("data_clean/weibo_test.csv")


if __name__ == "__main__":
    process_chnsenticorp_from_tsv()
    process_online_shopping()
    process_weibo_senti_100k()
    print("\n当前这一步完成：已处理 3 个数据集。")