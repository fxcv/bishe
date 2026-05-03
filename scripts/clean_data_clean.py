from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Optional

import pandas as pd


PROJECT_DIR = Path(r"D:\Users\30126\Desktop\project")
DATA_CLEAN_DIR = PROJECT_DIR / "data_clean"
BACKUP_DIR = PROJECT_DIR / "data_clean_backup_before_reclean"


FILES = [
    DATA_CLEAN_DIR / "chn_senti_corp_train.csv",
    DATA_CLEAN_DIR / "chn_senti_corp_valid.csv",
    DATA_CLEAN_DIR / "chn_senti_corp_test.csv",

    DATA_CLEAN_DIR / "online_shopping_10_cats_train.csv",
    DATA_CLEAN_DIR / "online_shopping_10_cats_valid.csv",
    DATA_CLEAN_DIR / "online_shopping_10_cats_test.csv",

    DATA_CLEAN_DIR / "weibo_train.csv",
    DATA_CLEAN_DIR / "weibo_valid.csv",
    DATA_CLEAN_DIR / "weibo_test.csv",
]


def clean_text(text: object, is_weibo: bool = False) -> str:
    if pd.isna(text):
        return ""

    s = str(text)

    # 去掉 BOM / 不可见字符
    s = s.replace("\ufeff", "").replace("\u200b", "").replace("\xa0", " ")

    # 去 html 标签
    s = re.sub(r"<[^>]+>", " ", s)

    # 去网址
    s = re.sub(r"http[s]?://\S+", " ", s)
    s = re.sub(r"www\.\S+", " ", s)

    # 微博特殊清洗
    if is_weibo:
        s = re.sub(r"//@.*", " ", s)                   # 转发链
        s = re.sub(r"回复@[^:： ]+[:：]?", " ", s)      # 回复@
        s = re.sub(r"@[\w\-\u4e00-\u9fa5]+", " ", s)   # @用户
        s = re.sub(r"#([^#]+)#", r"\1", s)             # 话题保留内容

    # 去控制字符
    s = re.sub(r"[\r\n\t]+", " ", s)

    # 连续标点压缩
    s = re.sub(r"[！!]{2,}", "！", s)
    s = re.sub(r"[？?]{2,}", "？", s)
    s = re.sub(r"[。\.]{2,}", "。", s)
    s = re.sub(r"[,，]{2,}", "，", s)

    # 压缩多余空格
    s = re.sub(r"\s+", " ", s).strip()

    return s


def invalid_text(text: str) -> bool:
    if not text:
        return True

    # 太短基本没信息
    if len(text) < 2:
        return True

    # 全是符号/数字/空格
    if re.fullmatch(r"[\W\d_]+", text, flags=re.UNICODE):
        return True

    return False


def normalize_label(x) -> Optional[int]:
    if pd.isna(x):
        return None

    mapping = {
        1: 1, 0: 0,
        "1": 1, "0": 0,
        1.0: 1, 0.0: 0,
        "1.0": 1, "0.0": 0,
        "positive": 1, "negative": 0,
        "pos": 1, "neg": 0,
        "正向": 1, "负向": 0,
        "积极": 1, "消极": 0,
        "好评": 1, "差评": 0,
    }

    if x in mapping:
        return mapping[x]

    sx = str(x).strip().lower()
    if sx in mapping:
        return mapping[sx]

    return None


def read_csv_safe(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        try:
            return pd.read_csv(path, encoding="utf-8")
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="gbk")


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # 去掉列名前后空格
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]

    # 如果已经是标准列
    if "text" in df.columns and "label" in df.columns:
        return df[["text", "label"]].copy()

    # 常见候选列
    text_candidates = ["text", "review", "content", "sentence", "comment"]
    label_candidates = ["label", "sentiment", "category", "class"]

    text_col = next((c for c in text_candidates if c in df.columns), None)
    label_col = next((c for c in label_candidates if c in df.columns), None)

    if text_col is not None and label_col is not None:
        out = df[[text_col, label_col]].copy()
        out.columns = ["text", "label"]
        return out

    # 实在找不到就报错
    raise ValueError(f"列名无法识别，当前列名为：{list(df.columns)}")


def backup_file(path: Path) -> None:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    target = BACKUP_DIR / path.name
    shutil.copy2(path, target)


def process_one_file(path: Path) -> None:
    if not path.exists():
        print(f"❌ 文件不存在：{path}")
        return

    print("\n" + "=" * 80)
    print(f"正在处理：{path.name}")

    backup_file(path)

    df = read_csv_safe(path)
    print(f"原始样本数：{len(df)}")
    print(f"原始列名：{list(df.columns)}")

    df = standardize_columns(df)

    is_weibo = "weibo" in path.name.lower()

    # 清洗文本
    df["text"] = df["text"].apply(lambda x: clean_text(x, is_weibo=is_weibo))

    # 统一标签
    df["label"] = df["label"].apply(normalize_label)

    # 删除无效标签
    before_label = len(df)
    df = df[df["label"].notna()].copy()
    print(f"删除无效标签：{before_label - len(df)}")

    df["label"] = df["label"].astype(int)

    # 仅保留 0/1
    before_binary = len(df)
    df = df[df["label"].isin([0, 1])].copy()
    print(f"删除非二分类标签：{before_binary - len(df)}")

    # 删除空文本/无效文本
    before_text = len(df)
    df = df[~df["text"].apply(invalid_text)].copy()
    print(f"删除空文本/无效文本：{before_text - len(df)}")

    # 去重（同一个 split 内按 text 去重）
    before_dup = len(df)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"删除重复文本：{before_dup - len(df)}")

    # 保存前统计
    print(f"清洗后样本数：{len(df)}")
    print("标签分布：")
    print(df["label"].value_counts().sort_index())

    # 覆盖保存回原文件
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"✅ 已覆盖保存：{path}")


def main() -> None:
    print("开始对 data_clean 中的文件进行二次清洗...")
    print(f"原文件备份目录：{BACKUP_DIR}")

    for path in FILES:
        process_one_file(path)

    print("\n全部完成。")
    print("data_clean 中的文件已更新，原文件已自动备份。")


if __name__ == "__main__":
    main()