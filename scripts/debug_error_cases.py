from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(r"D:\Users\30126\Desktop\project")
MODELS_DIR = PROJECT_DIR / "models"

TARGET_DIRS = [
    "bert_chnsenticorp",
    "bert_bigru_online",
    "roberta_weibo",
]

def try_read_csv(path: Path):
    for enc in ["utf-8-sig", "utf-8", "gbk"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    raise ValueError(f"无法读取文件: {path}")

for folder in TARGET_DIRS:
    pred_file = MODELS_DIR / folder / "test_predictions.csv"
    print("\n" + "=" * 80)
    print("目录:", folder)
    print("文件:", pred_file)

    if not pred_file.exists():
        print("文件不存在")
        continue

    df = try_read_csv(pred_file)

    print("总行数:", len(df))
    print("列名:", list(df.columns))

    # 尝试显示前5行
    print("\n前5行:")
    print(df.head())

    # 自动猜测列
    text_cols = [c for c in df.columns if c.lower() in ["text", "sentence", "content", "review", "comment"]]
    true_cols = [c for c in df.columns if c.lower() in ["label", "true_label", "gold_label", "y_true"]]
    pred_cols = [c for c in df.columns if c.lower() in ["pred_label", "prediction", "pred", "y_pred", "pred_label_id"]]

    print("\n识别到的文本列:", text_cols)
    print("识别到的真实标签列:", true_cols)
    print("识别到的预测标签列:", pred_cols)

    if true_cols and pred_cols:
        true_col = true_cols[0]
        pred_col = pred_cols[0]

        mismatch = df[df[true_col].astype(str) != df[pred_col].astype(str)]
        print("识别到的错误样本数:", len(mismatch))

        if len(mismatch) > 0:
            print("\n前5条错误样本:")
            print(mismatch.head())
    else:
        print("没有成功识别真实标签列或预测标签列。")
        