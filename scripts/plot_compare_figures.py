from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 1. 固定路径：按你现在目录写死
# =========================
PROJECT_DIR = Path(r"D:\Users\30126\Desktop\project")
MODELS_DIR = PROJECT_DIR / "models"
FIG_DIR = PROJECT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# 2. 你最终论文里用的 5 个模型
#    这里顺序就是图上的显示顺序
# =========================
MODEL_KEYS = ["bert", "macbert", "roberta", "bert_bigru", "bert_lora"]
MODEL_NAMES = {
    "bert": "BERT",
    "macbert": "MacBERT",
    "roberta": "RoBERTa-wwm-ext",
    "bert_bigru": "BERT+BiGRU",
    "bert_lora": "BERT+LoRA",
}

# =========================
# 3. 3 个数据集
# =========================
DATASET_KEYS = ["chn", "online", "weibo"]
DATASET_NAMES = {
    "chn": "ChnSentiCorp",
    "online": "OnlineShopping10Cats",
    "weibo": "WeiboSenti100k",
}


# =========================
# 4. 中文显示设置
# =========================
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def get_experiment_dir(model_key: str, dataset_key: str) -> Path:
    if dataset_key == "chn":
        return MODELS_DIR / f"{model_key}_chnsenticorp"
    return MODELS_DIR / f"{model_key}_{dataset_key}"


def read_best_eval_loss(exp_dir: Path) -> float | None:
    log_file = exp_dir / "train_log.csv"
    if not log_file.exists():
        return None

    df = pd.read_csv(log_file, encoding="utf-8-sig")

    if "eval_loss" not in df.columns:
        return None

    eval_losses = pd.to_numeric(df["eval_loss"], errors="coerce").dropna()
    if len(eval_losses) == 0:
        return None

    # 取最小验证损失，更适合做模型对比
    return float(eval_losses.min())


def read_test_f1(exp_dir: Path) -> float | None:
    metrics_file = exp_dir / "test_metrics.json"
    if not metrics_file.exists():
        return None

    with open(metrics_file, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    if "f1" not in metrics:
        return None

    return float(metrics["f1"])


def collect_data():
    loss_data = {dataset: [] for dataset in DATASET_KEYS}
    f1_data = {dataset: [] for dataset in DATASET_KEYS}

    for dataset_key in DATASET_KEYS:
        for model_key in MODEL_KEYS:
            exp_dir = get_experiment_dir(model_key, dataset_key)

            best_eval_loss = read_best_eval_loss(exp_dir)
            test_f1 = read_test_f1(exp_dir)

            loss_data[dataset_key].append(best_eval_loss)
            f1_data[dataset_key].append(test_f1)

    return loss_data, f1_data


def print_result_table(loss_data, f1_data):
    rows = []
    for dataset_key in DATASET_KEYS:
        for model_key in MODEL_KEYS:
            model_idx = MODEL_KEYS.index(model_key)
            rows.append({
                "数据集": DATASET_NAMES[dataset_key],
                "模型": MODEL_NAMES[model_key],
                "最小eval_loss": loss_data[dataset_key][model_idx],
                "测试集F1": f1_data[dataset_key][model_idx],
            })

    df = pd.DataFrame(rows)
    print("\n===== 汇总结果 =====")
    print(df.to_string(index=False))
    df.to_csv(FIG_DIR / "result_summary_for_figures.csv", index=False, encoding="utf-8-sig")


def plot_loss_line(loss_data):
    x = np.arange(len(MODEL_KEYS))
    model_labels = [MODEL_NAMES[m] for m in MODEL_KEYS]

    plt.figure(figsize=(12, 6))

    for dataset_key in DATASET_KEYS:
        y = loss_data[dataset_key]
        plt.plot(x, y, marker="o", linewidth=2, label=DATASET_NAMES[dataset_key])

    plt.xticks(x, model_labels, rotation=20)
    plt.ylabel("Loss")
    plt.xlabel("模型")
    plt.title("不同模型在各数据集上的最小验证损失对比")
    plt.legend()
    plt.tight_layout()

    save_path_png = FIG_DIR / "loss_compare.png"
    save_path_svg = FIG_DIR / "loss_compare.svg"
    plt.savefig(save_path_png, dpi=300, bbox_inches="tight")
    plt.savefig(save_path_svg, bbox_inches="tight")
    plt.close()

    print(f"Loss 图已保存到: {save_path_png}")
    print(f"Loss 图已保存到: {save_path_svg}")


def plot_f1_bar(f1_data):
    x = np.arange(len(DATASET_KEYS))
    width = 0.15

    plt.figure(figsize=(12, 6))

    for i, model_key in enumerate(MODEL_KEYS):
        y = []
        for dataset_key in DATASET_KEYS:
            f1 = f1_data[dataset_key][i]
            if f1 is None:
                y.append(np.nan)
            else:
                y.append(f1 * 100)   # 转成百分比，更像论文里的图

        plt.bar(x + (i - 2) * width, y, width, label=MODEL_NAMES[model_key])

        plt.xticks(x, [DATASET_NAMES[d] for d in DATASET_KEYS], rotation=0)
        plt.ylabel("F1 (%)")
        plt.xlabel("数据集")
        plt.title("不同模型在各数据集上的F1值对比")
        plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
        plt.tight_layout()

    save_path_png = FIG_DIR / "f1_compare.png"
    save_path_svg = FIG_DIR / "f1_compare.svg"
    plt.savefig(save_path_png, dpi=300, bbox_inches="tight")
    plt.savefig(save_path_svg, bbox_inches="tight")
    plt.close()

    print(f"F1 图已保存到: {save_path_png}")
    print(f"F1 图已保存到: {save_path_svg}")


def main():
    loss_data, f1_data = collect_data()
    print_result_table(loss_data, f1_data)
    plot_loss_line(loss_data)
    plot_f1_bar(f1_data)
    print(f"\n全部完成，图片保存在: {FIG_DIR}")


if __name__ == "__main__":
    main()