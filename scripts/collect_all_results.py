from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


# =========================
# 固定路径：按你现在目录写死
# =========================
PROJECT_DIR = Path(r"D:\Users\30126\Desktop\project")
MODELS_DIR = PROJECT_DIR / "models"
OUTPUT_FILE = PROJECT_DIR / "all_experiment_results.xlsx"


def read_json_safe(path: Path):
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def read_csv_safe(path: Path):
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        try:
            return pd.read_csv(path, encoding="utf-8")
        except Exception:
            return None


def parse_dataset_and_model(folder_name: str):
    # 目录命名一般像：
    # bert_chn
    # macbert_online
    # bert_bigru_weibo
    # bert_lora_chn
    # roberta_chn
    parts = folder_name.split("_")
    if len(parts) < 2:
        return folder_name, ""

    dataset_key = parts[-1]
    model_key = "_".join(parts[:-1])
    return model_key, dataset_key


def collect_metrics():
    rows = []

    for exp_dir in sorted(MODELS_DIR.iterdir()):
        if not exp_dir.is_dir():
            continue

        folder_name = exp_dir.name
        model_key, dataset_key = parse_dataset_and_model(folder_name)

        metrics_file = exp_dir / "test_metrics.json"
        run_info_file = exp_dir / "run_info.json"

        metrics = read_json_safe(metrics_file) or {}
        run_info = read_json_safe(run_info_file) or {}

        row = {
            "experiment_dir": folder_name,
            "model_key": model_key,
            "dataset_key": dataset_key,
            "dataset_name": run_info.get("dataset", ""),
            "model_name": run_info.get("model_name", ""),
            "accuracy": metrics.get("accuracy"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1": metrics.get("f1"),
            "train_seconds": metrics.get("train_seconds"),
            "batch_size": run_info.get("batch_size"),
            "epochs": run_info.get("epochs"),
            "learning_rate": run_info.get("learning_rate"),
            "weight_decay": run_info.get("weight_decay"),
            "warmup_steps": run_info.get("warmup_steps"),
            "max_len": run_info.get("max_len"),
            "device": run_info.get("device"),
            "gpu_name": run_info.get("gpu_name"),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def collect_loss_summary():
    rows = []

    for exp_dir in sorted(MODELS_DIR.iterdir()):
        if not exp_dir.is_dir():
            continue

        folder_name = exp_dir.name
        model_key, dataset_key = parse_dataset_and_model(folder_name)

        train_log_file = exp_dir / "train_log.csv"
        run_info_file = exp_dir / "run_info.json"

        log_df = read_csv_safe(train_log_file)
        run_info = read_json_safe(run_info_file) or {}

        if log_df is None:
            rows.append({
                "experiment_dir": folder_name,
                "model_key": model_key,
                "dataset_key": dataset_key,
                "dataset_name": run_info.get("dataset", ""),
                "model_name": run_info.get("model_name", ""),
                "min_eval_loss": None,
                "last_eval_loss": None,
                "min_train_loss": None,
            })
            continue

        # 取数值列
        min_eval_loss = None
        last_eval_loss = None
        min_train_loss = None

        if "eval_loss" in log_df.columns:
            eval_loss_series = pd.to_numeric(log_df["eval_loss"], errors="coerce").dropna()
            if len(eval_loss_series) > 0:
                min_eval_loss = float(eval_loss_series.min())
                last_eval_loss = float(eval_loss_series.iloc[-1])

        if "loss" in log_df.columns:
            train_loss_series = pd.to_numeric(log_df["loss"], errors="coerce").dropna()
            if len(train_loss_series) > 0:
                min_train_loss = float(train_loss_series.min())

        rows.append({
            "experiment_dir": folder_name,
            "model_key": model_key,
            "dataset_key": dataset_key,
            "dataset_name": run_info.get("dataset", ""),
            "model_name": run_info.get("model_name", ""),
            "min_eval_loss": min_eval_loss,
            "last_eval_loss": last_eval_loss,
            "min_train_loss": min_train_loss,
        })

    return pd.DataFrame(rows)


def collect_confusion_summary():
    rows = []

    for exp_dir in sorted(MODELS_DIR.iterdir()):
        if not exp_dir.is_dir():
            continue

        folder_name = exp_dir.name
        model_key, dataset_key = parse_dataset_and_model(folder_name)

        cm_file = exp_dir / "confusion_matrix.csv"
        run_info_file = exp_dir / "run_info.json"

        cm_df = read_csv_safe(cm_file)
        run_info = read_json_safe(run_info_file) or {}

        tn = fp = fn = tp = None

        if cm_df is not None and cm_df.shape[0] >= 2 and cm_df.shape[1] >= 3:
            # 第一列一般是索引列，后两列是真实值
            try:
                tn = cm_df.iloc[0, 1]
                fp = cm_df.iloc[0, 2]
                fn = cm_df.iloc[1, 1]
                tp = cm_df.iloc[1, 2]
            except Exception:
                pass

        rows.append({
            "experiment_dir": folder_name,
            "model_key": model_key,
            "dataset_key": dataset_key,
            "dataset_name": run_info.get("dataset", ""),
            "model_name": run_info.get("model_name", ""),
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
        })

    return pd.DataFrame(rows)


def collect_raw_train_logs():
    rows = []

    for exp_dir in sorted(MODELS_DIR.iterdir()):
        if not exp_dir.is_dir():
            continue

        folder_name = exp_dir.name
        model_key, dataset_key = parse_dataset_and_model(folder_name)

        train_log_file = exp_dir / "train_log.csv"
        run_info_file = exp_dir / "run_info.json"

        log_df = read_csv_safe(train_log_file)
        run_info = read_json_safe(run_info_file) or {}

        if log_df is None:
            continue

        log_df = log_df.copy()
        log_df.insert(0, "experiment_dir", folder_name)
        log_df.insert(1, "model_key", model_key)
        log_df.insert(2, "dataset_key", dataset_key)
        log_df.insert(3, "dataset_name", run_info.get("dataset", ""))
        log_df.insert(4, "model_name", run_info.get("model_name", ""))

        rows.append(log_df)

    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame()


def collect_prediction_examples(max_rows_per_exp: int = 20):
    rows = []

    for exp_dir in sorted(MODELS_DIR.iterdir()):
        if not exp_dir.is_dir():
            continue

        folder_name = exp_dir.name
        model_key, dataset_key = parse_dataset_and_model(folder_name)

        pred_file = exp_dir / "test_predictions.csv"
        run_info_file = exp_dir / "run_info.json"

        pred_df = read_csv_safe(pred_file)
        run_info = read_json_safe(run_info_file) or {}

        if pred_df is None or len(pred_df) == 0:
            continue

        temp = pred_df.copy().head(max_rows_per_exp)
        temp.insert(0, "experiment_dir", folder_name)
        temp.insert(1, "model_key", model_key)
        temp.insert(2, "dataset_key", dataset_key)
        temp.insert(3, "dataset_name", run_info.get("dataset", ""))
        temp.insert(4, "model_name", run_info.get("model_name", ""))

        rows.append(temp)

    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame()


def main():
    if not MODELS_DIR.exists():
        raise FileNotFoundError(f"找不到目录：{MODELS_DIR}")

    metrics_df = collect_metrics()
    loss_df = collect_loss_summary()
    cm_df = collect_confusion_summary()
    logs_df = collect_raw_train_logs()
    pred_df = collect_prediction_examples(max_rows_per_exp=20)

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        metrics_df.to_excel(writer, sheet_name="metrics_summary", index=False)
        loss_df.to_excel(writer, sheet_name="loss_summary", index=False)
        cm_df.to_excel(writer, sheet_name="confusion_summary", index=False)
        logs_df.to_excel(writer, sheet_name="train_logs", index=False)
        pred_df.to_excel(writer, sheet_name="prediction_examples", index=False)

    print("===== 汇总完成 =====")
    print(f"输出文件：{OUTPUT_FILE}")
    print("包含工作表：")
    print("- metrics_summary")
    print("- loss_summary")
    print("- confusion_summary")
    print("- train_logs")
    print("- prediction_examples")


if __name__ == "__main__":
    main()