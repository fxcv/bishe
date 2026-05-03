from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(r"D:\Users\30126\Desktop\project")
MODELS_DIR = PROJECT_DIR / "models"
OUTPUT_FILE = PROJECT_DIR / "analysis_data_for_thesis.xlsx"


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
    for enc in ["utf-8-sig", "utf-8", "gbk"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return None


def parse_folder(folder_name: str):
    parts = folder_name.split("_")
    if len(parts) < 2:
        return folder_name, ""
    dataset_key = parts[-1]
    model_key = "_".join(parts[:-1])
    return model_key, dataset_key


def collect_metrics_summary():
    rows = []

    for exp_dir in sorted(MODELS_DIR.iterdir()):
        if not exp_dir.is_dir():
            continue

        folder_name = exp_dir.name
        model_key, dataset_key = parse_folder(folder_name)

        metrics = read_json_safe(exp_dir / "test_metrics.json") or {}
        run_info = read_json_safe(exp_dir / "run_info.json") or {}

        rows.append({
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
        })

    return pd.DataFrame(rows)


def collect_loss_summary():
    rows = []

    for exp_dir in sorted(MODELS_DIR.iterdir()):
        if not exp_dir.is_dir():
            continue

        folder_name = exp_dir.name
        model_key, dataset_key = parse_folder(folder_name)

        log_df = read_csv_safe(exp_dir / "train_log.csv")
        run_info = read_json_safe(exp_dir / "run_info.json") or {}

        min_eval_loss = None
        last_eval_loss = None
        min_train_loss = None
        last_eval_f1 = None

        if log_df is not None:
            if "eval_loss" in log_df.columns:
                s = pd.to_numeric(log_df["eval_loss"], errors="coerce").dropna()
                if len(s) > 0:
                    min_eval_loss = float(s.min())
                    last_eval_loss = float(s.iloc[-1])

            if "loss" in log_df.columns:
                s = pd.to_numeric(log_df["loss"], errors="coerce").dropna()
                if len(s) > 0:
                    min_train_loss = float(s.min())

            if "eval_f1" in log_df.columns:
                s = pd.to_numeric(log_df["eval_f1"], errors="coerce").dropna()
                if len(s) > 0:
                    last_eval_f1 = float(s.iloc[-1])

        rows.append({
            "experiment_dir": folder_name,
            "model_key": model_key,
            "dataset_key": dataset_key,
            "dataset_name": run_info.get("dataset", ""),
            "model_name": run_info.get("model_name", ""),
            "min_eval_loss": min_eval_loss,
            "last_eval_loss": last_eval_loss,
            "min_train_loss": min_train_loss,
            "last_eval_f1": last_eval_f1,
        })

    return pd.DataFrame(rows)


def collect_confusion_summary():
    rows = []

    for exp_dir in sorted(MODELS_DIR.iterdir()):
        if not exp_dir.is_dir():
            continue

        folder_name = exp_dir.name
        model_key, dataset_key = parse_folder(folder_name)

        cm_df = read_csv_safe(exp_dir / "confusion_matrix.csv")
        run_info = read_json_safe(exp_dir / "run_info.json") or {}

        tn = fp = fn = tp = None

        if cm_df is not None and cm_df.shape[0] >= 2 and cm_df.shape[1] >= 3:
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


def collect_last_epochs_logs(max_rows_per_exp: int = 10):
    rows = []

    for exp_dir in sorted(MODELS_DIR.iterdir()):
        if not exp_dir.is_dir():
            continue

        folder_name = exp_dir.name
        model_key, dataset_key = parse_folder(folder_name)

        log_df = read_csv_safe(exp_dir / "train_log.csv")
        run_info = read_json_safe(exp_dir / "run_info.json") or {}

        if log_df is None or len(log_df) == 0:
            continue

        temp = log_df.copy().tail(max_rows_per_exp)
        temp.insert(0, "experiment_dir", folder_name)
        temp.insert(1, "model_key", model_key)
        temp.insert(2, "dataset_key", dataset_key)
        temp.insert(3, "dataset_name", run_info.get("dataset", ""))
        temp.insert(4, "model_name", run_info.get("model_name", ""))

        rows.append(temp)

    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame()


def collect_prediction_examples(max_correct: int = 5, max_wrong: int = 5):
    rows = []

    for exp_dir in sorted(MODELS_DIR.iterdir()):
        if not exp_dir.is_dir():
            continue

        folder_name = exp_dir.name
        model_key, dataset_key = parse_folder(folder_name)

        pred_df = read_csv_safe(exp_dir / "test_predictions.csv")
        run_info = read_json_safe(exp_dir / "run_info.json") or {}

        if pred_df is None or len(pred_df) == 0:
            continue

        if "label" in pred_df.columns and "pred_label" in pred_df.columns:
            correct_df = pred_df[pred_df["label"] == pred_df["pred_label"]].head(max_correct).copy()
            wrong_df = pred_df[pred_df["label"] != pred_df["pred_label"]].head(max_wrong).copy()
            sample_df = pd.concat([correct_df, wrong_df], ignore_index=True)
        else:
            sample_df = pred_df.head(max_correct + max_wrong).copy()

        sample_df.insert(0, "experiment_dir", folder_name)
        sample_df.insert(1, "model_key", model_key)
        sample_df.insert(2, "dataset_key", dataset_key)
        sample_df.insert(3, "dataset_name", run_info.get("dataset", ""))
        sample_df.insert(4, "model_name", run_info.get("model_name", ""))

        rows.append(sample_df)

    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame()


def main():
    if not MODELS_DIR.exists():
        raise FileNotFoundError(f"找不到目录：{MODELS_DIR}")

    metrics_df = collect_metrics_summary()
    loss_df = collect_loss_summary()
    confusion_df = collect_confusion_summary()
    logs_df = collect_last_epochs_logs(max_rows_per_exp=10)
    pred_df = collect_prediction_examples(max_correct=5, max_wrong=5)

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        metrics_df.to_excel(writer, sheet_name="metrics_summary", index=False)
        loss_df.to_excel(writer, sheet_name="loss_summary", index=False)
        confusion_df.to_excel(writer, sheet_name="confusion_summary", index=False)
        logs_df.to_excel(writer, sheet_name="last_epoch_logs", index=False)
        pred_df.to_excel(writer, sheet_name="prediction_examples", index=False)

    print("===== 汇总完成 =====")
    print(f"输出文件：{OUTPUT_FILE}")
    print("工作表：")
    print("- metrics_summary")
    print("- loss_summary")
    print("- confusion_summary")
    print("- last_epoch_logs")
    print("- prediction_examples")


if __name__ == "__main__":
    main()