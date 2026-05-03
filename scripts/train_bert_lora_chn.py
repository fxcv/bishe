from __future__ import annotations

import os
import json
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType


# =========================
# 固定路径：不用改
# =========================
TRAIN_FILE = r"D:\Users\30126\Desktop\project\data_clean\chn_senti_corp_train.csv"
VALID_FILE = r"D:\Users\30126\Desktop\project\data_clean\chn_senti_corp_valid.csv"
TEST_FILE  = r"D:\Users\30126\Desktop\project\data_clean\chn_senti_corp_test.csv"

MODEL_NAME = "bert-base-chinese"
EXPERIMENT_NAME = "bert_lora_chnsenticorp"

BASE_OUTPUT_DIR = Path(r"D:\Users\30126\Desktop\project\models") / EXPERIMENT_NAME
BEST_MODEL_DIR = BASE_OUTPUT_DIR / "best_model"

METRICS_FILE = BASE_OUTPUT_DIR / "test_metrics.json"
RESULT_TXT = BASE_OUTPUT_DIR / "test_result.txt"
PRED_CSV = BASE_OUTPUT_DIR / "test_predictions.csv"
TRAIN_LOG_CSV = BASE_OUTPUT_DIR / "train_log.csv"
CONFUSION_MATRIX_CSV = BASE_OUTPUT_DIR / "confusion_matrix.csv"
RUN_INFO_JSON = BASE_OUTPUT_DIR / "run_info.json"

MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
SEED = 42

# LoRA 参数
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1


def fix_random_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def ensure_dir() -> None:
    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_csv(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在：{file_path}")

    df = pd.read_csv(file_path, encoding="utf-8-sig")
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError(f"{file_path} 必须包含 text 和 label 两列，当前列：{list(df.columns)}")

    df = df[["text", "label"]].copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""].reset_index(drop=True)
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["label"].notna()].copy()
    df["label"] = df["label"].astype(int)
    df = df[df["label"].isin([0, 1])].reset_index(drop=True)
    return df


class SentimentDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: BertTokenizer, max_len: int = 128):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        text = self.df.loc[idx, "text"]
        label = int(self.df.loc[idx, "label"])

        encoded = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {k: v.squeeze(0) for k, v in encoded.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )

    return {
        "accuracy": round(float(acc), 6),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
    }


class LogHistoryCallback(TrainerCallback):
    def __init__(self):
        self.rows = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        row = {"step": state.global_step}
        row.update({k: v for k, v in logs.items()})
        self.rows.append(row)

    def save(self, file_path: Path):
        if not self.rows:
            return
        df = pd.DataFrame(self.rows)
        df.to_csv(file_path, index=False, encoding="utf-8-sig")


def save_confusion_matrix(labels: np.ndarray, preds: np.ndarray, file_path: Path):
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    cm_df = pd.DataFrame(
        cm,
        index=["真实负向(0)", "真实正向(1)"],
        columns=["预测负向(0)", "预测正向(1)"],
    )
    cm_df.to_csv(file_path, encoding="utf-8-sig")


def save_test_result(trainer: Trainer, test_df: pd.DataFrame, tokenizer: BertTokenizer, train_seconds: float):
    test_dataset = SentimentDataset(test_df, tokenizer, MAX_LEN)
    predictions = trainer.predict(test_dataset)
    logits = predictions.predictions
    labels = predictions.label_ids
    preds = np.argmax(logits, axis=1)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )

    metrics = {
        "accuracy": round(float(acc), 6),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
        "train_seconds": round(float(train_seconds), 2),
    }

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    with open(RESULT_TXT, "w", encoding="utf-8") as f:
        f.write("ChnSentiCorp 测试集结果（BERT+LoRA）\n")
        f.write(f"accuracy:      {metrics['accuracy']}\n")
        f.write(f"precision:     {metrics['precision']}\n")
        f.write(f"recall:        {metrics['recall']}\n")
        f.write(f"f1:            {metrics['f1']}\n")
        f.write(f"train_seconds: {metrics['train_seconds']}\n")

    result_df = test_df.copy()
    result_df["pred_label"] = preds
    result_df.to_csv(PRED_CSV, index=False, encoding="utf-8-sig")

    save_confusion_matrix(labels, preds, CONFUSION_MATRIX_CSV)

    run_info = {
        "experiment_name": EXPERIMENT_NAME,
        "dataset": "ChnSentiCorp",
        "model_name": MODEL_NAME,
        "architecture": "BERT+LoRA",
        "max_len": MAX_LEN,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "seed": SEED,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "train_file": TRAIN_FILE,
        "valid_file": VALID_FILE,
        "test_file": TEST_FILE,
        "train_size": len(pd.read_csv(TRAIN_FILE, encoding="utf-8-sig")),
        "valid_size": len(pd.read_csv(VALID_FILE, encoding="utf-8-sig")),
        "test_size": len(pd.read_csv(TEST_FILE, encoding="utf-8-sig")),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "metrics_file": str(METRICS_FILE),
        "result_txt": str(RESULT_TXT),
        "pred_csv": str(PRED_CSV),
        "train_log_csv": str(TRAIN_LOG_CSV),
        "confusion_matrix_csv": str(CONFUSION_MATRIX_CSV),
    }
    with open(RUN_INFO_JSON, "w", encoding="utf-8") as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2)

    print("\n===== 测试集结果 =====")
    print(f"accuracy      : {metrics['accuracy']}")
    print(f"precision     : {metrics['precision']}")
    print(f"recall        : {metrics['recall']}")
    print(f"f1            : {metrics['f1']}")
    print(f"train_seconds : {metrics['train_seconds']}")

    print(f"\n最优模型目录: {BEST_MODEL_DIR}")
    print(f"指标文件: {METRICS_FILE}")
    print(f"结果文本: {RESULT_TXT}")
    print(f"预测明细: {PRED_CSV}")
    print(f"训练日志: {TRAIN_LOG_CSV}")
    print(f"混淆矩阵: {CONFUSION_MATRIX_CSV}")
    print(f"运行信息: {RUN_INFO_JSON}")


def main():
    ensure_dir()
    fix_random_seed(SEED)

    print("===== 读取数据 =====")
    train_df = load_csv(TRAIN_FILE)
    valid_df = load_csv(VALID_FILE)
    test_df = load_csv(TEST_FILE)

    print(f"训练集: {len(train_df)}")
    print(f"验证集: {len(valid_df)}")
    print(f"测试集: {len(test_df)}")

    print("\n训练集标签分布：")
    print(train_df["label"].value_counts().sort_index())
    print("\n验证集标签分布：")
    print(valid_df["label"].value_counts().sort_index())
    print("\n测试集标签分布：")
    print(test_df["label"].value_counts().sort_index())

    print("\n===== 检查设备 =====")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"当前显卡: {torch.cuda.get_device_name(0)}")
    else:
        print("当前使用 CPU 训练")

    print("\n===== 加载 tokenizer 和 model =====")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    base_model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=["query", "value"],
    )

    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    train_dataset = SentimentDataset(train_df, tokenizer, MAX_LEN)
    valid_dataset = SentimentDataset(valid_df, tokenizer, MAX_LEN)

    log_callback = LogHistoryCallback()

    training_args = TrainingArguments(
        output_dir=str(BASE_OUTPUT_DIR),
        do_train=True,
        do_eval=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=250,
        max_grad_norm=1.0,
        save_total_limit=2,
        report_to="none",
        fp16=False,
        bf16=False,
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=2),
            log_callback,
        ],
    )

    print("\n===== 开始训练 =====")
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    train_seconds = end_time - start_time

    log_callback.save(TRAIN_LOG_CSV)

    print("\n===== 保存最优模型 =====")
    trainer.save_model(str(BEST_MODEL_DIR))
    tokenizer.save_pretrained(str(BEST_MODEL_DIR))
    print(f"最优模型已保存到: {BEST_MODEL_DIR}")

    print("\n===== 在测试集上评估 =====")
    save_test_result(trainer, test_df, tokenizer, train_seconds)


if __name__ == "__main__":
    main()