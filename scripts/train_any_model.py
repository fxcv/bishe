from __future__ import annotations

import os
os.environ["TRANSFORMERS_DISABLE_AUTO_CONVERSION"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import json
import time
import math
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BertTokenizer,
    BertModel,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback,
    set_seed,
)
from transformers.modeling_outputs import SequenceClassifierOutput

# 强行关闭 transformers 后台 safetensors auto conversion 线程
try:
    import transformers.safetensors_conversion as _stc

    def _disable_auto_conversion(*args, **kwargs):
        return None

    _stc.auto_conversion = _disable_auto_conversion
except Exception:
    pass


# =========================
# 路径配置：按你现在目录写死
# =========================
PROJECT_DIR = Path(r"D:\Users\30126\Desktop\project")
DATA_DIR = PROJECT_DIR / "data_clean"
MODEL_OUT_DIR = PROJECT_DIR / "models"

DATASETS = {
    "chn": {
        "train": DATA_DIR / "chn_senti_corp_train.csv",
        "valid": DATA_DIR / "chn_senti_corp_valid.csv",
        "test":  DATA_DIR / "chn_senti_corp_test.csv",
        "name": "ChnSentiCorp",
    },
    "online": {
        "train": DATA_DIR / "online_shopping_10_cats_train.csv",
        "valid": DATA_DIR / "online_shopping_10_cats_valid.csv",
        "test":  DATA_DIR / "online_shopping_10_cats_test.csv",
        "name": "OnlineShopping10Cats",
    },
    "weibo": {
        "train": DATA_DIR / "weibo_train.csv",
        "valid": DATA_DIR / "weibo_valid.csv",
        "test":  DATA_DIR / "weibo_test.csv",
        "name": "WeiboSenti100k",
    },
}

SIMPLE_MODELS = {
    "bert": "bert-base-chinese",
    "macbert": "hfl/chinese-macbert-base",
    "roberta": "hfl/chinese-roberta-wwm-ext",
}

BERT_BACKBONE = "bert-base-chinese"

MAX_LEN = 128
EPOCHS = 5
SEED = 42
NUM_LABELS = 2

# 统一主参数
DEFAULT_BATCH_SIZE = 16
DEFAULT_LR = 2e-5
DEFAULT_WEIGHT_DECAY = 0.01

# BERT+BiGRU
BIGRU_HIDDEN_SIZE = 256
BIGRU_DROPOUT = 0.3

# LoRA
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1


def fix_random_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def load_csv(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
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
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int = 128):
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


class BertBiGRUClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        hidden_size: int = 256,
        num_labels: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.bigru = nn.GRU(
            input_size=self.bert.config.hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        **kwargs,
    ):
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = bert_outputs.last_hidden_state
        gru_output, _ = self.bigru(sequence_output)

        # 正向最后一步 + 反向第一步
        last_forward = gru_output[:, -1, :self.bigru.hidden_size]
        first_backward = gru_output[:, 0, self.bigru.hidden_size:]
        final_output = torch.cat([last_forward, first_backward], dim=-1)

        final_output = self.dropout(final_output)
        logits = self.classifier(final_output)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )


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
        pd.DataFrame(self.rows).to_csv(file_path, index=False, encoding="utf-8-sig")


def save_confusion_matrix(labels: np.ndarray, preds: np.ndarray, file_path: Path):
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    cm_df = pd.DataFrame(
        cm,
        index=["真实负向(0)", "真实正向(1)"],
        columns=["预测负向(0)", "预测正向(1)"],
    )
    cm_df.to_csv(file_path, encoding="utf-8-sig")


def prepare_tokenizer_common(tokenizer):
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    return tokenizer


def get_model_and_tokenizer(model_key: str):
    if model_key in SIMPLE_MODELS:
        model_name = SIMPLE_MODELS[model_key]
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False,
        )
        tokenizer = prepare_tokenizer_common(tokenizer)

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            use_safetensors=False,
            trust_remote_code=False,
        )
        return model, tokenizer, model_name

    if model_key == "bert_bigru":
        tokenizer = BertTokenizer.from_pretrained(BERT_BACKBONE)
        tokenizer = prepare_tokenizer_common(tokenizer)

        model = BertBiGRUClassifier(
            model_name=BERT_BACKBONE,
            hidden_size=BIGRU_HIDDEN_SIZE,
            num_labels=NUM_LABELS,
            dropout=BIGRU_DROPOUT,
        )
        return model, tokenizer, BERT_BACKBONE

    if model_key == "bert_lora":
        try:
            from peft import LoraConfig, get_peft_model, TaskType
        except Exception as e:
            raise ImportError("LoRA 需要先安装 peft：pip install peft") from e

        tokenizer = BertTokenizer.from_pretrained(BERT_BACKBONE)
        tokenizer = prepare_tokenizer_common(tokenizer)

        base_model = BertForSequenceClassification.from_pretrained(
            BERT_BACKBONE,
            num_labels=2,
        )

        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            target_modules=["query", "value"],
            modules_to_save=["classifier"],
        )
        model = get_peft_model(base_model, lora_config)
        model.print_trainable_parameters()
        return model, tokenizer, BERT_BACKBONE

    raise ValueError(f"不支持的模型：{model_key}")


def get_hyperparams(model_key: str):
    batch_size = DEFAULT_BATCH_SIZE
    learning_rate = DEFAULT_LR
    weight_decay = DEFAULT_WEIGHT_DECAY

    if model_key == "bert_bigru":
        batch_size = 16
        learning_rate = 2e-5
    elif model_key == "bert_lora":
        batch_size = 16
        learning_rate = 2e-5

    return batch_size, learning_rate, weight_decay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["bert", "macbert", "roberta", "bert_bigru", "bert_lora"])
    parser.add_argument("--dataset", required=True, choices=["chn", "online", "weibo"])
    args = parser.parse_args()

    model_key = args.model
    dataset_key = args.dataset

    dataset_info = DATASETS[dataset_key]
    experiment_name = f"{model_key}_{dataset_key}"

    base_output_dir = MODEL_OUT_DIR / experiment_name
    best_model_dir = base_output_dir / "best_model"

    metrics_file = base_output_dir / "test_metrics.json"
    result_txt = base_output_dir / "test_result.txt"
    pred_csv = base_output_dir / "test_predictions.csv"
    train_log_csv = base_output_dir / "train_log.csv"
    confusion_matrix_csv = base_output_dir / "confusion_matrix.csv"
    run_info_json = base_output_dir / "run_info.json"

    base_output_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir.mkdir(parents=True, exist_ok=True)

    fix_random_seed(SEED)

    print("===== 读取数据 =====")
    train_df = load_csv(dataset_info["train"])
    valid_df = load_csv(dataset_info["valid"])
    test_df = load_csv(dataset_info["test"])

    print(f"数据集: {dataset_info['name']}")
    print(f"模型: {model_key}")
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
    model, tokenizer, model_name = get_model_and_tokenizer(model_key)

    train_dataset = SentimentDataset(train_df, tokenizer, MAX_LEN)
    valid_dataset = SentimentDataset(valid_df, tokenizer, MAX_LEN)

    batch_size, learning_rate, weight_decay = get_hyperparams(model_key)

    steps_per_epoch = math.ceil(len(train_dataset) / batch_size)
    total_steps = steps_per_epoch * EPOCHS
    warmup_steps = max(50, int(total_steps * 0.1))

    log_callback = LogHistoryCallback()

    training_args = TrainingArguments(
        output_dir=str(base_output_dir),
        do_train=True,
        do_eval=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=EPOCHS,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        max_grad_norm=1.0,
        save_total_limit=2,
        report_to="none",
        fp16=False,
        bf16=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
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

    log_callback.save(train_log_csv)

    print("\n===== 保存最优模型 =====")
    trainer.save_model(str(best_model_dir))
    tokenizer.save_pretrained(str(best_model_dir))
    print(f"最优模型已保存到: {best_model_dir}")

    print("\n===== 在测试集上评估 =====")
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

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    with open(result_txt, "w", encoding="utf-8") as f:
        f.write(f"{dataset_info['name']} 测试集结果（{model_key}）\n")
        f.write(f"accuracy:      {metrics['accuracy']}\n")
        f.write(f"precision:     {metrics['precision']}\n")
        f.write(f"recall:        {metrics['recall']}\n")
        f.write(f"f1:            {metrics['f1']}\n")
        f.write(f"train_seconds: {metrics['train_seconds']}\n")

    result_df = test_df.copy()
    result_df["pred_label"] = preds
    result_df.to_csv(pred_csv, index=False, encoding="utf-8-sig")

    save_confusion_matrix(labels, preds, confusion_matrix_csv)

    run_info = {
        "experiment_name": experiment_name,
        "dataset": dataset_info["name"],
        "model_key": model_key,
        "model_name": model_name,
        "max_len": MAX_LEN,
        "batch_size": batch_size,
        "epochs": EPOCHS,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "warmup_steps": warmup_steps,
        "seed": SEED,
        "train_file": str(dataset_info["train"]),
        "valid_file": str(dataset_info["valid"]),
        "test_file": str(dataset_info["test"]),
        "train_size": len(train_df),
        "valid_size": len(valid_df),
        "test_size": len(test_df),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "metrics_file": str(metrics_file),
        "result_txt": str(result_txt),
        "pred_csv": str(pred_csv),
        "train_log_csv": str(train_log_csv),
        "confusion_matrix_csv": str(confusion_matrix_csv),
    }
    with open(run_info_json, "w", encoding="utf-8") as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2)

    print("\n===== 测试集结果 =====")
    print(f"accuracy      : {metrics['accuracy']}")
    print(f"precision     : {metrics['precision']}")
    print(f"recall        : {metrics['recall']}")
    print(f"f1            : {metrics['f1']}")
    print(f"train_seconds : {metrics['train_seconds']}")

    print(f"\n最优模型目录: {best_model_dir}")
    print(f"指标文件: {metrics_file}")
    print(f"结果文本: {result_txt}")
    print(f"预测明细: {pred_csv}")
    print(f"训练日志: {train_log_csv}")
    print(f"混淆矩阵: {confusion_matrix_csv}")
    print(f"运行信息: {run_info_json}")


if __name__ == "__main__":
    main()