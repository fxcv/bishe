from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

PROJECT_DIR = Path(r"D:\Users\30126\Desktop\project")
MODELS_DIR = PROJECT_DIR / "models"
SAVE_DIR = MODELS_DIR / "base_bert_base_chinese"

print("准备下载并保存本地底座模型到：", SAVE_DIR)

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese", use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=2)

SAVE_DIR.mkdir(parents=True, exist_ok=True)
tokenizer.save_pretrained(str(SAVE_DIR))
model.save_pretrained(str(SAVE_DIR))

print("完成。")
print("本地底座模型目录：", SAVE_DIR)