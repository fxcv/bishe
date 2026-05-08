import os
os.environ["TRANSFORMERS_DISABLE_AUTO_CONVERSION"] = "1"

import re
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import altair as alt
from huggingface_hub import snapshot_download
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
)
from safetensors.torch import load_file as safe_load_file

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False


# =========================
# 基础配置
# =========================
PROJECT_DIR = Path(__file__).resolve().parent
MODELS_DIR = Path(os.getenv("SENTIMENT_MODELS_DIR", str(PROJECT_DIR / "models"))).expanduser()

# 离线底座模型目录
LOCAL_BASE_BERT_DIR = MODELS_DIR / "base_bert_base_chinese"
PUBLIC_BASE_BERT_MODEL = "bert-base-chinese"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 128

st.set_page_config(
    page_title="中文文本情感分类系统",
    layout="wide",
)


# =========================
# 模型目录
# 这里已经全部切成 weibo 数据集训练出的模型
# 如果你的真实文件夹名不一样，只改下面这 5 行
# =========================
MODEL_SPECS: Dict[str, Dict[str, str]] = {
    "BERT": {
        "local_subdir": "bert_weibo/best_model",
        "repo_secret": "bert",
        "repo_env": "HF_REPO_BERT",
    },
    "MacBERT": {
        "local_subdir": "macbert_weibo/best_model",
        "repo_secret": "macbert",
        "repo_env": "HF_REPO_MACBERT",
    },
    "RoBERTa-wwm-ext": {
        "local_subdir": "roberta_weibo/best_model",
        "repo_secret": "roberta",
        "repo_env": "HF_REPO_ROBERTA",
    },
    "BERT+BiGRU": {
        "local_subdir": "bert_bigru_weibo/best_model",
        "repo_secret": "bert_bigru",
        "repo_env": "HF_REPO_BERT_BIGRU",
    },
    "BERT+LoRA": {
        "local_subdir": "bert_lora_weibo/best_model",
        "repo_secret": "bert_lora",
        "repo_env": "HF_REPO_BERT_LORA",
    },
}

MODEL_PATHS: Dict[str, Path] = {
    model_name: MODELS_DIR / spec["local_subdir"]
    for model_name, spec in MODEL_SPECS.items()
}

# =========================
# 随机示例
# =========================
RANDOM_EXAMPLES = [
    "这家店的服务很细致，菜品新鲜，整体体验非常满意。",
    "房间隔音太差，晚上一直被吵醒，入住体验不太好。",
    "物流速度比预期快，包装也完整，没有出现破损。",
    "客服回复很慢，问题拖了两天还没有解决。",
    "手机屏幕清晰，续航也不错，用起来很顺手。",
    "衣服色差明显，面料也偏硬，和页面描述不太一致。",
    "电影节奏紧凑，演员表现自然，看完之后很有共鸣。",
    "这次外卖送到时已经凉了，味道也比较一般。",
    "价格实惠，质量稳定，作为日常用品很合适。",
    "排队时间太长，现场安排混乱，体验低于预期。",
    "课程讲解清楚，例子贴近实际，对学习很有帮助。",
    "软件更新后经常闪退，希望后续能尽快修复。",
]

if "single_text" not in st.session_state:
    st.session_state.single_text = ""

if "batch_result_df" not in st.session_state:
    st.session_state.batch_result_df = None


# =========================
# 工具函数
# =========================
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = text.replace("\u3000", " ")
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@\S+", "", text)
    text = re.sub(r"#([^#]+)#", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def label_to_text(label_id: int) -> str:
    return "负向" if int(label_id) == 0 else "正向"


def fill_random_example():
    st.session_state.single_text = random.choice(RANDOM_EXAMPLES)


MODEL_DOWNLOAD_PATTERNS = [
    "*.json",
    "*.safetensors",
    "*.bin",
    "*.pt",
    "*.pth",
    "*.model",
    "*.txt",
]


def get_streamlit_secret(key: str, default=None):
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


def get_secret_mapping(section: str) -> Mapping[str, str]:
    value = get_streamlit_secret(section, {})
    return value if isinstance(value, Mapping) else {}


def get_hf_token() -> Optional[str]:
    token = os.getenv("HF_TOKEN") or get_streamlit_secret("HF_TOKEN", "")
    return str(token) if token else None


def get_hf_repo_id(secret_key: str, env_key: str) -> Optional[str]:
    repo_id = os.getenv(env_key)
    if not repo_id:
        repo_id = get_secret_mapping("model_repos").get(secret_key)
    return str(repo_id).strip() if repo_id else None


@st.cache_resource(show_spinner=False)
def download_model_snapshot(repo_id: str) -> Path:
    return Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            token=get_hf_token(),
            allow_patterns=MODEL_DOWNLOAD_PATTERNS,
        )
    )


def resolve_model_dir(model_name: str) -> Path:
    local_path = MODEL_PATHS[model_name]
    if local_path.exists():
        return local_path

    spec = MODEL_SPECS[model_name]
    repo_id = get_hf_repo_id(spec["repo_secret"], spec["repo_env"])
    if repo_id:
        return download_model_snapshot(repo_id)

    raise FileNotFoundError(
        f"模型目录不存在：{local_path}\n"
        f"云端部署时，请在 Streamlit Secrets 的 [model_repos] 中配置 "
        f"{spec['repo_secret']}，或设置环境变量 {spec['repo_env']}。"
    )


def resolve_base_bert_source() -> Tuple[str, bool]:
    if LOCAL_BASE_BERT_DIR.exists():
        return str(LOCAL_BASE_BERT_DIR), True

    repo_id = get_hf_repo_id("base_bert", "HF_REPO_BASE_BERT")
    if repo_id:
        return str(download_model_snapshot(repo_id)), True

    return PUBLIC_BASE_BERT_MODEL, False


def get_available_model_names():
    available = []
    for model_name, spec in MODEL_SPECS.items():
        has_local_model = MODEL_PATHS[model_name].exists()
        has_remote_model = get_hf_repo_id(spec["repo_secret"], spec["repo_env"])
        if has_local_model or has_remote_model:
            available.append(model_name)
    return available or list(MODEL_SPECS.keys())


def find_weight_file(model_dir: Path) -> Path:
    candidates = [
        "model.safetensors",
        "pytorch_model.bin",
        "model.bin",
        "best_model.bin",
        "adapter_model.bin",
        "model.pt",
        "best_model.pt",
        "model.pth",
        "best_model.pth",
    ]
    for name in candidates:
        p = model_dir / name
        if p.exists():
            return p

    all_candidates = []
    for pattern in ["*.safetensors", "*.bin", "*.pt", "*.pth"]:
        all_candidates.extend(model_dir.glob(pattern))

    bad_keywords = [
        "tokenizer", "optimizer", "scheduler", "trainer",
        "training_args", "vocab", "special_tokens"
    ]

    filtered = []
    for p in all_candidates:
        lower_name = p.name.lower()
        if any(k in lower_name for k in bad_keywords):
            continue
        filtered.append(p)

    if filtered:
        filtered = sorted(filtered, key=lambda x: x.stat().st_size, reverse=True)
        return filtered[0]

    raise FileNotFoundError(f"在 {model_dir} 下找不到可用权重文件")


# =========================
# BERT+BiGRU 自定义模型
# =========================
class BertBiGRUClassifier(nn.Module):
    def __init__(
        self,
        base_model_source: str,
        base_local_files_only: bool,
        num_labels: int = 2,
        hidden_size: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(
            base_model_source,
            local_files_only=base_local_files_only,
        )
        bert_hidden = self.bert.config.hidden_size

        self.bigru = nn.GRU(
            input_size=bert_hidden,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_labels)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        bert_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            bert_inputs["token_type_ids"] = token_type_ids

        outputs = self.bert(**bert_inputs)
        sequence_output = outputs.last_hidden_state

        gru_output, _ = self.bigru(sequence_output)

        hidden_size = gru_output.size(-1) // 2
        last_forward = gru_output[:, -1, :hidden_size]
        first_backward = gru_output[:, 0, hidden_size:]
        final_output = torch.cat([last_forward, first_backward], dim=-1)

        final_output = self.dropout(final_output)
        logits = self.classifier(final_output)
        return logits


# =========================
# 模型加载
# =========================
def load_standard_model(model_path: Path):
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        use_fast=False,
        local_files_only=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path),
        local_files_only=True,
    )
    model.to(DEVICE)
    model.eval()
    return tokenizer, model, "standard"


def load_bigru_model(model_path: Path):
    base_source, base_local_files_only = resolve_base_bert_source()

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            use_fast=False,
            local_files_only=True,
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            base_source,
            use_fast=False,
            local_files_only=base_local_files_only,
        )

    model = BertBiGRUClassifier(
        base_model_source=base_source,
        base_local_files_only=base_local_files_only,
        num_labels=2,
        hidden_size=256,
        dropout=0.3,
    )

    weight_file = find_weight_file(model_path)

    if weight_file.suffix == ".safetensors":
        state_dict = safe_load_file(str(weight_file))
    elif weight_file.suffix in [".bin", ".pt", ".pth"]:
        state_dict = torch.load(weight_file, map_location="cpu")
    else:
        raise ValueError(f"BERT+BiGRU 当前暂不支持文件格式：{weight_file.name}")

    if isinstance(state_dict, dict):
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model, "bigru"


def load_lora_model(model_path: Path):
    base_source, base_local_files_only = resolve_base_bert_source()

    if not PEFT_AVAILABLE:
        raise RuntimeError("当前环境未安装 peft，无法加载 BERT+LoRA。")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            use_fast=False,
            local_files_only=True,
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            base_source,
            use_fast=False,
            local_files_only=base_local_files_only,
        )

    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_source,
        num_labels=2,
        local_files_only=base_local_files_only,
    )

    model = PeftModel.from_pretrained(
        base_model,
        str(model_path),
        local_files_only=True,
    )

    model.to(DEVICE)
    model.eval()
    return tokenizer, model, "standard"


@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer(model_name: str):
    model_path = resolve_model_dir(model_name)

    if model_name == "BERT+BiGRU":
        return load_bigru_model(model_path)
    if model_name == "BERT+LoRA":
        return load_lora_model(model_path)
    return load_standard_model(model_path)


# =========================
# 预测
# =========================
def predict_single(text: str, model_name: str) -> Tuple[str, float]:
    cleaned_text = clean_text(text)
    if not cleaned_text:
        return "", 0.0

    tokenizer, model, model_type = load_model_and_tokenizer(model_name)

    inputs = tokenizer(
        cleaned_text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        if model_type == "standard":
            outputs = model(**inputs)
            logits = outputs.logits
        else:
            logits = model(**inputs)

        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        pred_id = int(probs.argmax())
        confidence = float(probs[pred_id])

    return label_to_text(pred_id), confidence


def predict_batch_with_progress(
    df: pd.DataFrame,
    text_col: str,
    model_name: str,
    progress_bar,
    status_text
) -> pd.DataFrame:
    tokenizer, model, model_type = load_model_and_tokenizer(model_name)

    results = []
    texts = df[text_col].fillna("").astype(str).tolist()
    total = len(texts)
    batch_size = 32

    for i in range(0, total, batch_size):
        batch_texts = texts[i:i + batch_size]
        cleaned_batch = [clean_text(t) for t in batch_texts]

        inputs = tokenizer(
            cleaned_batch,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            if model_type == "standard":
                outputs = model(**inputs)
                logits = outputs.logits
            else:
                logits = model(**inputs)

            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            pred_ids = probs.argmax(axis=1)

        for raw_text, pred_id, prob in zip(batch_texts, pred_ids, probs):
            pred_id = int(pred_id)
            results.append({
                "文本": raw_text,
                "预测标签": label_to_text(pred_id),
                "置信度": round(float(prob[pred_id]), 4),
            })

        progress = min((i + batch_size) / total, 1.0)
        progress_bar.progress(progress)
        status_text.text(f"正在预测：{min(i + batch_size, total)}/{total}")

    status_text.text("批量预测完成")
    return pd.DataFrame(results)


# =========================
# 紧凑统计图
# =========================
def draw_single_stat_chart(result_df: pd.DataFrame):
    counts = result_df["预测标签"].value_counts()
    pos_count = int(counts.get("正向", 0))
    neg_count = int(counts.get("负向", 0))
    total = pos_count + neg_count

    if total == 0:
        chart_df = pd.DataFrame({
            "情感类别": ["正向", "负向"],
            "数量": [0, 0],
            "百分比": [0.0, 0.0],
            "标签": ["0 (0.0%)", "0 (0.0%)"],
        })
    else:
        pos_ratio = pos_count / total * 100
        neg_ratio = neg_count / total * 100
        chart_df = pd.DataFrame({
            "情感类别": ["正向", "负向"],
            "数量": [pos_count, neg_count],
            "百分比": [pos_ratio, neg_ratio],
            "标签": [
                f"{pos_count} ({pos_ratio:.1f}%)",
                f"{neg_count} ({neg_ratio:.1f}%)",
            ],
        })

    base = alt.Chart(chart_df).encode(
        x=alt.X(
            "情感类别:N",
            title="情感类别",
            axis=alt.Axis(labelAngle=0, labelFontSize=13, titleFontSize=14),
            scale=alt.Scale(paddingInner=0.75, paddingOuter=0.45),
        ),
        y=alt.Y(
            "百分比:Q",
            title="占比（%）",
            scale=alt.Scale(domain=[0, 100]),
            axis=alt.Axis(labelFontSize=12, titleFontSize=14, tickCount=6),
        ),
    )

    bars = base.mark_bar(
        color="#9E9E9E",
        stroke="black",
        strokeWidth=1.2,
        cornerRadiusTopLeft=5,
        cornerRadiusTopRight=5,
        size=70,
    )

    text = base.mark_text(
        dy=-10,
        color="black",
        fontSize=13,
    ).encode(
        text="标签:N"
    )

    final_chart = (bars + text).properties(
        width=360,
        height=280,
    )

    left, center, right = st.columns([1.4, 2.2, 1.4])
    with center:
        st.altair_chart(final_chart, use_container_width=False)


# =========================
# 页面
# =========================
st.title("中文文本情感分类系统")

st.sidebar.header("系统设置")
model_options = get_available_model_names()
selected_model = st.sidebar.selectbox(
    "选择模型",
    model_options,
    index=0,
)

try:
    with st.spinner(f"正在加载 {selected_model} 模型..."):
        load_model_and_tokenizer(selected_model)
    st.sidebar.success(f"{selected_model} 模型加载成功")
except Exception as e:
    st.sidebar.error(f"模型加载失败：{e}")

st.sidebar.markdown("---")
st.sidebar.subheader("使用说明")
st.sidebar.write("1. 选择模型")
st.sidebar.write("2. 输入单条文本或上传 CSV 文件")
st.sidebar.write("3. 查看预测结果和统计图")

tab1, tab2 = st.tabs(["单文本预测", "批量预测"])


with tab1:
    st.subheader("单文本情感预测")

    st.button("随机示例", on_click=fill_random_example)

    st.text_area(
        "请输入中文文本",
        key="single_text",
        height=180,
    )

    if st.button("开始文本预测", type="primary", use_container_width=True):
        if not st.session_state.single_text.strip():
            st.warning("请输入文本后再预测。")
        else:
            try:
                pred_label, confidence = predict_single(
                    st.session_state.single_text,
                    selected_model,
                )

                st.markdown("### 预测结果")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("当前模型", selected_model)
                with col2:
                    st.metric("预测类别", pred_label)
                with col3:
                    st.metric("置信度", f"{confidence:.4f}")

            except Exception as e:
                st.error(f"预测失败：{e}")


with tab2:
    st.subheader("批量文本预测")
    st.write("请上传包含文本列的 CSV 文件。")

    uploaded_file = st.file_uploader(
        "上传 CSV 文件",
        type=["csv"],
        accept_multiple_files=False,
    )

    if uploaded_file is not None:
        try:
            df = None
            for enc in ["utf-8-sig", "utf-8", "gbk"]:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding=enc)
                    break
                except Exception:
                    continue

            if df is None:
                st.error("文件读取失败，请检查编码格式。")
            else:
                st.success("文件读取成功")
                st.dataframe(df.head(), use_container_width=True)

                possible_text_cols = [c for c in df.columns if df[c].dtype == "object"]
                if not possible_text_cols:
                    st.error("未找到可用文本列，请确保 CSV 中包含文本列。")
                else:
                    text_col = st.selectbox("选择文本列", possible_text_cols)

                    if st.button("开始批量预测", type="primary"):
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        with st.spinner("正在进行批量预测，请稍候..."):
                            result_df = predict_batch_with_progress(
                                df,
                                text_col,
                                selected_model,
                                progress_bar,
                                status_text,
                            )
                            st.session_state.batch_result_df = result_df

            if st.session_state.batch_result_df is not None:
                st.markdown("### 批量预测结果")
                st.dataframe(st.session_state.batch_result_df.head(20), use_container_width=True)

                st.markdown("### 可视化统计结果")
                st.caption("情感分类统计图")
                draw_single_stat_chart(st.session_state.batch_result_df)

                csv_bytes = st.session_state.batch_result_df.to_csv(
                    index=False,
                    encoding="utf-8-sig"
                ).encode("utf-8-sig")

                st.download_button(
                    label="下载批量预测结果 CSV",
                    data=csv_bytes,
                    file_name=f"batch_predict_{selected_model}.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.error(f"批量预测失败：{e}")
