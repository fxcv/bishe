# Streamlit Cloud 部署说明

本项目的 `app.py` 已改成同时支持本地模型目录和云端模型仓库：

- 本地运行时，优先读取 `models/` 下的现有模型。
- 如果模型通过 Git LFS 放在 GitHub 仓库中，Streamlit Cloud 会直接从仓库读取模型。
- 如果 Streamlit Cloud 上没有模型目录，还会从 Hugging Face Hub 下载 Secrets 中配置的模型仓库。

## 方案 A：模型随 GitHub 仓库部署（Git LFS）

本仓库已配置 `.gitattributes` 和 `.gitignore`，只允许下面这些上线必需模型目录进入 Git：

- `models/base_bert_base_chinese`
- `models/bert_weibo/best_model`
- `models/macbert_weibo/best_model`
- `models/roberta_weibo/best_model`
- `models/bert_bigru_weibo/best_model`
- `models/bert_lora_weibo/best_model`

提交前先确认 Git LFS 已启用：

```powershell
git lfs install
git lfs track
git add .gitattributes .gitignore app.py requirements.txt requirements_app.txt DEPLOY_STREAMLIT.md .streamlit/secrets.example.toml
git add models/base_bert_base_chinese models/bert_weibo/best_model models/macbert_weibo/best_model models/roberta_weibo/best_model models/bert_bigru_weibo/best_model models/bert_lora_weibo/best_model
git commit -m "Deploy Streamlit sentiment app"
git push origin main
```

注意：这 6 个目录合计约 1.96GB。GitHub Free 当前包含 10GiB LFS 存储和 10GiB/月下载带宽；Streamlit 首次部署和重启下载模型会消耗 LFS 带宽。

## 方案 B：模型放 Hugging Face Hub

建议在 Hugging Face Hub 创建 5 个模型仓库，并把下面目录中的文件分别上传到对应仓库根目录：

| Streamlit 模型名 | 本地目录 | 建议仓库名 |
| --- | --- | --- |
| BERT | `models/bert_weibo/best_model` | `your-hf-name/bishe-bert-weibo` |
| MacBERT | `models/macbert_weibo/best_model` | `your-hf-name/bishe-macbert-weibo` |
| RoBERTa-wwm-ext | `models/roberta_weibo/best_model` | `your-hf-name/bishe-roberta-weibo` |
| BERT+BiGRU | `models/bert_bigru_weibo/best_model` | `your-hf-name/bishe-bert-bigru-weibo` |
| BERT+LoRA | `models/bert_lora_weibo/best_model` | `your-hf-name/bishe-bert-lora-weibo` |

如果模型仓库设为私有，需要创建一个 Hugging Face read token，后面填到 Streamlit Secrets。

### 配置 Streamlit Secrets

在 Streamlit Community Cloud 创建应用时，进入 Advanced settings，将 `.streamlit/secrets.example.toml` 的内容复制进去，并把 `your-hf-name/...` 替换为真实仓库名：

```toml
HF_TOKEN = "hf_xxx"

[model_repos]
bert = "your-hf-name/bishe-bert-weibo"
macbert = "your-hf-name/bishe-macbert-weibo"
roberta = "your-hf-name/bishe-roberta-weibo"
bert_bigru = "your-hf-name/bishe-bert-bigru-weibo"
bert_lora = "your-hf-name/bishe-bert-lora-weibo"
```

如果只想先部署一个可用版本，可以只配置一个模型仓库；应用侧边栏只会显示本地存在或已配置仓库的模型。

## 创建 Streamlit 应用

在 Streamlit Community Cloud 中选择：

- Repository: `fxcv/bishe`
- Branch: `main`
- Main file path: `app.py`

确认根目录存在 `requirements.txt` 后点击 Deploy。首次访问时需要下载模型，启动会比普通 Streamlit 应用慢一些。

## 参考

- Streamlit Community Cloud 会从仓库根目录运行应用，并读取根目录或入口文件旁边的依赖文件：<https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization>
- Python 依赖应放在 `requirements.txt`：<https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies>
- Secrets 不应提交到仓库，应通过 Streamlit 的 Secrets 管理界面填写：<https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management>
- 应用使用 `huggingface_hub.snapshot_download` 下载并缓存模型仓库：<https://huggingface.co/docs/huggingface_hub/guides/download>
