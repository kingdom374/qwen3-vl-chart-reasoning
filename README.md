# Qwen3-VL 图表问答微调与评测

基于 Qwen3-VL-2B-Instruct 和 ChartQA，完成基座评测（P0）及多模态 LoRA SFT（P1）。训练只对助手答案与结束符计算损失，并通过单样本交叉熵复算、梯度检查和 adapter 重载验证训练流程。

## 实测结果

### P1 官方验证集全量对照

在同一台 RTX 5090 上，用相同的推理和评分设置比较原始 Qwen3-VL-2B-Instruct（A）与 1,000 步 LoRA 微调后的模型（B）。微调使用 ChartQA 官方 `train`。评测覆盖官方 `val` 全部 **1,920 道题**，由 P0 固定的 100 道题和其余 1,820 道题组成，两组均生成了全部预测。

| 指标 | 原始基座 A | LoRA 微调 B | 增幅 |
| --- | ---: | ---: | ---: |
| Exact Match | 1,273 / 1,920（66.30%） | 1,304 / 1,920（67.92%） | **+1.61 个百分点** |
| Relaxed Accuracy | 1,469 / 1,920（76.51%） | 1,508 / 1,920（78.54%） | **+2.03 个百分点** |

增幅按原始正确数之差计算。Relaxed 口径下有 79 道从错变对、40 道从对变错。正式训练实测约 118 秒，PyTorch 峰值分配显存约 6.34 GiB。模型与数据版本、分组分数、训练配置和耗时见 [P1 实验摘要](outputs/chartqa_p1/summary.json)。

### P0 本机基座评测

在 ChartQA 官方 `val` 的 1,920 道题中，用种子 `42` 从 `human` 和 `augmented` 各抽 50 道，并在 RTX 4060 Laptop GPU 上运行原始基座。下表统计本机的 100 道固定开发题。

| 题目来源 | 题数 | Exact Match | P0 Relaxed Accuracy |
| --- | ---: | ---: | ---: |
| `human` | 50 | 62%（31/50） | 74%（37/50） |
| `augmented` | 50 | 76%（38/50） | 86%（43/50） |
| **总体** | **100** | **69%（69/100）** | **80%（80/100）** |

100 道均生成预测，缺失为 0。对保存的预测执行两次评分，整体及分组统计一致。批量推理的 PyTorch 峰值分配显存为 **4,220.5 MiB（约 4.12 GiB）**，峰值预留为 4,318.0 MiB。该数值包含加载与推理阶段，测量范围为 PyTorch CUDA 分配器。

核对证据可直接打开[统计汇总](outputs/chartqa_p0/batch_bf16_512/summary.json)、[原始预测](outputs/chartqa_p0/batch_bf16_512/predictions.jsonl)、[逐题评分](outputs/chartqa_p0/batch_bf16_512/scored_predictions.jsonl)和[运行配置](outputs/chartqa_p0/batch_bf16_512/config.json)。

## P0 基座评测流程

```mermaid
flowchart LR
    A[ChartQA val] --> B[固定 100 道题]
    B --> C[单图推理验证]
    C --> D[逐题批量推理]
    D --> E[预测 JSONL]
    E --> F[独立评分]
    F --> G[指标与错例复盘]
```

| 脚本 | 作用 |
| --- | --- |
| [inspect_chartqa.py](scripts/inspect_chartqa.py) | 下载验证 Parquet，固定抽样清单，导出首题图片及问答 |
| [run_baseline.py](scripts/run_baseline.py) | 用首题验证模型加载、图像处理、文本生成与显存 |
| [run_baseline_batch.py](scripts/run_baseline_batch.py) | 读取清单和单图配置，加载一次模型，逐题保存预测并支持续跑 |
| [evaluate.py](scripts/evaluate.py) | 校验样本 ID、元数据和清单哈希，计算指标并导出错例 |

推理时输入图片和问题，参考答案用于生成完成后的评分。批量阶段每次处理 1 道题，模型始终驻留同一块 GPU。

## 快速复算 P0 分数

仓库保留了小型文本实验记录。使用 Python 3.12，在项目根目录运行以下命令即可复算分数。评分脚本仅使用 Python 标准库，CPU 即可完成。

```powershell
python scripts/evaluate.py
```

预期输出的关键数字为 `exact_correct = 69`、`relaxed_correct = 80`、`missing_predictions = 0` 和 `Bad cases: 20`。脚本会重新写入派生的逐题评分、汇总及错例文件。

`manifest.json` 的原始字节参与哈希校验，`.gitattributes` 为实验记录保留原始字节。编辑器格式化或重新保存清单后，需要重新核对清单与运行配置的一致性。

## 本地 GPU 推理

### 已验证环境

| 组件 | 版本或配置 |
| --- | --- |
| 系统 | Windows 11，PowerShell |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU，8GB |
| 驱动 | 591.74 |
| Python | 3.12.3 |
| PyTorch / torchvision | `2.10.0+cu130` / `0.25.0+cu130` |
| PyTorch CUDA runtime | `13.0` |
| Transformers / Accelerate | `4.57.6` / `1.12.0` |
| PyArrow / Pillow | `22.0.0` / `11.3.0` |

### 准备环境

以下命令适用于已经具有表中 GPU 版 PyTorch 的 Python。先检查现有环境，再创建可复用这些包的项目虚拟环境。

```powershell
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.version.cuda); print(torch.cuda.is_available())"
python -m venv --system-site-packages .venv
$P0Python = '.\.venv\Scripts\python.exe'
& $P0Python -m pip install --dry-run -r requirements.txt
```

审查预演清单，再安装项目依赖。`constraints-gpu.txt` 将 GPU 包固定为 P0 实测版本。已有这两个版本时 pip 会复用它们。新机器先依据自身系统和 GPU 配置安装兼容的 PyTorch，再准备项目环境。 [PyTorch 安装指南](https://pytorch.org/get-started/locally/)

```powershell
& $P0Python -m pip install -r requirements.txt
& $P0Python -c "from transformers import AutoProcessor, Qwen3VLForConditionalGeneration; import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"
```

VS Code 中使用 `Python: Select Interpreter` 选择 `.venv\Scripts\python.exe`。下面的命令显式指定了该解释器。

### 下载验证数据和运行单图推理

首次运行需要从 Hugging Face 获取约 50 MB 的验证 Parquet，以及约 4.26 GB 的权重和配套处理器文件。缓存放在项目的 `.cache/huggingface/hub`。已有完整缓存时下载器会复用；数据和单图入口仍会联网查询固定版本的元数据。

```powershell
$env:HF_HOME = Join-Path $PWD.Path '.cache\huggingface'
& '.\.venv\Scripts\python.exe' '.\scripts\inspect_chartqa.py'
& '.\.venv\Scripts\python.exe' '.\scripts\run_baseline.py'
```

首题是 `val:6`，题目为 “What is the unfavourable value in 2014?”，参考和本次预测都是 `64`。该图输入含 120 个视觉 token，整条输入为 151 个 token。[单图记录](outputs/chartqa_p0/smoke_bf16_512.json)

### 批量推理和评分

```powershell
& '.\.venv\Scripts\python.exe' '.\scripts\run_baseline_batch.py'
& '.\.venv\Scripts\python.exe' '.\scripts\evaluate.py'
```

批量脚本按 `sample_id` 跳过已有预测，启动时校验清单和配置。下载仓库里的 100 条预测记录后再运行，脚本会报告已完成。如果要重新生成全部预测，先把已有的整个 `outputs/chartqa_p0/batch_bf16_512` 目录备份到项目外，再运行批量命令。单图、清单和缓存继续供它读取。

当前批量入口沿用 512-token 单图配置，使用离线缓存。单图的 `--max-image-tokens 256` 可用于显存排查；固定 100 题实验采用 512 设置，修改配置时需另开实验记录。

### 固定的模型与解码设置

| 设置 | 值 |
| --- | --- |
| 模型 commit | `89644892e4d85e24eaac8bacfd4f463576704203` |
| 数据 commit | `af8b6f5c08c95085271561c2a3f9d15f2b5a9031` |
| 精度 / 注意力实现 | BF16 / SDPA |
| 视觉 token 上限 / 新增 token 上限 | 512 / 32 |
| 解码 | 贪心，`do_sample=False` |
| 题目后追加的提示词 | `Answer with only the final answer.` |

## P1 单卡 LoRA 训练与评测

### 环境与缓存

P1 实测环境为 Linux、RTX 5090、Python `3.12.3`、PyTorch `2.8.0+cu128`、torchvision `0.23.0+cu128`。以下命令在项目根目录运行，先使用已有这两个 GPU 包的 Python 创建虚拟环境。P0 的 `requirements.txt` 对应 Windows 推理环境。

```bash
python -m venv --system-site-packages .venv-p1
P1_PYTHON=./.venv-p1/bin/python
"$P1_PYTHON" -m pip install \
  'torch==2.8.0+cu128' 'torchvision==0.23.0+cu128' \
  'transformers==4.57.6' 'accelerate==1.12.0' 'peft==0.18.0' \
  'pyarrow==22.0.0' 'huggingface-hub==0.36.2' 'tokenizers==0.22.2' \
  'regex==2026.11.9' 'Pillow==11.3.0' 'filelock==3.20.0'
"$P1_PYTHON" -m pip check
```

缓存中需要固定版本的模型权重、处理器文件和 ChartQA 的三个 `train` 分片及一个 `val` 分片。可以在有网络时运行以下命令准备缓存。已从其他机器同步完整缓存时，可直接进入离线阶段。

```bash
export HF_HOME="$PWD/.cache/huggingface"
"$P1_PYTHON" - <<'PY'
from huggingface_hub import hf_hub_download, snapshot_download

snapshot_download(
    "Qwen/Qwen3-VL-2B-Instruct",
    revision="89644892e4d85e24eaac8bacfd4f463576704203",
    allow_patterns=[
        "*.json", "merges.txt", "model.safetensors",
    ],
)

for filename in ["data/val-00000-of-00001.parquet", *(
    f"data/train-{index:05d}-of-00003.parquet" for index in range(3)
)]:
    hf_hub_download(
        "ahmed-masry/ChartQA",
        filename=filename,
        repo_type="dataset",
        revision="af8b6f5c08c95085271561c2a3f9d15f2b5a9031",
    )
PY
```

### 离线复现

| 脚本 | 作用 |
| --- | --- |
| [inspect_sft_sample.py](scripts/inspect_sft_sample.py) | 用 `train:0` 核对图文输入、助手答案边界和 `labels` |
| [run_lora_smoke.py](scripts/run_lora_smoke.py) | 复算首条样本的交叉熵，验证梯度、权重更新和 adapter 重载 |
| [train_lora.py](scripts/train_lora.py) | 使用官方 `train` 做 LoRA SFT，保存训练日志和 adapter |
| [run_ab.py](scripts/run_ab.py) | 在同条件下评测基座和 LoRA，支持逐题续跑及 P0 评分规则 |

以下三项预检可在 CPU 上运行。

```bash
export HF_HUB_OFFLINE=1
"$P1_PYTHON" scripts/inspect_sft_sample.py
"$P1_PYTHON" scripts/train_lora.py --check-only
"$P1_PYTHON" scripts/run_ab.py --eval-set remaining-val --check-only
```

以下训练和推理命令使用一张支持 BF16 的 CUDA GPU。B 默认读取正式训练保存的 adapter，两个评测子集各有一个结果目录。重复训练可指定新的 `--output-dir`，评测时用 `--adapter` 加载相应权重，并用 `--output-root` 保存独立对照。

```bash
"$P1_PYTHON" -u scripts/run_lora_smoke.py
"$P1_PYTHON" -u scripts/train_lora.py --max-steps 1000 --save-steps 100
"$P1_PYTHON" -u scripts/run_ab.py --group A
"$P1_PYTHON" -u scripts/run_ab.py --group B
"$P1_PYTHON" -u scripts/run_ab.py --eval-set remaining-val --group A
"$P1_PYTHON" -u scripts/run_ab.py --eval-set remaining-val --group B
```

## 评分规则与错误分析

**Exact Match** 统一大小写及空白后比较答案文本。**P0 Relaxed Accuracy** 对能解析的数值使用以下规则，其他答案按规范化文本匹配。

```text
|prediction - reference| <= 0.05 * |reference|
```

数值可包含正负号、小数和规范的千分位分隔符。参考为零时要求预测数值为零，百分号口径必须一致。列表、同义词和解释性回答目前使用文本匹配规则。评分脚本内置了零值、负数、容差边界、百分号和空输出等规则检查。5% 数值容差参考 ChartQA 论文，项目的文本规范化与解析规则可在评分源码中检查。 [ChartQA 论文](https://aclanthology.org/2022.findings-acl.177.pdf)

对 20 条自动判错中的 3 条做了原图复盘：

| 样本 | 参考 / 预测 | 原图核查 |
| --- | --- | --- |
| `val:159` | `0.354` / `35.4` | 图上标注英国为 `35.4%`，预测遗漏百分号，严格口径下判错 |
| `val:348` | `[Oppose, Favor]` / `Favor and Oppose` | 图例确为这两类，题目未指定顺序，人工判断语义等价，当前匹配规则未覆盖 |
| `val:281` | `No` / `Yes` | 深蓝最大值 47，浅蓝中位数 28，两者之和 75，确认答错 |

人工复盘保留了原始评分。后续若调整匹配规则，将用同一新版规则重评所有对照组。完整错例见 [bad_cases.csv](outputs/chartqa_p0/batch_bf16_512/bad_cases.csv)。

## 结果范围与后续实验

P0 固定的 100 道题用于开发与流程验证。P1 主结果覆盖官方 `val` 全集，来自一次 LoRA 训练，随机种子为 42。匹配规则会影响自动评测结果，官方 `test` 预留给后续最终评估。

| 阶段 | 状态 | 目标 |
| --- | --- | --- |
| P0 基座 Baseline | 已完成 | 单图、固定清单推理、评分、日志和错例复盘 |
| P1 LoRA SFT | 已完成 | 单样本损失核对、10 步冒烟、1,000 步训练和全量验证集 A/B 对照 |
| P2 可验证合成数据 | 待开展 | 程序绘图与答案计算，筛选并开展数据增强对照 |

## 数据与模型来源

仓库中的固定清单和预测记录含有 ChartQA 的题目及参考答案。下载数据、使用和分发时应遵循数据来源的说明。数据卡标注 `gpl-3.0`，模型卡标注 `apache-2.0`，原始图表的来源标识可在数据图像中查看。 [ChartQA 数据卡](https://huggingface.co/datasets/ahmed-masry/ChartQA) · [Qwen3-VL-2B-Instruct 模型卡](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)

参考项目为 [Qwen3-VL 官方仓库](https://github.com/QwenLM/Qwen3-VL) 与 [ChartQA 官方仓库](https://github.com/vis-nlp/ChartQA)。
