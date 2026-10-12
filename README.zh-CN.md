# HoloNet: Capsule + Conv-KAN + Transformer

[English](README.md)

这个工程基于两份文档中的核心思路实现了一个可训练的跨平台恶意软件图像分类模型：

- 使用 RGBA 恶意软件图像作为输入。
- 以 Capsule Network 作为主干表示。
- 在主干卷积阶段加入 Convolutional KAN 风格模块，增强高级细粒度特征建模和鲁棒性。
- 使用 Transformer 对主胶囊一次性进行全局聚合，替代传统的迭代动态路由。

当前代码直接读取工作区中的 RGBA 数据集目录：

- RGBA/BODMAS
- RGBA/CICMalDroid2020
- RGBA/ELF2020

## 目录说明

### 核心代码

- holonet/model.py: 模型定义
- holonet/data.py: 数据集扫描、分层验证划分、DataLoader 构建
- holonet/losses.py: 交叉熵 + Capsule margin loss
- holonet/metrics.py: 准确率、宏平均 Precision/Recall/F1

### 预处理

- preprocessing/bin2img.py: 将可执行文件转换为 96x96 RGBA 图像（Bin2Img）

### 入口脚本（在项目根目录运行）

- train.py: 训练与测试入口
- run_experiment.py / run_experiments.py: 批量消融实验入口
- run_experiement_robustness.py: 扰动鲁棒性评估入口
- run_explainability.py: 可解释性（显著图 / Grad-CAM）生成入口

### 整体目录结构

```
hOlo/
├── README.md, requirements.txt   # 说明与依赖
├── train.py, run_*.py            # 入口脚本（在根目录运行）
├── holonet/                      # 核心代码包（模型 / 数据 / 损失 / 指标）
├── preprocessing/                # Bin2Img 转换脚本（bin2img.py）
├── RGBA/                         # 数据集（BODMAS / CICMalDroid2020 / ELF2020）
├── experiments/                  # 实验配置（ablation_suite.json 等）
├── scripts/                      # 辅助脚本（批量实验、配图生成工具）
├── runs/                         # 当前实验输出（best.pt / history.json 等）
├── all_results/                  # 收集归档的原始 run 结果
├── article/                      # 论文工程（.tex / .pdf / images / icons / data）
├── result_tables/               # 人工整理的结果汇总表（鲁棒性 / 可解释性 .md/.tex）
├── reports/                      # 汇报与章节材料（slide.pptx、二章-MCKAN.*）
├── references/                   # 外部参考论文（FACILE 等）
├── docs/                         # 补充文档（TRAIN_ARGUMENTS.md）
└── .venv/, .vscode/             # 虚拟环境与编辑器配置
```

> 说明：`runs/` 是脚本实时写出的实验输出，`all_results/` 是收集归档后的原始结果，
> `result_tables/` 则是在此基础上人工整理的汇总表格，三者用途不同请勿混淆。

## 模型结构

模型主体为：

1. RGBA 输入经过三层卷积 stem 形成低级到中级纹理表示。
2. 第二阶段特征送入 Conv-KAN 模块：先做 depthwise 卷积提取局部结构，再对通道响应做 RBF basis 展开并用可学习权重聚合，模拟 KAN 中按维度学习非线性映射的能力。
3. 融合后的特征进入 Primary Capsules，构建空间位置上的主胶囊 token。
4. 将主胶囊序列与可学习 class capsule token 一起送入 Transformer Encoder，由自注意力完成全局关系建模与“路由”聚合。
5. 取类别胶囊长度作为分类置信度，同时结合 capsule margin loss 提升类间分离性。

## 安装依赖

```powershell
python -m pip install -r requirements.txt
```

如果需要 GPU 版本的 PyTorch，建议先按官方方式安装对应 CUDA 版本的 torch 和 torchvision，再补装其余依赖。

## 数据准备（Bin2Img）

`preprocessing/bin2img.py` 分两步把可执行文件转成 96x96 的 RGBA 输入图像。每个数据集按 `<划分>/<类别>/<文件>` 组织：

```powershell
python preprocessing/bin2img.py convert RAW/BODMAS/train RawIMG/BODMAS/train
python preprocessing/bin2img.py resize  RawIMG/BODMAS/train RGBA/BODMAS/train
```

`convert` 把每 4 个连续字节写成 n x n 图像中的一个 RGBA 像素（n = ceil(sqrt(len / 4))，不足补零，无损）；`resize` 用 Pillow 的默认滤波器缩放到 96x96，对 RGBA 图像是在预乘 alpha 上做双三次插值。`test` 划分和其他数据集同样处理。全部样本的 SHA-256/MD5 哈希清单及训练/测试归属见论文补充材料。

## Res2Net 基线源码

Res2Net-50 基线从官方 [Res2Net-PretrainedModels](https://github.com/Res2Net/Res2Net-PretrainedModels) 仓库加载 `res2net_v1b.py`，受其许可证限制不在本仓库中分发。请先克隆该仓库，再用环境变量 `RES2NET_SOURCE` 指向该文件：

```powershell
$env:RES2NET_SOURCE = "C:\path\to\Res2Net-PretrainedModels\res2net_v1b.py"
```

## 复现修改稿实验

`scripts/run_revision_experiments.py` 按统一协议（200 个 epoch、AdamW 3e-4、余弦退火、批大小 32、标签平滑 0.05、按验证集 Macro-F1 选检查点）运行修改稿的全部训练任务：HoloNet 及其消融（5 个种子）、五个图像基线（3 个种子）、输入通道消融和按近重复分组的划分。中断的任务从 `last.pt` 续跑，已完成的任务自动跳过：

```powershell
python scripts/run_revision_experiments.py --workers 2
```

之后用 `experiments_local/revision/` 中的分析脚本得到文中的数字：`grouped_split.py`（分组划分）、`nearest_neighbour.py` 和 `dedup_check.py`（重复分析）、`aggregate_results.py`（均值、Welch 与 McNemar 检验及 Holm 校正）、`efficiency_all.py --latency`（参数量、FLOPs、延迟）、`transfer_controls.py`、`confidence_analysis.py`、`faithfulness.py zero|blur` 和 `export_per_seed.py`。注意 `train.py` 默认 30 个 epoch，论文使用 200 个。

## 训练单个数据集

```powershell
python train.py --dataset BODMAS --data-root RGBA --epochs 30 --batch-size 32
```

可选数据集名称：

- BODMAS
- CICMalDroid2020
- ELF2020

## 顺序训练全部数据集

```powershell
python train.py --all-datasets --data-root RGBA --epochs 30 --batch-size 32
```

## 服务器批量实验

消融配置已经放在 experiments/ablation_suite.json，可以直接通过批量入口顺序执行：

```powershell
python run_experiment.py --config experiments/ablation_suite.json --all-datasets --device cuda:0
```

只跑消融项，不跑 full 模型：

```powershell
python run_experiment.py --config experiments/ablation_suite.json --all-datasets --device cuda:0 --ablation-only
```

只跑指定的某几个实验：

```powershell
python run_experiment.py --datasets BODMAS --device cuda:0 --experiments ablation_no_ckan ablation_no_transformer
```

## 鲁棒性实验

训练完成后，可以对已有 best.pt 做扰动鲁棒性评估。入口脚本为 run_experiement_robustness.py：

```powershell
python run_experiement_robustness.py --runs-root runs --device cuda:0
```

默认只会评估 full 模型，不会扫描 ablation_ 开头的消融 run。如果确实需要连消融一起评估，可显式关闭该筛选：

```powershell
python run_experiement_robustness.py --runs-root runs --device cuda:0 --disable-full-only
```

只评估某个 run 目录：

```powershell
python run_experiement_robustness.py --run-dirs runs/holonet_full_bodmas --device cuda:0
```

该脚本会实时把评估过程输出到终端，同时把完整日志写入 runs/robustness/logs。结果 json 会写到 runs/robustness，并生成汇总 summary.json。

## 可解释性实验

训练完成后，可以对已有 best.pt 生成可解释性结果。入口脚本为 run_explainability.py：

```powershell
python run_explainability.py --runs-root runs --device cuda:0
```

该脚本默认只扫描 full 模型，为每个类别挑选若干高置信正确样本，并生成：

- 原图
- 显著图叠加结果
- Grad-CAM 叠加结果
- 三图拼接的 panel 图
- report.md 与 summary.json

结果会写到 runs/explainability/<run_name> 目录下。

批量实验会把每个实验的标准输出写到 runs/launcher 下的独立日志文件中，并生成汇总 summary.json。

训练结果会保存在 runs 目录下：

- runs/[dataset]/best.pt: 最优权重
- runs/[dataset]/history.json: 训练历史与测试结果
- runs/summary.json: 全部数据集汇总

当前版本会把单次实验输出到按实验名自动生成的 runs/<run_name> 目录，并额外保存：

- best.pt: 验证集最优权重
- last.pt: 最近一轮权重，便于服务器恢复训练
- resolved_args.json: 本次训练最终生效的参数

## 默认训练策略

- 输入尺寸固定为 96x96，与当前 RGBA 样本一致。
- 从 train 划分 10% 作为验证集，保持类别分层。
- 默认启用加权采样，缓解类别不均衡。
- 损失函数为交叉熵 + Capsule margin loss。
- 学习率调度使用 CosineAnnealingLR。
- 若设备支持 CUDA，则默认启用 AMP。

## 说明

文档中公开的信息主要覆盖方法思路与实验结论，没有给出完整可复现实验代码或全部超参数。因此本实现遵循论文描述的核心设计目标，并将关键结构参数保留为代码中的可调配置，便于你后续继续贴近论文版本做消融或调参。

详细参数说明见 docs/TRAIN_ARGUMENTS.md。
