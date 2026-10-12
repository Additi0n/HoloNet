# train.py 参数说明

本文档列出 train.py 当前支持的主要命令行参数，适合在服务器上直接写进启动命令、任务脚本或调度平台配置中。

如果你使用批量实验入口 run_experiment.py，额外可用这几个筛选参数：

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| --datasets | str... | None | 只运行指定数据集，例如 BODMAS。 |
| --experiments | str... | None | 只运行指定实验名，例如 ablation_no_ckan。 |
| --ablation-only | flag | False | 只运行名称以 ablation_ 开头的消融实验。 |
| --dry-run | flag | False | 只打印将要执行的命令，不真正启动训练。 |
| --stop-on-error | flag | False | 任一实验失败时立即停止整个批量任务。 |

## 基础用法

```bash
python train.py --dataset BODMAS --data-root RGBA --epochs 30 --batch-size 32
```

## 参数列表

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| --config | Path | None | JSON 配置文件。若命令行显式传入某参数，则优先使用命令行值。 |
| --data-root | Path | RGBA | 数据集根目录。 |
| --dataset | str | BODMAS | 单次训练的数据集名称。 |
| --all-datasets | flag | False | 依次训练 data-root 下所有符合 train/test 结构的数据集。 |
| --experiment-name | str | holonet | 实验名，会参与输出目录命名。 |
| --run-name | str | None | 手动指定本次运行目录名；为空时自动生成。 |
| --epochs | int | 30 | 训练轮数。 |
| --batch-size | int | 32 | 批大小。 |
| --image-size | int | 96 | 输入图像尺寸，当前 RGBA 样本建议保持 96。 |
| --num-workers | int | 4 | DataLoader 进程数。服务器 CPU 足够时可调大。 |
| --val-ratio | float | 0.1 | 从 train 中切分验证集的比例。 |
| --lr | float | 3e-4 | AdamW 学习率。 |
| --weight-decay | float | 1e-4 | AdamW 权重衰减。 |
| --label-smoothing | float | 0.05 | 交叉熵标签平滑。 |
| --margin-weight | float | 0.25 | Capsule margin loss 权重。设为 0 即关闭该项。 |
| --seed | int | 42 | 随机种子。 |
| --device | str | cuda 或 cpu | 训练设备，例如 cuda、cuda:0、cpu。 |
| --stem-channels | int int int | 32 64 96 | 三段卷积 stem 的通道数。 |
| --primary-capsule-dim | int | 16 | 主胶囊维度。 |
| --primary-capsule-types | int | 12 | 主胶囊类型数量。 |
| --transformer-dim | int | 128 | Transformer 路由隐层维度。 |
| --transformer-heads | int | 4 | Transformer 多头数。 |
| --transformer-layers | int | 2 | Transformer 层数。 |
| --ckan-grid-size | int | 8 | Conv-KAN 中 RBF 网格数。 |
| --dropout | float | 0.1 | 模型内部 dropout。 |
| --resume | Path | None | 从 checkpoint 恢复训练，适合服务器中断后继续。 |
| --output-dir | Path | runs | 结果输出目录。 |

## 布尔开关

这些参数支持成对开关：

- --balanced-sampling / --disable-balanced-sampling
- --amp / --disable-amp
- --use-ckan / --disable-use-ckan
- --use-transformer-router / --disable-use-transformer-router
- --save-last / --disable-save-last

说明：

- balanced-sampling：使用类别加权采样，缓解类别不平衡。
- amp：在 CUDA 上启用混合精度。
- use-ckan：是否启用 Conv-KAN 细粒度特征模块。
- use-transformer-router：是否启用 Transformer 胶囊路由；关闭后会退化为池化路由版本，用于消融实验。
- save-last：每轮保存 last.pt，便于服务器中断后恢复。

## 服务器推荐写法

单卡训练：

```bash
python train.py \
  --dataset BODMAS \
  --data-root RGBA \
  --device cuda:0 \
  --epochs 50 \
  --batch-size 64 \
  --num-workers 8 \
  --experiment-name holonet_server
```

从中断点继续：

```bash
python train.py \
  --dataset BODMAS \
  --data-root RGBA \
  --device cuda:0 \
  --resume runs/holonet_full_bodmas/last.pt
```

关闭 Transformer 做消融：

```bash
python train.py \
  --dataset BODMAS \
  --data-root RGBA \
  --disable-use-transformer-router \
  --experiment-name ablation_no_transformer
```
