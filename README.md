# HoloNet: Capsule + Conv-KAN + Transformer

[中文说明](README.zh-CN.md)

This project implements a trainable cross-platform malware image classifier based on the core ideas of two design documents:

- RGBA malware images are used as input.
- A Capsule Network serves as the backbone representation.
- A Convolutional KAN-style module is added to the convolutional stem to strengthen fine-grained high-level feature modelling and robustness.
- A Transformer aggregates the primary capsules globally in a single pass, in place of iterative dynamic routing.

The code reads the RGBA dataset directories in the workspace directly:

- RGBA/BODMAS
- RGBA/CICMalDroid2020
- RGBA/ELF2020

## Directory Layout

### Core code

- holonet/model.py: model definition
- holonet/data.py: dataset scanning, stratified validation split, DataLoader construction
- holonet/losses.py: cross-entropy + capsule margin loss
- holonet/metrics.py: accuracy and macro-averaged Precision/Recall/F1

### Preprocessing

- preprocessing/bin2img.py: Bin2Img conversion of executables to 96x96 RGBA images

### Entry scripts (run from the project root)

- train.py: training and testing entry point
- run_experiment.py / run_experiments.py: batch ablation experiments
- run_experiement_robustness.py: perturbation robustness evaluation
- run_explainability.py: explainability outputs (saliency maps / Grad-CAM)

### Overall structure

```
hOlo/
├── README.md, requirements.txt   # documentation and dependencies
├── train.py, run_*.py            # entry scripts (run from the root)
├── holonet/                      # core package (model / data / losses / metrics)
├── preprocessing/                # Bin2Img conversion (bin2img.py)
├── RGBA/                         # datasets (BODMAS / CICMalDroid2020 / ELF2020)
├── experiments/                  # experiment configs (ablation_suite.json, etc.)
├── scripts/                      # helper scripts (batch experiments, figure tools)
├── runs/                         # current experiment outputs (best.pt / history.json, etc.)
├── all_results/                  # archived raw run results
├── article/                      # paper project (.tex / .pdf / images / icons / data)
├── result_tables/                # hand-curated result tables (robustness / explainability .md/.tex)
├── reports/                      # presentation and chapter material
├── references/                   # external reference papers
├── docs/                         # supplementary docs (TRAIN_ARGUMENTS.md)
└── .venv/, .vscode/              # virtual environment and editor settings
```

> Note: `runs/` holds outputs written live by the scripts, `all_results/` holds archived raw results,
> and `result_tables/` holds summary tables curated from them. The three serve different purposes.
>
> Datasets (`RGBA*/`), run outputs (`runs/`, `all_results/`), experiment results (`experiments_local/**/results/`, `result_tables/`), checkpoints, the manuscript (`article/`),
> reference papers and presentation material are not included in this repository.

## Model Architecture

The model consists of:

1. The RGBA input passes through a three-stage convolutional stem that builds low- to mid-level texture representations.
2. The second-stage features enter the Conv-KAN module: a depthwise convolution first extracts local structure, then the channel responses are expanded with RBF bases and combined with learnable weights, emulating the per-dimension learnable nonlinear mappings of a KAN.
3. The fused features enter the Primary Capsules layer, which builds primary-capsule tokens over spatial positions.
4. The primary-capsule sequence is fed, together with learnable class-capsule tokens, into a Transformer Encoder, whose self-attention performs global relation modelling and "routing" aggregation.
5. The length of each class capsule is used as the classification confidence, and a capsule margin loss is added to improve inter-class separation.

## Installing Dependencies

```powershell
python -m pip install -r requirements.txt
```

For a GPU build of PyTorch, first install torch and torchvision for your CUDA version following the official instructions, then install the remaining dependencies.

## Preparing the Data (Bin2Img)

`preprocessing/bin2img.py` turns executables into the 96x96 RGBA input images in two steps. Each dataset is laid out as `<split>/<class>/<file>`:

```powershell
python preprocessing/bin2img.py convert RAW/BODMAS/train RawIMG/BODMAS/train
python preprocessing/bin2img.py resize  RawIMG/BODMAS/train RGBA/BODMAS/train
```

`convert` packs 4 consecutive bytes into one RGBA pixel of an n x n image (n = ceil(sqrt(len / 4)), zero-padded, lossless). `resize` scales it to 96x96 with Pillow's default filter, which is bicubic and works on premultiplied alpha for RGBA images. Repeat for the `test` split and the other datasets. The SHA-256/MD5 hash lists and train/test assignments of every sample are provided with the paper's Supplementary Material.

## Res2Net Baseline Source

The Res2Net-50 baseline loads `res2net_v1b.py` from the official [Res2Net-PretrainedModels](https://github.com/Res2Net/Res2Net-PretrainedModels) repository, which is not redistributed here because of its licence. Clone that repository and point the `RES2NET_SOURCE` environment variable to the file:

```powershell
$env:RES2NET_SOURCE = "C:\path\to\Res2Net-PretrainedModels\res2net_v1b.py"
```

## Reproducing the Revision Experiments

`scripts/run_revision_experiments.py` runs every training job of the revised paper under one protocol (200 epochs, AdamW 3e-4, cosine schedule, batch size 32, label smoothing 0.05, best-validation-Macro-F1 checkpoint): HoloNet and its ablations over five seeds, the five image baselines over three seeds, the input-channel ablation and the near-duplicate-grouped split. It resumes interrupted jobs from `last.pt` and skips finished ones:

```powershell
python scripts/run_revision_experiments.py --workers 2
```

The analysis scripts in `experiments_local/revision/` then produce the reported numbers: `grouped_split.py` (near-duplicate-grouped split), `nearest_neighbour.py` and `dedup_check.py` (duplicate analysis), `aggregate_results.py` (means, Welch and McNemar tests with Holm correction), `efficiency_all.py --latency` (parameters, FLOPs, latency), `transfer_controls.py`, `confidence_analysis.py`, `faithfulness.py zero|blur` and `export_per_seed.py`. Note that `train.py` defaults to 30 epochs; the paper uses 200.

## Training on a Single Dataset

```powershell
python train.py --dataset BODMAS --data-root RGBA --epochs 30 --batch-size 32
```

Available dataset names:

- BODMAS
- CICMalDroid2020
- ELF2020

## Training on All Datasets in Sequence

```powershell
python train.py --all-datasets --data-root RGBA --epochs 30 --batch-size 32
```

## Batch Experiments on a Server

The ablation configuration is in experiments/ablation_suite.json and can be run in sequence through the batch entry point:

```powershell
python run_experiment.py --config experiments/ablation_suite.json --all-datasets --device cuda:0
```

Run only the ablations, without the full model:

```powershell
python run_experiment.py --config experiments/ablation_suite.json --all-datasets --device cuda:0 --ablation-only
```

Run only selected experiments:

```powershell
python run_experiment.py --datasets BODMAS --device cuda:0 --experiments ablation_no_ckan ablation_no_transformer
```

## Robustness Experiments

After training, existing best.pt checkpoints can be evaluated under perturbations with run_experiement_robustness.py:

```powershell
python run_experiement_robustness.py --runs-root runs --device cuda:0
```

By default only full models are evaluated; runs whose names start with ablation_ are skipped. To include the ablations, turn this filter off explicitly:

```powershell
python run_experiement_robustness.py --runs-root runs --device cuda:0 --disable-full-only
```

Evaluate a single run directory:

```powershell
python run_experiement_robustness.py --run-dirs runs/holonet_full_bodmas --device cuda:0
```

The script prints progress to the terminal and writes the full log to runs/robustness/logs. Result JSON files are written to runs/robustness, together with a summary.json.

## Explainability Experiments

After training, explainability outputs can be generated from existing best.pt checkpoints with run_explainability.py:

```powershell
python run_explainability.py --runs-root runs --device cuda:0
```

By default the script scans only full models, selects several high-confidence correctly classified samples per class, and produces:

- the original image
- the saliency-map overlay
- the Grad-CAM overlay
- a panel combining the three images
- report.md and summary.json

Results are written to runs/explainability/<run_name>.

Batch experiments write each experiment's standard output to a separate log file under runs/launcher and produce a summary.json.

Training results are saved under runs:

- runs/[dataset]/best.pt: best weights
- runs/[dataset]/history.json: training history and test results
- runs/summary.json: summary across all datasets

Each run is written to a directory runs/<run_name> named after the experiment, which additionally contains:

- best.pt: weights with the best validation score
- last.pt: weights from the latest epoch, for resuming training on a server
- resolved_args.json: the arguments actually used for the run

## Default Training Strategy

- The input size is fixed at 96x96, matching the current RGBA samples.
- 10% of the training split is held out as a class-stratified validation set.
- Weighted sampling is enabled by default to mitigate class imbalance.
- The loss is cross-entropy + capsule margin loss.
- The learning rate follows CosineAnnealingLR.
- AMP is enabled by default when CUDA is available.

## Notes

The source documents mainly describe the method and the experimental conclusions; they do not provide complete reproducible code or all hyperparameters. This implementation therefore follows the core design goals described there and keeps the key structural parameters configurable in code, so that ablations and tuning can be continued later.

See docs/TRAIN_ARGUMENTS.md for a detailed description of the arguments.
