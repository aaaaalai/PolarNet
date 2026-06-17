# PolarNet

> Enhanced Polar Coordinate Lane Detection via
Feature Fusion and Geometry-Aware Adaptive
Weighting

## 1. Introduction

Polar-Net is a lane detection project developed based on the Polar R-CNN framework. The current codebase focuses on **2D lane detection** and supports complete **training, evaluation, visualization, and FPS testing** workflows.

Compared with the original baseline, this repository has been adapted and reorganized for personal research and engineering use. According to the current code, the project mainly includes the following characteristics:

- Supports **CULane**, **TuSimple**, and **CurveLanes** datasets.
- Supports multiple backbones, including **ResNet18** and **DLA34**.
- Introduces **CBAM / Polar-CBAM style attention enhancement** into the backbone implementation.
- Adopts a **two-stage anchor-based lane detection framework**, including backbone, neck, RPN head, ROI head, and loss computation.
- Provides independent scripts for **training (`train.py`)**, **testing (`test.py`)**, and **FPS benchmarking (`tools/test_fps.py`)**.
- Supports both **single-GPU** and **multi-GPU** training.

This repository is suitable for:

- reproducing lane detection experiments,
- conducting ablation studies on attention modules or backbones,
- extending the framework to new lane datasets or custom modules.

---

## 2. Project Structure

```text
Polar-net/
├── Config/                 # Configuration files for different datasets/backbones
├── Dataset/                # Dataset loading and preprocessing
├── Eval/                   # Evaluation and visualization logic
├── Loss/                   # Overall loss, ROI loss, RPN loss
├── Models/
│   ├── Backbone/           # Backbone definitions (ResNet, DLA, ConvNeXt)
│   ├── Neck/               # Neck modules (e.g., FPN)
│   ├── Head/               # RPN head and ROI head
│   ├── anchor_based_lane_detector.py
│   └── attention_cbamLL.py
├── ops/                    # Custom operators (e.g., NMSOPS)
├── tools/                  # Config tools, logging, FPS test, multi-GPU utils
├── utils/                  # Coordinate transform, plotting, dataloader utilities
├── train.py                # Training entry
├── test.py                 # Evaluation / visualization entry
├── exclude_culane.py       # Script for generating filtered CULane training list
├── requirements.txt        # Python dependencies
└── INSTALL.md              # Installation and dataset preparation notes
```

---

## 3. Supported Datasets

Based on the current implementation in `Dataset/build.py` and `Eval/build.py`, this version supports the following datasets:

- **CULane**
- **TuSimple**
- **CurveLanes**

> Note: the old README mentions LLAMAS and DL-Rail, but the current public code path of this repository only includes complete training/evaluation builders for the three datasets above.

### Recommended dataset directory layout

#### CULane

```text
datasets/CULane/
├── driver_23_30frame/
├── driver_37_30frame/
├── ...
└── list/
    ├── test_split/
    ├── test.txt
    ├── train.txt
    ├── train_gt.txt
    ├── train_gt_new.txt
    └── val_gt.txt
```

#### TuSimple

```text
datasets/TUSimple/
├── train_set/
│   ├── clips/
│   ├── seg_label/
│   ├── label_data_0313.json
│   ├── label_data_0531.json
│   └── label_data_0601.json
├── test_set/
│   ├── clips/
│   └── test_tasks_0627.json
└── test_label.json
```

#### CurveLanes

```text
datasets/Curvelanes/
├── train/
│   ├── images/
│   ├── labels/
│   └── train.txt
├── valid/
│   ├── images/
│   ├── labels/
│   └── valid.txt
└── test/
    └── images/
```

---

## 4. Environment Setup

### 4.1 Basic environment

Recommended environment:

- Python >= 3.10
- PyTorch >= 2.0.0
- CUDA-enabled GPU

### 4.2 Install dependencies

```bash
pip install -r requirements.txt
```

### 4.3 Build custom operators

```bash
cd ops/NMSOPS
python setup.py install
cd ../..
```

---

## 5. Configuration Files

The project uses configuration files under the `Config/` directory. The current repository includes:

- `Config/polarrcnn_culane_dla34.py`
- `Config/polarrcnn_tusimple_r18.py`
- `Config/polarrcnn_curvelanes_dla34.py`

These config files define:

- dataset type and data root
- input resolution and crop settings
- maximum lane number
- data augmentation strategy
- backbone / neck / head settings
- number of priors and offsets
- training hyperparameters
- post-processing thresholds

The function `tools/get_config.py` supports loading config in three ways:

- Python file path, e.g. `./Config/polarrcnn_tusimple_r18.py`
- YAML file path
- Python module path

---

## 6. Training

### 6.1 Single-GPU training

```bash
python -u train.py \
  --gpu_no 0 \
  --save_path ./work_dir/culane_dla34 \
  --cfg ./Config/polarrcnn_culane_dla34.py \
  --iter_display 20
```

### 6.2 Multi-GPU training

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 torchrun \
  --nproc_per_node=2 \
  --master_port=12345 \
  train.py \
  --save_path ./work_dir/culane_dla34 \
  --cfg ./Config/polarrcnn_culane_dla34.py \
  --iter_display 20 \
  --is_multigpu 1
```

### 6.3 Training notes

- Model weights are saved to the folder specified by `--save_path`.
- In the current code, checkpoints are saved every 2 epochs as `para_*.pth`.
- The optimizer is **AdamW**.
- The learning rate schedule is **warmup + cosine decay**.
- Random seed is set through the config file.

---

## 7. Evaluation

### 7.1 Evaluate on the test set

```bash
python -u test.py \
  --gpu_no 0 \
  --weight_path ./work_dir/culane_dla34/para_31.pth \
  --result_path ./result/culane \
  --cfg ./Config/polarrcnn_culane_dla34.py
```

### 7.2 Evaluate on the validation set

```bash
python -u test.py \
  --gpu_no 0 \
  --weight_path ./work_dir/culane_dla34/para_31.pth \
  --result_path ./result/culane_val \
  --cfg ./Config/polarrcnn_culane_dla34.py \
  --is_val 1
```

---

## 8. Visualization

To save visualized predictions:

```bash
python -u test.py \
  --gpu_no 0 \
  --weight_path ./work_dir/culane_dla34/para_31.pth \
  --cfg ./Config/polarrcnn_culane_dla34.py \
  --is_view 1 \
  --view_path ./view/culane
```

---

## 9. FPS Testing

The repository also provides a dedicated FPS script:

```bash
python -u tools/test_fps.py \
  --gpu_no 0 \
  --cfg ./Config/polarrcnn_culane_dla34.py \
  --weight_path ./work_dir/culane_dla34/para_31.pth \
  --test_batch_size 32 \
  --fps_warmup_batches 10 \
  --max_batches 200
```

### FPS script behavior

- performs warmup before timing,
- measures average throughput on the test dataloader,
- is suitable for comparing inference speed under the current codebase.

---

## 10. Model Overview

The current framework follows a **two-stage anchor-based lane detector** design.

### Main pipeline

1. **Backbone** extracts multi-scale features.
2. **Neck (FPN)** fuses feature maps from different stages.
3. **RPN Head** predicts local polar priors.
4. **ROI Head** refines lane representations and outputs final lane predictions.
5. **Loss module** combines ROI loss and RPN loss for end-to-end training.

### Backbone options

According to `Models/Backbone/build.py`, the framework currently supports:

- `resnet18` / other resnet variants
- `dla34`
- `convnextT`

### Attention enhancement

According to the current codebase, CBAM-related logic is integrated in:

- `Models/attention_cbamLL.py`
- `Models/Backbone/dla34.py`
- `Models/Backbone/resnet.py`

This makes the repository suitable for research on:

- attention-enhanced lane detection,
- backbone replacement,
- feature refinement in polar-representation based detection.

---

## 11. Currently Included Configurations

| Config File | Dataset | Backbone |
|---|---|---|
| `polarrcnn_culane_dla34.py` | CULane | DLA34 |
| `polarrcnn_tusimple_r18.py` | TuSimple | ResNet18 |
| `polarrcnn_curvelanes_dla34.py` | CurveLanes | DLA34 |

---

## 12. Important Notes Before Running

1. Please make sure the dataset path in each config file is correct.
2. Please compile `ops/NMSOPS` before training or testing.
3. Multi-GPU training requires the total batch size in the config to be divisible by the number of GPUs.
4. The code currently uses `num_workers=0` in dataloaders for better stability, especially on Windows.
5. If you release this repository publicly, it is recommended to clean unnecessary files such as:
   - `.idea/`
   - `__pycache__/`
   - local test outputs
   - temporary checkpoints

---


## 13. Acknowledgement

This project is developed based on the Polar R-CNN framework and further adapted for personal research.
Thanks to the open-source community for providing the baseline implementation and related resources.
