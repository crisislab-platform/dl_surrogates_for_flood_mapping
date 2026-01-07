# Deep Learning for Flood Inundation Forecasting

A comprehensive framework for benchmarking deep leaning mdoels for flood inundation extent and depth using modelling. 

## 🌊 Overview

 The framework supports multiple model architectures and provides extensive evaluation metrics including TOPSIS-based multi-criteria decision analysis for model comparison.


## 📋 Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Data Generation](#data-generation)
- [Training](#training)
- [Evaluation](#evaluation)

## 🔧 Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 18.04+ recommended)
- **GPU**: CUDA-capable GPU with 96GB+ VRAM for reproducibility
- **RAM**: 16GB+ recommended
- **Storage**: 100GB+ for data and models

### Software Dependencies
- Python 3.8 or higher
- CUDA Toolkit 11.0+
- LISFLOOD-FP version 8.2 (for simulation data generation)
- GNU Parallel (for batch training)
- Conda (for environment management)

## 📦 Installation

1. **Clone the repository**
```bash
git clone <https://github.com/crisislab-platform/dl_surrogates_for_flood_mapping/>
```

2. **Create virtual environment**
```bash
conda create -n carlisle_env python=3.8 -y --file environment.yml
conda activate carlisle_env
```

3. **Install LISFLOOD-FP** (optional, for data generation)
```bash
# Follow LISFLOOD-FP installation guide
# https://www.seamlesswave.com/LISFLOOD-FP.html
```

4. **Configure paths**
```python
# Edit modules/lib/constants.py to set paths to your directories and files
PROJECT_ROOT = "/path/to/project/root"
DATA_DIR = "/path/to/data/directory"
OUTPUT_DIR = "/path/to/outputs"
DEM_FILE = "path/to/dem/file"
```

## 🚀 Quick Start

### 1. Generate Training Data

Run flood simulations using LISFLOOD-FP:

```bash
bash  run_scripts/simulation_data_generator.sh
```
After completion, simulated flood data will be stored in the specified DATA_DIR, under `simulation_data/`.

This generates 9 flood scenarios with varying intensities and durations.

### 2. Train a Single Model

Train a 1DCNN model with default parameters:

```bash
bash run_scripts/1dcnn/train_1dcnn_final.sh
```

The trained model and logs will be saved in the specified `RUN_DIR`

Other model training scripts are available in the `run_scripts` directory.

### 3. Run evaluation metrics computational efficiency analysis

```bash
bash  run_scripts/analyse_model_predictions.sh
```


## 📧 Contact

For questions or collaboration:
- Email: [malintha.mahindakumarage.1@uni.massey.ac.nz]

