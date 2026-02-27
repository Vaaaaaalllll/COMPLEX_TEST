<!-- WARNING: template code, may need edits -->
# COMPLEX_TEST - Cat Image Classifier

A simple, educational image classification model for detecting cats in images. Built from scratch without heavy frameworks, optimized for 4-8GB GPU.

## Features
- Custom CNN architecture optimized for cat detection
- Training pipeline with data augmentation
- Testing/validation module
- Simple inference interface
- Runs on modest hardware (4-8GB GPU, 32GB RAM)

## Quick Start

### Prerequisites
```bash
python >= 3.8
nvidia-cuda-toolkit (for GPU support)
```

### Installation
```bash
pip install -r requirements.txt
```

### Usage

#### 1. Prepare Your Data
Organize your images:
```
data/
  train/
    cat/
    not_cat/
  test/
    cat/
    not_cat/
```

#### 2. Train the Model
```bash
python src/train.py --epochs 50 --batch-size 32
```

#### 3. Test the Model
```bash
python src/test.py --model-path models/best_model.pth
```

#### 4. Run Inference
```bash
python src/inference.py --image path/to/cat.jpg
```

## Project Structure
```
COMPLEX_TEST/
├── src/              # Source code
├── data/             # Dataset directory
├── models/           # Saved models
├── docs/             # Documentation
├── tests/            # Unit tests
└── outputs/          # Inference results
```

## Hardware Requirements
- GPU: 4-8GB VRAM (NVIDIA recommended)
- RAM: 32GB
- Storage: 10GB+ for dataset

## License
MIT
