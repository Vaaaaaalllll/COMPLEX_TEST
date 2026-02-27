<!-- WARNING: template code, may need edits -->
# Complete Guide to Cat Classifier

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Dataset Preparation](#dataset-preparation)
4. [Training](#training)
5. [Testing](#testing)
6. [Inference](#inference)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Usage](#advanced-usage)

## Introduction

This project implements a simple yet effective cat image classifier using a custom Convolutional Neural Network (CNN). The model is designed to run on modest hardware (4-8GB GPU) and is built without heavy frameworks for educational purposes.

### What This Project Does
- Trains a neural network to recognize cats in images
- Provides easy-to-use training, testing, and inference scripts
- Optimized for learning and understanding deep learning concepts

### What You Need
- Python 3.8 or higher
- NVIDIA GPU with 4-8GB VRAM (or CPU, but slower)
- 32GB RAM recommended
- 10GB+ storage for dataset

## Installation

### Step 1: Clone or Download the Project
```bash
cd COMPLEX_TEST
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Linux/Mac:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

You should see PyTorch version and CUDA availability.

## Dataset Preparation

### Option 1: Use Existing Dataset

If you have a cat dataset (like Kaggle's Dogs vs Cats):

1. **Organize your images:**
```
data/
  train/
    cat/           <- Put 80% of cat images here
    not_cat/       <- Put 80% of non-cat images here
  test/
    cat/           <- Put 20% of cat images here
    not_cat/       <- Put 20% of non-cat images here
```

2. **Run the setup script:**
```bash
python src/download_data.py
```

3. **Validate your dataset:**
```bash
python src/download_data.py --validate
```

### Option 2: Create Your Own Dataset

1. Collect at least 500 images per class (cat and not_cat)
2. Ensure images are:
   - Clear and well-lit
   - Various angles and backgrounds
   - Different cat breeds (if applicable)
   - Mix of colors and sizes

3. Split into training (80%) and testing (20%)

### Dataset Tips
- **More data = better results**
- Aim for 1000+ images per class for good performance
- Balance your classes (equal number of cat and not_cat images)
- Include diverse examples

## Training

### Basic Training

Simplest way to start training:

```bash
python src/train.py
```

This uses default settings from `config.yaml`.

### Custom Training

Override settings with command-line arguments:

```bash
python src/train.py --epochs 100 --batch-size 16 --lr 0.0001
```

### What Happens During Training

1. **Data Loading**: Images are loaded and augmented
2. **Model Creation**: Neural network is initialized
3. **Training Loop**: Model learns from images
4. **Validation**: Model is tested on validation set
5. **Checkpointing**: Best model is saved

### Monitoring Training

You'll see output like:
```
Epoch 1/50:
  Train Loss: 0.6234, Train Acc: 65.23%
  Val Loss: 0.5123, Val Acc: 72.45%
  7 Saved best model (acc: 72.45%)
```

### Training Tips

**If accuracy is low:**
- Train for more epochs
- Add more diverse training data
- Reduce learning rate

**If training is slow:**
- Reduce batch size
- Use fewer workers in config
- Ensure GPU is being used

**If GPU runs out of memory:**
- Reduce batch size: `--batch-size 16` or `--batch-size 8`
- Reduce image size in `config.yaml`

### Configuration File

Edit `config.yaml` to change:
- Model architecture
- Training hyperparameters
- Data augmentation
- Hardware settings

## Testing

### Basic Testing

Test your trained model:

```bash
python src/test.py
```

### With Plots

Generate evaluation plots:

```bash
python src/test.py --save-plots
```

This creates:
- `outputs/confusion_matrix.png` - Shows classification errors
- `outputs/roc_curve.png` - Shows model performance curve

### Understanding Results

**Confusion Matrix:**
- Shows true vs predicted labels
- Diagonal = correct predictions
- Off-diagonal = errors

**Classification Report:**
- Precision: How many predicted cats are actually cats
- Recall: How many actual cats were found
- F1-Score: Balance between precision and recall

**Good Results:**
- Accuracy > 90%
- Precision and Recall > 85%
- Balanced performance on both classes

## Inference

### Single Image Prediction

Predict on a single image:

```bash
python src/inference.py --image path/to/cat.jpg
```

### With Visualization

Save prediction with visualization:

```bash
python src/inference.py --image path/to/cat.jpg --output result.png
```

### Using in Your Code

```python
from src.inference import CatInference
import yaml

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create inferencer
inferencer = CatInference('models/best_model.pth', config)

# Predict
result, image = inferencer.predict_single('cat.jpg', return_probs=True)

print(f"Prediction: {result['class']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Batch Inference

For multiple images:

```python
import os
from pathlib import Path

image_dir = Path('path/to/images')
for img_path in image_dir.glob('*.jpg'):
    result, _ = inferencer.predict_single(str(img_path))
    print(f"{img_path.name}: {result['class']} ({result['confidence']:.2%})")
```

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```
RuntimeError: CUDA out of memory
```
Solution:
- Reduce batch size: `--batch-size 8`
- Close other GPU applications
- Reduce image size in config

**2. No Images Found**
```
RuntimeError: No images found in data/train
```
Solution:
- Check directory structure
- Ensure images have correct extensions (.jpg, .png)
- Run `python src/download_data.py --validate`

**3. Low Accuracy**
- Add more training data
- Train for more epochs
- Check data quality
- Ensure classes are balanced

**4. Training is Slow**
- Verify GPU is being used
- Reduce number of workers
- Use smaller image size

**5. Model Not Learning**
- Check learning rate (try 0.001 or 0.0001)
- Verify data is correct
- Check for data augmentation issues

## Advanced Usage

### Custom Model Architecture

Edit `src/model.py` to modify the network:

```python
# Add more layers
self.conv6 = ConvBlock(512, 1024, kernel_size=3, stride=2, padding=1)

# Change dropout
self.dropout = nn.Dropout(0.5)
```

### Custom Augmentation

Edit `src/dataset.py` to add augmentations:

```python
transforms.RandomRotation(30),  # More rotation
transforms.RandomGrayscale(p=0.1),  # Random grayscale
```

### Learning Rate Scheduling

In `src/train.py`, try different schedulers:

```python
# Step decay
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Exponential decay
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
```

### Transfer Learning

Use a pre-trained model backbone:

```python
import torchvision.models as models

# Load pre-trained ResNet
backbone = models.resnet18(pretrained=True)
# Modify final layer
backbone.fc = nn.Linear(512, 2)
```

### Hyperparameter Tuning

Key parameters to experiment with:
- Learning rate: [0.0001, 0.001, 0.01]
- Batch size: [8, 16, 32, 64]
- Dropout: [0.2, 0.3, 0.5]
- Weight decay: [0.0001, 0.001]

### Export for Production

Export model to ONNX for deployment:

```python
import torch.onnx

# Load model
model = create_model()
model.load_state_dict(torch.load('models/best_model.pth')['model_state_dict'])
model.eval()

# Export
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, 'cat_classifier.onnx')
```

## Performance Optimization

### For Faster Training
1. Use mixed precision training (enabled by default)
2. Increase batch size if GPU allows
3. Use more data workers
4. Pin memory for faster data transfer

### For Better Accuracy
1. More training data
2. Longer training (more epochs)
3. Better data augmentation
4. Ensemble multiple models

### For Smaller Model
1. Reduce number of filters
2. Use fewer layers
3. Apply pruning techniques
4. Quantize the model

## Best Practices

1. **Always validate your dataset first**
2. **Start with default settings**
3. **Monitor training closely**
4. **Save multiple checkpoints**
5. **Test on diverse images**
6. **Document your experiments**
7. **Use version control (git)**
8. **Keep backups of trained models**

## Next Steps

1. **Improve the model**: Try different architectures
2. **Collect more data**: Better data = better results
3. **Deploy the model**: Create a web app or API
4. **Extend functionality**: Multi-class classification
5. **Optimize performance**: Make it faster and smaller

## Support

If you encounter issues:
1. Check this guide
2. Review error messages carefully
3. Validate your dataset
4. Check GPU memory usage
5. Try with smaller batch size

## Conclusion

You now have a complete cat classifier! Experiment, learn, and have fun with deep learning.

Remember: Deep learning is iterative. Don't expect perfect results immediately. Keep experimenting and improving!
