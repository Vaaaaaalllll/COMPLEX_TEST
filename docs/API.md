<!-- WARNING: template code, may need edits -->
# API Reference

## Module: `src.model`

### `CatClassifier`

Main neural network model for cat classification.

**Class Definition:**
```python
class CatClassifier(nn.Module):
    def __init__(self, num_classes=2, dropout=0.3)
```

**Parameters:**
- `num_classes` (int): Number of output classes. Default: 2
- `dropout` (float): Dropout probability. Default: 0.3

**Methods:**

#### `forward(x)`
Forward pass through the network.

**Args:**
- `x` (torch.Tensor): Input tensor of shape (batch_size, 3, 224, 224)

**Returns:**
- torch.Tensor: Output logits of shape (batch_size, num_classes)

**Example:**
```python
model = CatClassifier()
x = torch.randn(1, 3, 224, 224)
output = model(x)
```

#### `get_num_params()`
Get total number of parameters.

**Returns:**
- int: Total parameter count

---

### `create_model()`

Factory function to create a model instance.

**Function Signature:**
```python
def create_model(num_classes=2, dropout=0.3)
```

**Parameters:**
- `num_classes` (int): Number of output classes
- `dropout` (float): Dropout rate

**Returns:**
- CatClassifier: Model instance

**Example:**
```python
from src.model import create_model

model = create_model(num_classes=2, dropout=0.3)
print(f"Parameters: {model.get_num_params():,}")
```

---

## Module: `src.dataset`

### `CatDataset`

Custom PyTorch dataset for loading cat images.

**Class Definition:**
```python
class CatDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_train=True)
```

**Parameters:**
- `root_dir` (str): Root directory containing 'cat' and 'not_cat' subdirectories
- `transform` (callable, optional): Transform to apply to images
- `is_train` (bool): Whether this is training data

**Methods:**

#### `__len__()`
Get dataset size.

**Returns:**
- int: Number of samples

#### `__getitem__(idx)`
Get a single sample.

**Args:**
- `idx` (int): Sample index

**Returns:**
- tuple: (image, label)

**Example:**
```python
from src.dataset import CatDataset, get_transforms

transform = get_transforms(224, is_train=True)
dataset = CatDataset('data/train', transform=transform)

image, label = dataset[0]
print(f"Image shape: {image.shape}, Label: {label}")
```

---

### `get_transforms()`

Get image transformation pipeline.

**Function Signature:**
```python
def get_transforms(image_size=224, is_train=True)
```

**Parameters:**
- `image_size` (int): Target image size
- `is_train` (bool): Whether to apply training augmentations

**Returns:**
- torchvision.transforms.Compose: Transform pipeline

**Example:**
```python
train_transform = get_transforms(224, is_train=True)
test_transform = get_transforms(224, is_train=False)
```

---

### `create_dataloaders()`

Create training and testing data loaders.

**Function Signature:**
```python
def create_dataloaders(train_dir, test_dir, batch_size=32, 
                       num_workers=4, image_size=224)
```

**Parameters:**
- `train_dir` (str): Path to training data
- `test_dir` (str): Path to test data
- `batch_size` (int): Batch size. Default: 32
- `num_workers` (int): Number of data loading workers. Default: 4
- `image_size` (int): Image size. Default: 224

**Returns:**
- tuple: (train_loader, test_loader)

**Example:**
```python
train_loader, test_loader = create_dataloaders(
    train_dir='data/train',
    test_dir='data/test',
    batch_size=32
)

for images, labels in train_loader:
    print(f"Batch shape: {images.shape}")
    break
```

---

## Module: `src.inference`

### `CatInference`

Inference class for cat detection.

**Class Definition:**
```python
class CatInference:
    def __init__(self, model_path, config, device='cuda')
```

**Parameters:**
- `model_path` (str): Path to trained model checkpoint
- `config` (dict): Configuration dictionary
- `device` (str): Device to use ('cuda' or 'cpu')

**Methods:**

#### `predict_single(image_path, return_probs=False)`
Predict on a single image.

**Args:**
- `image_path` (str): Path to image file
- `return_probs` (bool): Whether to return class probabilities

**Returns:**
- tuple: (result_dict, PIL.Image)

**Result Dictionary:**
```python
{
    'class': str,              # Predicted class name
    'class_id': int,           # Class ID (0 or 1)
    'confidence': float,       # Confidence score
    'inference_time_ms': float,# Inference time in milliseconds
    'probabilities': dict      # Class probabilities (if return_probs=True)
}
```

**Example:**
```python
from src.inference import CatInference
import yaml

with open('config.yaml') as f:
    config = yaml.safe_load(f)

inferencer = CatInference('models/best_model.pth', config)
result, image = inferencer.predict_single('cat.jpg', return_probs=True)

print(f"Prediction: {result['class']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Time: {result['inference_time_ms']:.1f}ms")
```

#### `visualize_prediction(image_path, save_path=None)`
Visualize prediction on image.

**Args:**
- `image_path` (str): Path to image file
- `save_path` (str, optional): Path to save visualization

**Returns:**
- dict: Prediction result

---

## Module: `src.utils`

### `set_seed()`

Set random seeds for reproducibility.

**Function Signature:**
```python
def set_seed(seed=42)
```

**Parameters:**
- `seed` (int): Random seed value

**Example:**
```python
from src.utils import set_seed

set_seed(42)
```

---

### `get_device()`

Get the appropriate device for computation.

**Function Signature:**
```python
def get_device(device_name='cuda')
```

**Parameters:**
- `device_name` (str): 'cuda' or 'cpu'

**Returns:**
- torch.device: Device object

**Example:**
```python
device = get_device('cuda')
print(f"Using device: {device}")
```

---

### `save_checkpoint()`

Save model checkpoint.

**Function Signature:**
```python
def save_checkpoint(model, optimizer, epoch, accuracy, loss, filepath)
```

**Parameters:**
- `model` (nn.Module): Model to save
- `optimizer` (torch.optim.Optimizer): Optimizer state
- `epoch` (int): Current epoch
- `accuracy` (float): Current accuracy
- `loss` (float): Current loss
- `filepath` (str): Path to save checkpoint

**Example:**
```python
save_checkpoint(
    model, optimizer, epoch=10, 
    accuracy=92.5, loss=0.23,
    filepath='models/checkpoint.pth'
)
```

---

### `load_checkpoint()`

Load model checkpoint.

**Function Signature:**
```python
def load_checkpoint(filepath, model, optimizer=None)
```

**Parameters:**
- `filepath` (str): Path to checkpoint file
- `model` (nn.Module): Model to load weights into
- `optimizer` (torch.optim.Optimizer, optional): Optimizer to load state into

**Returns:**
- dict: Checkpoint dictionary

**Example:**
```python
checkpoint = load_checkpoint('models/best_model.pth', model)
print(f"Loaded epoch {checkpoint['epoch']}")
print(f"Accuracy: {checkpoint['accuracy']:.2f}%")
```

---

### `AverageMeter`

Utility class for tracking averages.

**Class Definition:**
```python
class AverageMeter:
    def __init__(self)
```

**Methods:**

#### `update(val, n=1)`
Update with new value.

**Args:**
- `val` (float): New value
- `n` (int): Number of samples

#### `reset()`
Reset all statistics.

**Example:**
```python
losses = AverageMeter()

for batch in data_loader:
    loss = compute_loss(batch)
    losses.update(loss.item(), batch_size)

print(f"Average loss: {losses.avg:.4f}")
```

---

### `EarlyStopping`

Early stopping utility.

**Class Definition:**
```python
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0)
```

**Parameters:**
- `patience` (int): Number of epochs to wait
- `verbose` (bool): Whether to print messages
- `delta` (float): Minimum improvement threshold

**Methods:**

#### `__call__(val_loss)`
Check if should stop.

**Args:**
- `val_loss` (float): Current validation loss

**Example:**
```python
early_stopping = EarlyStopping(patience=10)

for epoch in range(num_epochs):
    val_loss = validate()
    early_stopping(val_loss)
    
    if early_stopping.early_stop:
        print("Early stopping!")
        break
```

---

## Configuration File (`config.yaml`)

### Structure

```yaml
model:
  name: str                 # Model name
  input_size: int          # Input image size
  num_classes: int         # Number of classes
  dropout: float           # Dropout rate

train:
  epochs: int              # Number of epochs
  batch_size: int          # Batch size
  learning_rate: float     # Learning rate
  weight_decay: float      # Weight decay
  early_stopping_patience: int
  save_best_only: bool

data:
  train_dir: str           # Training data directory
  test_dir: str            # Test data directory
  image_size: int          # Image size
  num_workers: int         # Data loading workers

augmentation:
  horizontal_flip: bool
  rotation_range: int
  brightness: float
  contrast: float

hardware:
  device: str              # 'cuda' or 'cpu'
  mixed_precision: bool
  pin_memory: bool

paths:
  model_dir: str           # Model save directory
  output_dir: str          # Output directory
  log_dir: str             # Log directory
```

### Loading Configuration

```python
import yaml

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Access values
batch_size = config['train']['batch_size']
learning_rate = config['train']['learning_rate']
```

---

## Command Line Interface

### Training

```bash
python src/train.py [OPTIONS]

Options:
  --config PATH          Path to config file [default: config.yaml]
  --epochs INT          Number of epochs
  --batch-size INT      Batch size
  --lr FLOAT            Learning rate
```

### Testing

```bash
python src/test.py [OPTIONS]

Options:
  --config PATH          Path to config file [default: config.yaml]
  --model-path PATH     Path to trained model [default: models/best_model.pth]
  --save-plots          Save evaluation plots
```

### Inference

```bash
python src/inference.py [OPTIONS]

Options:
  --config PATH          Path to config file [default: config.yaml]
  --model-path PATH     Path to trained model [default: models/best_model.pth]
  --image PATH          Path to input image [required]
  --output PATH         Path to save visualization
  --device STR          Device to use [default: cuda]
```
