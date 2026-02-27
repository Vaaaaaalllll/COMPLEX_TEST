# WARNING: template code, may need edits
"""Example: Running inference with the cat classifier."""

import sys
from pathlib import Path
import yaml
import torch
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import create_model
from src.dataset import get_transforms
from src.utils import load_checkpoint, get_device


def simple_inference_example():
    """Simple inference example without using CatInference class."""
    print("Simple Inference Example")
    print("=" * 60)
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup
    device = get_device('cuda')
    
    # Load model
    print("\nLoading model...")
    model = create_model(
        num_classes=config['model']['num_classes'],
        dropout=config['model']['dropout']
    ).to(device)
    
    # Check if model exists
    model_path = 'models/best_model.pth'
    if not Path(model_path).exists():
        print(f"\nError: Model not found at {model_path}")
        print("Please train a model first: python src/train.py")
        return
    
    checkpoint = load_checkpoint(model_path, model)
    model.eval()
    
    print(f"Model loaded (epoch {checkpoint['epoch']}, acc: {checkpoint['accuracy']:.2f}%)")
    
    # Load transform
    transform = get_transforms(
        image_size=config['data']['image_size'],
        is_train=False
    )
    
    # Example: Create a dummy image
    print("\nCreating dummy image for demonstration...")
    dummy_image = Image.new('RGB', (224, 224), color='orange')
    
    # Preprocess
    input_tensor = transform(dummy_image).unsqueeze(0).to(device)
    
    # Inference
    print("Running inference...")
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = outputs.argmax(1).item()
    
    # Results
    class_names = ['Not Cat', 'Cat']
    print("\nResults:")
    print(f"  Prediction: {class_names[predicted_class]}")
    print(f"  Confidence: {probabilities[0][predicted_class].item():.2%}")
    print("\nClass Probabilities:")
    for i, name in enumerate(class_names):
        print(f"  {name}: {probabilities[0][i].item():.2%}")
    
    print("\n" + "=" * 60)
    print("For real inference on your images, run:")
    print("  python src/inference.py --image path/to/your/image.jpg")


def batch_inference_example():
    """Example of batch inference."""
    print("\nBatch Inference Example")
    print("=" * 60)
    
    # This would process multiple images
    print("\nFor batch inference, you can:")
    print("1. Load multiple images into a batch tensor")
    print("2. Process them all at once")
    print("3. Get predictions for all images")
    
    print("\nExample code:")
    print("""
    images = []
    for img_path in image_paths:
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img)
        images.append(img_tensor)
    
    batch = torch.stack(images).to(device)
    
    with torch.no_grad():
        outputs = model(batch)
        predictions = outputs.argmax(1)
    """)


if __name__ == "__main__":
    simple_inference_example()
    batch_inference_example()
