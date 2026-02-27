# WARNING: template code, may need edits
"""Inference script for cat detection."""

import argparse
from pathlib import Path
import yaml
import time

import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from model import create_model
from dataset import get_transforms
from utils import load_checkpoint, get_device


class CatInference:
    """Simple inference class for cat detection."""
    
    def __init__(self, model_path, config, device='cuda'):
        """
        Args:
            model_path: Path to trained model checkpoint
            config: Configuration dictionary
            device: Device to run inference on
        """
        self.device = get_device(device)
        self.config = config
        self.class_names = ['Not Cat', 'Cat']
        
        # Load model
        print(f"Loading model from {model_path}...")
        self.model = create_model(
            num_classes=config['model']['num_classes'],
            dropout=config['model']['dropout']
        ).to(self.device)
        
        checkpoint = load_checkpoint(model_path, self.model)
        self.model.eval()
        
        print(f"Model loaded successfully!")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  Accuracy: {checkpoint['accuracy']:.2f}%")
        
        # Load transforms
        self.transform = get_transforms(
            image_size=config['data']['image_size'],
            is_train=False
        )
    
    def predict_single(self, image_path, return_probs=False):
        """Predict on a single image.
        
        Args:
            image_path: Path to image file
            return_probs: Whether to return probabilities
            
        Returns:
            Prediction result (class name and optionally probabilities)
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Inference
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = outputs.argmax(1).item()
        
        inference_time = (time.time() - start_time) * 1000  # ms
        
        result = {
            'class': self.class_names[predicted_class],
            'class_id': predicted_class,
            'confidence': probabilities[0][predicted_class].item(),
            'inference_time_ms': inference_time
        }
        
        if return_probs:
            result['probabilities'] = {
                name: prob.item() 
                for name, prob in zip(self.class_names, probabilities[0])
            }
        
        return result, image
    
    def visualize_prediction(self, image_path, save_path=None):
        """Visualize prediction on image.
        
        Args:
            image_path: Path to image file
            save_path: Optional path to save visualization
        """
        result, image = self.predict_single(image_path, return_probs=True)
        
        # Create visualization
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.imshow(image)
        ax.axis('off')
        
        # Add prediction text
        text = f"Prediction: {result['class']}\n"
        text += f"Confidence: {result['confidence']:.2%}\n"
        text += f"Time: {result['inference_time_ms']:.1f}ms"
        
        # Color based on prediction
        color = 'green' if result['class'] == 'Cat' else 'red'
        
        ax.text(10, 30, text, fontsize=14, color='white',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        return result


def main():
    parser = argparse.ArgumentParser(description='Run inference on cat images')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--model-path', type=str, default='models/best_model.pth',
                        help='Path to trained model')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save output visualization')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check if image exists
    if not Path(args.image).exists():
        print(f"Error: Image not found at {args.image}")
        return
    
    # Create inference object
    inferencer = CatInference(args.model_path, config, device=args.device)
    
    # Run inference
    print(f"\nRunning inference on {args.image}...")
    
    if args.output:
        result = inferencer.visualize_prediction(args.image, args.output)
    else:
        result, _ = inferencer.predict_single(args.image, return_probs=True)
    
    # Print results
    print("\nResults:")
    print("=" * 50)
    print(f"Prediction: {result['class']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Inference Time: {result['inference_time_ms']:.1f}ms")
    
    if 'probabilities' in result:
        print("\nClass Probabilities:")
        for class_name, prob in result['probabilities'].items():
            print(f"  {class_name}: {prob:.2%}")


if __name__ == "__main__":
    main()
