# WARNING: template code, may need edits
"""Testing and evaluation script for cat classifier."""

import argparse
from pathlib import Path
import yaml

import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc
)
import seaborn as sns

from model import create_model
from dataset import create_dataloaders
from utils import load_checkpoint, get_device


def test_model(model, test_loader, device):
    """Test the model and collect predictions.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to test on
        
    Returns:
        Tuple of (all_labels, all_predictions, all_probabilities)
    """
    model.eval()
    
    all_labels = []
    all_predictions = []
    all_probabilities = []
    
    correct = 0
    total = 0
    
    print("\nRunning evaluation...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities[:, 1].cpu().numpy())  # Probability of cat class
    
    accuracy = 100. * correct / total
    print(f"\nTest Accuracy: {accuracy:.2f}% ({correct}/{total})")
    
    return np.array(all_labels), np.array(all_predictions), np.array(all_probabilities)


def plot_confusion_matrix(labels, predictions, save_path):
    """Plot and save confusion matrix.
    
    Args:
        labels: True labels
        predictions: Predicted labels
        save_path: Path to save the plot
    """
    cm = confusion_matrix(labels, predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Cat', 'Cat'],
                yticklabels=['Not Cat', 'Cat'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def plot_roc_curve(labels, probabilities, save_path):
    """Plot and save ROC curve.
    
    Args:
        labels: True labels
        probabilities: Predicted probabilities
        save_path: Path to save the plot
    """
    fpr, tpr, _ = roc_curve(labels, probabilities)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"ROC curve saved to {save_path}")
    plt.close()


def print_classification_report(labels, predictions):
    """Print detailed classification report.
    
    Args:
        labels: True labels
        predictions: Predicted labels
    """
    print("\nClassification Report:")
    print("=" * 60)
    report = classification_report(
        labels, predictions,
        target_names=['Not Cat', 'Cat'],
        digits=4
    )
    print(report)


def main():
    parser = argparse.ArgumentParser(description='Test cat classifier')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--model-path', type=str, default='models/best_model.pth',
                        help='Path to trained model')
    parser.add_argument('--save-plots', action='store_true',
                        help='Save evaluation plots')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup
    device = get_device(config['hardware']['device'])
    output_dir = Path(config['paths']['output_dir'])
    output_dir.mkdir(exist_ok=True)
    
    # Load test data
    print("Loading test dataset...")
    _, test_loader = create_dataloaders(
        train_dir=config['data']['train_dir'],
        test_dir=config['data']['test_dir'],
        batch_size=config['train']['batch_size'],
        num_workers=config['data']['num_workers'],
        image_size=config['data']['image_size']
    )
    
    # Load model
    print(f"\nLoading model from {args.model_path}...")
    model = create_model(
        num_classes=config['model']['num_classes'],
        dropout=config['model']['dropout']
    ).to(device)
    
    checkpoint = load_checkpoint(args.model_path, model)
    print(f"Model loaded (epoch {checkpoint['epoch']}, acc: {checkpoint['accuracy']:.2f}%)")
    
    # Test model
    labels, predictions, probabilities = test_model(model, test_loader, device)
    
    # Print classification report
    print_classification_report(labels, predictions)
    
    # Save plots
    if args.save_plots:
        print("\nGenerating evaluation plots...")
        plot_confusion_matrix(labels, predictions, output_dir / 'confusion_matrix.png')
        plot_roc_curve(labels, probabilities, output_dir / 'roc_curve.png')
        print(f"\nPlots saved to {output_dir}")


if __name__ == "__main__":
    main()
