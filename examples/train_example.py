# WARNING: template code, may need edits
"""Example: Training the cat classifier."""

import sys
from pathlib import Path
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import create_model
from src.dataset import create_dataloaders
from src.utils import get_device, set_seed
import torch.nn as nn
import torch.optim as optim


def main():
    """Simple training example."""
    print("Training Example")
    print("=" * 60)
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup
    set_seed(42)
    device = get_device('cuda')
    
    # Create dataloaders
    print("\nLoading data...")
    train_loader, test_loader = create_dataloaders(
        train_dir=config['data']['train_dir'],
        test_dir=config['data']['test_dir'],
        batch_size=config['train']['batch_size'],
        num_workers=config['data']['num_workers'],
        image_size=config['data']['image_size']
    )
    
    # Create model
    print("\nCreating model...")
    model = create_model(
        num_classes=config['model']['num_classes'],
        dropout=config['model']['dropout']
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['train']['learning_rate']
    )
    
    # Training loop (simplified)
    print("\nTraining for 2 epochs (example)...")
    for epoch in range(1, 3):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            if batch_idx >= 10:  # Only 10 batches for example
                break
            
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 5 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / min(10, len(train_loader))
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Accuracy: {accuracy:.2f}%")
        print("-" * 60)
    
    print("\nTraining example completed!")
    print("For full training, run: python src/train.py")


if __name__ == "__main__":
    main()
