# WARNING: template code, may need edits
"""Training script for cat classifier."""

import os
import argparse
from pathlib import Path
import yaml
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from model import create_model
from dataset import create_dataloaders
from utils import (
    set_seed, get_device, save_checkpoint, load_checkpoint,
    AverageMeter, EarlyStopping
)


def train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch):
    """Train for one epoch.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        scaler: Gradient scaler for mixed precision
        device: Device to train on
        epoch: Current epoch number
        
    Returns:
        Average loss and accuracy for the epoch
    """
    model.train()
    losses = AverageMeter()
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        # Mixed precision training
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Statistics
        losses.update(loss.item(), images.size(0))
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })
    
    accuracy = 100. * correct / total
    return losses.avg, accuracy


def validate(model, test_loader, criterion, device, epoch):
    """Validate the model.
    
    Args:
        model: Neural network model
        test_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        epoch: Current epoch number
        
    Returns:
        Average loss and accuracy
    """
    model.eval()
    losses = AverageMeter()
    correct = 0
    total = 0
    
    pbar = tqdm(test_loader, desc=f"Epoch {epoch} [Val]")
    
    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            losses.update(loss.item(), images.size(0))
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })
    
    accuracy = 100. * correct / total
    return losses.avg, accuracy


def train(config):
    """Main training function.
    
    Args:
        config: Configuration dictionary
    """
    # Setup
    set_seed(42)
    device = get_device(config['hardware']['device'])
    
    # Create directories
    model_dir = Path(config['paths']['model_dir'])
    model_dir.mkdir(exist_ok=True)
    
    # Create dataloaders
    print("Loading datasets...")
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
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['train']['learning_rate'],
        weight_decay=config['train']['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['train']['epochs']
    )
    
    # Mixed precision scaler
    scaler = GradScaler() if config['hardware']['mixed_precision'] else None
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config['train']['early_stopping_patience'],
        verbose=True
    )
    
    # Training loop
    print("\nStarting training...")
    best_acc = 0.0
    start_time = time.time()
    
    for epoch in range(1, config['train']['epochs'] + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, test_loader, criterion, device, epoch
        )
        
        # Update learning rate
        scheduler.step()
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{config['train']['epochs']}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(
                model, optimizer, epoch, val_acc, val_loss,
                model_dir / 'best_model.pth'
            )
            print(f"  7 Saved best model (acc: {best_acc:.2f}%)")
        
        # Save latest checkpoint
        save_checkpoint(
            model, optimizer, epoch, val_acc, val_loss,
            model_dir / 'latest_model.pth'
        )
        
        # Early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("\nEarly stopping triggered!")
            break
        
        print("-" * 60)
    
    # Training complete
    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time / 60:.2f} minutes")
    print(f"Best validation accuracy: {best_acc:.2f}%")


def main():
    parser = argparse.ArgumentParser(description='Train cat classifier')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (overrides config)')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments
    if args.epochs is not None:
        config['train']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['train']['batch_size'] = args.batch_size
    if args.lr is not None:
        config['train']['learning_rate'] = args.lr
    
    # Print configuration
    print("Configuration:")
    print(yaml.dump(config, default_flow_style=False))
    
    # Train
    train(config)


if __name__ == "__main__":
    main()
