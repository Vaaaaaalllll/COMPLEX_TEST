# WARNING: template code, may need edits
"""Dataset utilities for loading and augmenting cat images."""

import os
from pathlib import Path
from typing import Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np


class CatDataset(Dataset):
    """Custom dataset for cat image classification."""
    
    def __init__(self, root_dir: str, transform=None, is_train=True):
        """
        Args:
            root_dir: Root directory with 'cat' and 'not_cat' subdirectories
            transform: Optional transform to be applied on images
            is_train: Whether this is training data (for augmentation)
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.is_train = is_train
        self.classes = ['not_cat', 'cat']  # 0: not_cat, 1: cat
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Load all image paths and labels
        self.samples = self._load_samples()
        
        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {root_dir}")
        
        print(f"Loaded {len(self.samples)} images from {root_dir}")
        self._print_class_distribution()
    
    def _load_samples(self):
        """Load all image paths and their labels."""
        samples = []
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                print(f"Warning: {class_dir} does not exist")
                continue
            
            for img_path in class_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    samples.append((str(img_path), self.class_to_idx[class_name]))
        
        return samples
    
    def _print_class_distribution(self):
        """Print dataset statistics."""
        class_counts = {cls: 0 for cls in self.classes}
        for _, label in self.samples:
            class_counts[self.classes[label]] += 1
        
        print("Class distribution:")
        for cls, count in class_counts.items():
            print(f"  {cls}: {count} images")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a black image if loading fails
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(image_size=224, is_train=True):
    """Get image transforms for training or testing.
    
    Args:
        image_size: Target image size
        is_train: Whether to apply training augmentations
        
    Returns:
        torchvision.transforms.Compose object
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def create_dataloaders(train_dir: str, test_dir: str, batch_size=32, 
                       num_workers=4, image_size=224):
    """Create train and test dataloaders.
    
    Args:
        train_dir: Path to training data
        test_dir: Path to test data
        batch_size: Batch size
        num_workers: Number of worker processes
        image_size: Image size
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Create datasets
    train_dataset = CatDataset(
        train_dir,
        transform=get_transforms(image_size, is_train=True)
    )
    
    test_dataset = CatDataset(
        test_dir,
        transform=get_transforms(image_size, is_train=False)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


if __name__ == "__main__":
    # Test dataset loading
    train_transform = get_transforms(224, is_train=True)
    test_transform = get_transforms(224, is_train=False)
    
    print("Training transforms:")
    print(train_transform)
    print("\nTest transforms:")
    print(test_transform)
