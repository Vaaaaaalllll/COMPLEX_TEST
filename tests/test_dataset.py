# WARNING: template code, may need edits
"""Unit tests for dataset module."""

import unittest
import torch
import sys
from pathlib import Path
import tempfile
import shutil
from PIL import Image
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import CatDataset, get_transforms


class TestTransforms(unittest.TestCase):
    """Test cases for image transforms."""
    
    def test_train_transforms(self):
        """Test training transforms."""
        transform = get_transforms(224, is_train=True)
        self.assertIsNotNone(transform)
    
    def test_test_transforms(self):
        """Test testing transforms."""
        transform = get_transforms(224, is_train=False)
        self.assertIsNotNone(transform)
    
    def test_transform_output_shape(self):
        """Test transform produces correct output shape."""
        transform = get_transforms(224, is_train=False)
        
        # Create dummy image
        img = Image.new('RGB', (300, 300), color='red')
        transformed = transform(img)
        
        self.assertEqual(transformed.shape, (3, 224, 224))
    
    def test_transform_normalization(self):
        """Test transform normalizes values."""
        transform = get_transforms(224, is_train=False)
        
        img = Image.new('RGB', (224, 224), color='white')
        transformed = transform(img)
        
        # Values should be normalized (not in [0, 1] range)
        self.assertNotEqual(transformed.max().item(), 1.0)


class TestCatDataset(unittest.TestCase):
    """Test cases for CatDataset."""
    
    def setUp(self):
        """Create temporary dataset for testing."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create directory structure
        for split in ['cat', 'not_cat']:
            Path(self.temp_dir, split).mkdir(parents=True)
            
            # Create dummy images
            for i in range(5):
                img = Image.new('RGB', (100, 100), color='red' if split == 'cat' else 'blue')
                img.save(Path(self.temp_dir, split, f'img_{i}.jpg'))
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)
    
    def test_dataset_creation(self):
        """Test dataset can be created."""
        transform = get_transforms(224, is_train=False)
        dataset = CatDataset(self.temp_dir, transform=transform)
        self.assertIsNotNone(dataset)
    
    def test_dataset_length(self):
        """Test dataset has correct length."""
        transform = get_transforms(224, is_train=False)
        dataset = CatDataset(self.temp_dir, transform=transform)
        self.assertEqual(len(dataset), 10)  # 5 cats + 5 not_cats
    
    def test_dataset_getitem(self):
        """Test getting item from dataset."""
        transform = get_transforms(224, is_train=False)
        dataset = CatDataset(self.temp_dir, transform=transform)
        
        image, label = dataset[0]
        
        self.assertIsInstance(image, torch.Tensor)
        self.assertIsInstance(label, int)
        self.assertEqual(image.shape, (3, 224, 224))
        self.assertIn(label, [0, 1])
    
    def test_class_balance(self):
        """Test dataset has both classes."""
        transform = get_transforms(224, is_train=False)
        dataset = CatDataset(self.temp_dir, transform=transform)
        
        labels = [dataset[i][1] for i in range(len(dataset))]
        unique_labels = set(labels)
        
        self.assertEqual(len(unique_labels), 2)
        self.assertIn(0, unique_labels)
        self.assertIn(1, unique_labels)


if __name__ == '__main__':
    unittest.main()
