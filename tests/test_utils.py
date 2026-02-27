# WARNING: template code, may need edits
"""Unit tests for utils module."""

import unittest
import torch
import sys
from pathlib import Path
import tempfile
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import (
    set_seed, get_device, save_checkpoint, load_checkpoint,
    AverageMeter, EarlyStopping
)
from src.model import create_model


class TestSeedSetting(unittest.TestCase):
    """Test cases for seed setting."""
    
    def test_set_seed_reproducibility(self):
        """Test setting seed produces reproducible results."""
        set_seed(42)
        x1 = torch.randn(10)
        
        set_seed(42)
        x2 = torch.randn(10)
        
        self.assertTrue(torch.allclose(x1, x2))


class TestDeviceSelection(unittest.TestCase):
    """Test cases for device selection."""
    
    def test_get_device_cpu(self):
        """Test getting CPU device."""
        device = get_device('cpu')
        self.assertEqual(device.type, 'cpu')
    
    def test_get_device_cuda(self):
        """Test getting CUDA device if available."""
        device = get_device('cuda')
        if torch.cuda.is_available():
            self.assertEqual(device.type, 'cuda')
        else:
            self.assertEqual(device.type, 'cpu')


class TestCheckpointing(unittest.TestCase):
    """Test cases for model checkpointing."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = create_model()
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pth')
        self.temp_file.close()
    
    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_file.name):
            os.remove(self.temp_file.name)
    
    def test_save_checkpoint(self):
        """Test saving checkpoint."""
        save_checkpoint(
            self.model, self.optimizer,
            epoch=10, accuracy=85.5, loss=0.3,
            filepath=self.temp_file.name
        )
        self.assertTrue(os.path.exists(self.temp_file.name))
    
    def test_load_checkpoint(self):
        """Test loading checkpoint."""
        # Save checkpoint
        save_checkpoint(
            self.model, self.optimizer,
            epoch=10, accuracy=85.5, loss=0.3,
            filepath=self.temp_file.name
        )
        
        # Load checkpoint
        new_model = create_model()
        checkpoint = load_checkpoint(self.temp_file.name, new_model)
        
        self.assertEqual(checkpoint['epoch'], 10)
        self.assertEqual(checkpoint['accuracy'], 85.5)
        self.assertEqual(checkpoint['loss'], 0.3)
    
    def test_checkpoint_model_equivalence(self):
        """Test loaded model produces same outputs."""
        # Save checkpoint
        save_checkpoint(
            self.model, self.optimizer,
            epoch=10, accuracy=85.5, loss=0.3,
            filepath=self.temp_file.name
        )
        
        # Get output from original model
        self.model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output1 = self.model(x)
        
        # Load into new model
        new_model = create_model()
        load_checkpoint(self.temp_file.name, new_model)
        new_model.eval()
        
        with torch.no_grad():
            output2 = new_model(x)
        
        self.assertTrue(torch.allclose(output1, output2))


class TestAverageMeter(unittest.TestCase):
    """Test cases for AverageMeter."""
    
    def test_average_meter_single_update(self):
        """Test single update."""
        meter = AverageMeter()
        meter.update(5.0)
        self.assertEqual(meter.avg, 5.0)
    
    def test_average_meter_multiple_updates(self):
        """Test multiple updates."""
        meter = AverageMeter()
        meter.update(4.0)
        meter.update(6.0)
        self.assertEqual(meter.avg, 5.0)
    
    def test_average_meter_reset(self):
        """Test reset functionality."""
        meter = AverageMeter()
        meter.update(10.0)
        meter.reset()
        self.assertEqual(meter.avg, 0)
        self.assertEqual(meter.count, 0)


class TestEarlyStopping(unittest.TestCase):
    """Test cases for EarlyStopping."""
    
    def test_early_stopping_no_improvement(self):
        """Test early stopping triggers after no improvement."""
        early_stopping = EarlyStopping(patience=3)
        
        # Simulate no improvement
        for i in range(5):
            early_stopping(1.0)  # Same loss
        
        self.assertTrue(early_stopping.early_stop)
    
    def test_early_stopping_with_improvement(self):
        """Test early stopping doesn't trigger with improvement."""
        early_stopping = EarlyStopping(patience=3)
        
        # Simulate improvement
        for i in range(5):
            early_stopping(1.0 - i * 0.1)  # Decreasing loss
        
        self.assertFalse(early_stopping.early_stop)
    
    def test_early_stopping_counter_reset(self):
        """Test counter resets after improvement."""
        early_stopping = EarlyStopping(patience=3)
        
        # No improvement for 2 epochs
        early_stopping(1.0)
        early_stopping(1.0)
        self.assertEqual(early_stopping.counter, 2)
        
        # Improvement resets counter
        early_stopping(0.5)
        self.assertEqual(early_stopping.counter, 0)


if __name__ == '__main__':
    unittest.main()
