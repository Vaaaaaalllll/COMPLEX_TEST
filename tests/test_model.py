# WARNING: template code, may need edits
"""Unit tests for model module."""

import unittest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import CatClassifier, create_model


class TestCatClassifier(unittest.TestCase):
    """Test cases for CatClassifier model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = CatClassifier(num_classes=2, dropout=0.3)
        self.batch_size = 4
        self.input_tensor = torch.randn(self.batch_size, 3, 224, 224)
    
    def test_model_creation(self):
        """Test model can be created."""
        self.assertIsInstance(self.model, CatClassifier)
    
    def test_forward_pass(self):
        """Test forward pass produces correct output shape."""
        output = self.model(self.input_tensor)
        expected_shape = (self.batch_size, 2)
        self.assertEqual(output.shape, expected_shape)
    
    def test_output_range(self):
        """Test output values are reasonable (not NaN or Inf)."""
        output = self.model(self.input_tensor)
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
    
    def test_parameter_count(self):
        """Test model has expected number of parameters."""
        num_params = self.model.get_num_params()
        # Should be around 2-3 million parameters
        self.assertGreater(num_params, 2_000_000)
        self.assertLess(num_params, 4_000_000)
    
    def test_gradient_flow(self):
        """Test gradients can flow through the model."""
        output = self.model(self.input_tensor)
        loss = output.sum()
        loss.backward()
        
        # Check some parameters have gradients
        has_grad = any(p.grad is not None for p in self.model.parameters())
        self.assertTrue(has_grad)
    
    def test_eval_mode(self):
        """Test model behavior in eval mode."""
        self.model.eval()
        with torch.no_grad():
            output1 = self.model(self.input_tensor)
            output2 = self.model(self.input_tensor)
        
        # Outputs should be identical in eval mode
        self.assertTrue(torch.allclose(output1, output2))
    
    def test_different_batch_sizes(self):
        """Test model works with different batch sizes."""
        for batch_size in [1, 8, 16, 32]:
            x = torch.randn(batch_size, 3, 224, 224)
            output = self.model(x)
            self.assertEqual(output.shape[0], batch_size)


class TestModelFactory(unittest.TestCase):
    """Test cases for model factory function."""
    
    def test_create_model(self):
        """Test create_model function."""
        model = create_model(num_classes=2, dropout=0.3)
        self.assertIsInstance(model, CatClassifier)
    
    def test_custom_classes(self):
        """Test model with different number of classes."""
        model = create_model(num_classes=10)
        x = torch.randn(1, 3, 224, 224)
        output = model(x)
        self.assertEqual(output.shape[1], 10)


if __name__ == '__main__':
    unittest.main()
