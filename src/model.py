# WARNING: template code, may need edits
"""Efficient CNN model for cat image classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Convolutional block with BatchNorm and activation."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class CatClassifier(nn.Module):
    """Lightweight CNN for cat detection optimized for 4-8GB GPU.
    
    Architecture inspired by efficient nets but simplified for educational purposes.
    """
    
    def __init__(self, num_classes=2, dropout=0.3):
        super(CatClassifier, self).__init__()
        
        # Initial convolution
        self.conv1 = ConvBlock(3, 32, kernel_size=3, stride=2, padding=1)
        
        # Feature extraction blocks
        self.conv2 = ConvBlock(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = ConvBlock(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = ConvBlock(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv5 = ConvBlock(256, 512, kernel_size=3, stride=2, padding=1)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize network weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        x = self.conv1(x)  # 112x112
        x = self.conv2(x)  # 56x56
        x = self.conv3(x)  # 28x28
        x = self.conv4(x)  # 14x14
        x = self.conv5(x)  # 7x7
        
        x = self.global_pool(x)  # 1x1
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
    
    def get_num_params(self):
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())


def create_model(num_classes=2, dropout=0.3):
    """Factory function to create model.
    
    Args:
        num_classes: Number of output classes
        dropout: Dropout rate
        
    Returns:
        CatClassifier model instance
    """
    model = CatClassifier(num_classes=num_classes, dropout=dropout)
    print(f"Model created with {model.get_num_params():,} parameters")
    return model


if __name__ == "__main__":
    # Test model
    model = create_model()
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
