"""Efficient models for nanopore signal segmentation."""

from typing import Dict, List, Tuple, Optional, Union, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
import logging

logger = logging.getLogger(__name__)

class ResidualBlock(nn.Module):
    """Residual block for efficient signal processing."""
    
    def __init__(self, channels: int, dilation: int = 1, dropout: float = 0.1):
        """Initialize residual block.
        
        Args:
            channels: Number of channels
            dilation: Dilation factor for convolutional layers
            dropout: Dropout probability
        """
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.norm2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, channels, sequence_length)
            
        Returns:
            Output tensor with same shape as input
        """
        residual = x
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        
        out = out + residual
        out = F.relu(out)
        
        return out


class SegmentationModel(nn.Module):
    """Efficient model for nanopore signal segmentation."""
    
    def __init__(
        self,
        input_channels: int = 1,
        hidden_channels: int = 64,
        num_residual_blocks: int = 8,
        dropout: float = 0.1,
    ):
        """Initialize segmentation model.
        
        Args:
            input_channels: Number of input channels (usually 1 for raw signal)
            hidden_channels: Number of hidden channels
            num_residual_blocks: Number of residual blocks
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        
        # Initial convolution to expand channels
        self.input_conv = nn.Conv1d(input_channels, hidden_channels, kernel_size=7, padding=3)
        self.input_norm = nn.BatchNorm1d(hidden_channels)
        
        # Residual blocks with increasing dilation for larger receptive field
        self.residual_blocks = nn.ModuleList()
        for i in range(num_residual_blocks):
            dilation = 2 ** (i % 4)  # Cycle through dilations: 1, 2, 4, 8
            self.residual_blocks.append(ResidualBlock(hidden_channels, dilation=dilation, dropout=dropout))
        
        # Final layers for boundary prediction
        self.final_conv1 = nn.Conv1d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1)
        self.final_norm1 = nn.BatchNorm1d(hidden_channels // 2)
        self.final_conv2 = nn.Conv1d(hidden_channels // 2, 1, kernel_size=1)
        
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_channels, sequence_length)
            
        Returns:
            Output tensor of shape (batch_size, 1, sequence_length) with boundary predictions
        """
        # Initial convolution
        x = self.input_conv(x)
        x = self.input_norm(x)
        x = F.relu(x)
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Final layers
        x = self.final_conv1(x)
        x = self.final_norm1(x)
        x = F.relu(x)
        x = self.final_conv2(x)
        
        # Sigmoid activation for boundary probability
        x = torch.sigmoid(x)
        
        return x
    
    def predict_boundaries(self, x: Tensor, threshold: float = 0.5) -> Tensor:
        """Predict segment boundaries with thresholding.
        
        Args:
            x: Input tensor of shape (batch_size, input_channels, sequence_length)
            threshold: Threshold for boundary detection
            
        Returns:
            Binary tensor with 1 at predicted boundaries
        """
        with torch.no_grad():
            predictions = self.forward(x)
            binary_predictions = (predictions > threshold).float()
            return binary_predictions


class EfficientUNet(nn.Module):
    """Efficient U-Net architecture for nanopore signal segmentation."""
    
    def __init__(
        self,
        input_channels: int = 1,
        init_features: int = 32,
        depth: int = 4,
        dropout: float = 0.1,
    ):
        """Initialize U-Net model.
        
        Args:
            input_channels: Number of input channels
            init_features: Initial number of features
            depth: Depth of the U-Net
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.init_features = init_features
        self.depth = depth
        
        # Encoder (downsampling path)
        self.encoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Initial block
        self.encoder_blocks.append(self._create_block(input_channels, init_features))
        
        # Rest of encoder blocks
        for i in range(1, depth):
            in_channels = init_features * (2 ** (i-1))
            out_channels = init_features * (2 ** i)
            self.encoder_blocks.append(self._create_block(in_channels, out_channels))
        
        # Middle block
        middle_channels = init_features * (2 ** depth)
        self.middle_block = self._create_block(
            init_features * (2 ** (depth-1)), middle_channels
        )
        
        # Decoder (upsampling path)
        self.decoder_blocks = nn.ModuleList()
        self.upconv_blocks = nn.ModuleList()
        
        for i in range(depth):
            out_channels = init_features * (2 ** (depth-i-1))
            in_channels = out_channels * 2  # Double due to skip connection
            
            # Use transposed convolution for upsampling
            self.upconv_blocks.append(
                nn.ConvTranspose1d(
                    middle_channels if i == 0 else out_channels * 2,
                    out_channels,
                    kernel_size=2,
                    stride=2
                )
            )
            
            self.decoder_blocks.append(self._create_block(in_channels, out_channels))
        
        # Final convolution
        self.final_conv = nn.Conv1d(init_features, 1, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        
    def _create_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Create a convolutional block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            
        Returns:
            Sequential block with two convolutions
        """
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_channels, sequence_length)
            
        Returns:
            Output tensor of shape (batch_size, 1, sequence_length)
        """
        # Store encoder outputs for skip connections
        encoder_outputs = []
        
        # Encoder path
        for block in self.encoder_blocks:
            x = block(x)
            encoder_outputs.append(x)
            x = self.pool(x)
        
        # Middle block
        x = self.middle_block(x)
        x = self.dropout(x)
        
        # Decoder path with skip connections
        for i, (upconv, block) in enumerate(zip(self.upconv_blocks, self.decoder_blocks)):
            # Upsample
            x = upconv(x)
            
            # Get skip connection from encoder
            skip = encoder_outputs[-(i+1)]
            
            # Handle case where upsampled feature map size doesn't exactly match skip connection
            if x.size(-1) != skip.size(-1):
                # Adjust size with padding or cropping
                diff = skip.size(-1) - x.size(-1)
                if diff > 0:
                    # Pad x
                    x = F.pad(x, (diff // 2, diff - diff // 2))
                else:
                    # Crop skip
                    diff = abs(diff)
                    skip = skip[:, :, diff // 2:skip.size(-1) - (diff - diff // 2)]
            
            # Concatenate skip connection
            x = torch.cat([x, skip], dim=1)
            
            # Apply convolution block
            x = block(x)
        
        # Final convolution and sigmoid activation
        x = self.final_conv(x)
        x = torch.sigmoid(x)
        
        return x
    
    def predict_boundaries(self, x: Tensor, threshold: float = 0.5) -> Tensor:
        """Predict segment boundaries with thresholding.
        
        Args:
            x: Input tensor of shape (batch_size, input_channels, sequence_length)
            threshold: Threshold for boundary detection
            
        Returns:
            Binary tensor with 1 at predicted boundaries
        """
        with torch.no_grad():
            predictions = self.forward(x)
            binary_predictions = (predictions > threshold).float()
            return binary_predictions