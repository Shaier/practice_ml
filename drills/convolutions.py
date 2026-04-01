import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


# ---------------------------------------------------------------------------
# Residual block (ResNet-style, Conv2d)
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Basic residual block (He et al. 2016 — ResNet).
    Two 3x3 conv layers with BatchNorm and ReLU, plus a skip connection.
    If in_channels != out_channels or stride > 1, a 1x1 conv is used to match dimensions.
    The residual connection lets gradients flow directly, enabling very deep networks.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return F.relu(out + identity)


# ---------------------------------------------------------------------------
# Simple CNN image classifier (Conv2d)
# ---------------------------------------------------------------------------

class SimpleCNN(nn.Module):
    """Simple convolutional image classifier.
    Conv2d stack: each block = Conv → BatchNorm → ReLU → MaxPool.
    Global average pooling collapses spatial dims before the linear head.
    Output size formula: floor((input_size + 2*padding - kernel_size) / stride) + 1
    """

    def __init__(self, in_channels: int, num_classes: int, channels: List[int] = [32, 64, 128]):
        super().__init__()
        layers = []
        c_in = in_channels
        for c_out in channels:
            layers += [
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ]
            c_in = c_out
        self.features = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool2d(1)                                   # global average pool → (B, C, 1, 1)
        self.classifier = nn.Linear(channels[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x).flatten(1)                                          # (B, C)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Temporal Convolutional Network (Conv1d)
# ---------------------------------------------------------------------------

class TemporalBlock(nn.Module):
    """Single TCN block: two dilated causal Conv1d layers with residual connection.
    Causal: achieved by padding left only (no future leakage).
    Dilation exponentially increases receptive field with depth: 1, 2, 4, 8 ...
    Used for sequence modelling as a CNN alternative to RNNs.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float = 0.2):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation                         # left-only causal padding

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=self.padding)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation, padding=self.padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def _chomp(self, x: torch.Tensor) -> torch.Tensor:
        """Remove the right-side padding added by causal padding to keep output length == input length."""
        return x[:, :, : -self.padding] if self.padding > 0 else x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout(F.relu(self.bn1(self._chomp(self.conv1(x)))))
        out = self.dropout(F.relu(self.bn2(self._chomp(self.conv2(out)))))
        residual = x if self.downsample is None else self.downsample(x)
        return F.relu(out + residual)


class TCN(nn.Module):
    """Temporal Convolutional Network (Bai et al. 2018).
    Stack of TemporalBlocks with exponentially increasing dilation.
    Receptive field = 1 + 2 * (kernel_size - 1) * (2^num_layers - 1)
    Input:  (B, in_channels, T)
    Output: (B, out_channels, T)
    """

    def __init__(self, in_channels: int, out_channels: int, num_layers: int, kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            c_in = in_channels if i == 0 else out_channels
            layers.append(TemporalBlock(c_in, out_channels, kernel_size, dilation=dilation, dropout=dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
