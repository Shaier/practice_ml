import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Batch Normalization
# ---------------------------------------------------------------------------

class BatchNorm(nn.Module):
    """Batch Normalization (Ioffe & Szegedy 2015).
    Normalizes across the batch dimension per feature channel.
    Uses batch statistics (mean/var) during training, running estimates during eval.
    Key difference from LayerNorm: statistics computed over the batch, not per-token.
    Breaks with small batch sizes; doesn't work well for sequence models.
    Widely used in CNNs (ResNet etc).
    Input shape: (B, C) for linear layers, or (B, C, *) for conv layers.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            reduce_dims = (0,)
            shape = (1, -1)
        else:
            reduce_dims = (0,) + tuple(range(2, x.dim()))
            shape = (1, -1) + (1,) * (x.dim() - 2)

        if self.training:
            mean = x.mean(dim=reduce_dims)
            var = x.var(dim=reduce_dims, unbiased=False)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.detach()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.detach()
        else:
            mean = self.running_mean
            var = self.running_var

        x_norm = (x - mean.view(shape)) / torch.sqrt(var.view(shape) + self.eps)
        return self.gamma.view(shape) * x_norm + self.beta.view(shape)


# ---------------------------------------------------------------------------
# Manual LayerNorm
# ---------------------------------------------------------------------------

class LayerNorm(nn.Module):
    """Layer normalization (Ba et al. 2016).
    Normalizes across the last dimension (d_model) independently per token.
    Formula: (x - mean) / sqrt(var + eps) * gamma + beta
    gamma and beta are learned per-feature scale and shift parameters.
    """

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich 2019).
    Simpler than LayerNorm: no mean subtraction, no beta shift.
    Only normalizes by RMS and rescales with learned gamma.
    Formula: x / rms(x) * gamma, where rms(x) = sqrt(mean(x^2) + eps)
    Used in: LLaMA, Mistral, T5, PaLM — now preferred over LayerNorm.
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.gamma
