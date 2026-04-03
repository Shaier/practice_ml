import torch
import torch.nn as nn

class MixerBlock(nn.Module):
    def __init__(self, n_patches, n_channels, token_dim, channel_dim):
        super().__init__()
        
        # Token-mixing MLP
        self.ln1 = nn.LayerNorm(n_channels)
        self.token_mixer = nn.Sequential(
            nn.Linear(n_patches, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, n_patches),
        )
        
        # Channel-mixing MLP
        self.ln2 = nn.LayerNorm(n_channels)
        self.channel_mixer = nn.Sequential(
            nn.Linear(n_channels, channel_dim),
            nn.GELU(),
            nn.Linear(channel_dim, n_channels)
        )

    def forward(self, x):
        # 1. Token Mixing (Spatial communication)
        # Input shape: (Batch, Patches, Channels)
        residual = x
        x = self.ln1(x)
        x = x.transpose(1, 2)              # Shape: (Batch, Channels, Patches)
        x = self.token_mixer(x)
        x = x.transpose(1, 2)              # Shape: (Batch, Patches, Channels)
        x = x + residual
        
        # 2. Channel Mixing (Feature communication)
        residual = x
        x = self.ln2(x)
        x = self.channel_mixer(x)          # Linear operates on the last dim (Channels)
        x = x + residual
        
        return x

class MLPMixer(nn.Module):
    def __init__(self, in_channels, dim, num_classes, patch_size, image_size, depth, token_dim, channel_dim):
        super().__init__()
        
        n_patches = (image_size // patch_size) ** 2
        
        # Patch Embedding using a Conv2d
        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        
        self.mixer_blocks = nn.ModuleList([
            MixerBlock(n_patches, dim, token_dim, channel_dim) for _ in range(depth)
        ])
        
        self.ln_out = nn.LayerNorm(dim)
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.patch_embed(x)             # (B, dim, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)    # (B, Patches, dim)
        
        for block in self.mixer_blocks:
            x = block(x)
            
        x = self.ln_out(x)
        x = x.mean(dim=1)                   # Global Average Pooling
        return self.mlp_head(x)