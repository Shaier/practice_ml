import torch
import torch.nn as nn

class ViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, 
                                                   dim_feedforward=mlp_dim, activation='gelu',
                                                   batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        # Patchify: (B, C, H, W) -> (B, Patches, Patch_Dim)
        p = self.patch_to_embedding.weight.shape[1]**0.5 # dummy logic for shape
        x = img.unfold(2, 16, 16).unfold(3, 16, 16).flatten(2).transpose(1, 2).flatten(2)
        
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape

        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        
        x = self.transformer(x)
        return self.mlp_head(x[:, 0]) # Classify based on CLS token