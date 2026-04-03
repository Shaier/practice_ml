class SAMDecoder(nn.Module):
    def __init__(self, transformer_dim, num_multimask_outputs=3):
        super().__init__()
        self.transformer = nn.Transformer(d_model=transformer_dim, nhead=8, batch_first=True)
        # Learnable tokens for the mask and its quality (iou)
        self.mask_tokens = nn.Embedding(num_multimask_outputs + 1, transformer_dim)

    def forward(self, image_embedding, sparse_prompt_embeddings, dense_prompt_embeddings):
        # Concat prompt tokens with learnable mask tokens
        tokens = torch.cat([self.mask_tokens.weight, sparse_prompt_embeddings], dim=0)
        
        # Cross-attention: Prompts (queries) attend to Image (keys/values)
        # Simplified: SAM uses a specific two-way transformer block
        hs, src = self.transformer(tokens, image_embedding)
        
        # Upscale image_embedding and dot-product with tokens to get mask
        return hs # Returns predicted masks