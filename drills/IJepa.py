class IJepa(nn.Module):
    def __init__(self, encoder, predictor):
        super().__init__()
        self.context_encoder = encoder  # Vision Transformer
        self.target_encoder = encoder   # EMA version of context_encoder
        self.predictor = predictor      # Small ViT/MLP

    def forward(self, img, context_mask, target_mask):
        # 1. Target: Encode full image with EMA encoder, then extract target patches
        with torch.no_grad():
            target_embs = self.target_encoder(img)
            target_patches = target_embs[target_mask] 

        # 2. Context: Encode only visible patches
        context_embs = self.context_encoder(img[context_mask])

        # 3. Predict: Try to predict target embeddings from context embeddings + pos info
        prediction = self.predictor(context_embs, target_mask_coords)
        
        # Loss is L2 in embedding space, NOT pixel space
        return torch.nn.functional.mse_loss(prediction, target_patches)