import torch
import torch.nn as nn
from typing import Optional
from attention import MultiHeadAttention, CrossAttention
from feedforward import FFN
from normalization import RMSNorm


# ---------------------------------------------------------------------------
# Pre-norm transformer block (modern standard)
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Transformer block with pre-norm (norm before sublayer, not after).
    Layout per sublayer: x = x + sublayer(norm(x))
    Pre-norm stabilizes training and is standard in all modern models (GPT-3, LLaMA, etc.)
    """

    def __init__(self, d_model: int, num_heads: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.ffn = FFN(d_model, expansion=expansion, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x), mask=mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# Post-norm transformer block (original paper)
# ---------------------------------------------------------------------------

class TransformerBlockPostNorm(nn.Module):
    """Transformer block with post-norm (norm after sublayer + residual).
    Layout per sublayer: x = norm(x + sublayer(x))
    Original Vaswani et al. 2017 formulation. Harder to train deep without warmup.
    """

    def __init__(self, d_model: int, num_heads: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.ffn = FFN(d_model, expansion=expansion, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.attn(x, mask=mask)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


# ---------------------------------------------------------------------------
# Decoder block (pre-norm, with cross-attention)
# ---------------------------------------------------------------------------

class TransformerDecoderBlock(nn.Module):
    """Decoder block: self-attention → cross-attention → FFN, all pre-norm.
    Used in encoder-decoder architectures (original transformer, T5, etc.)
    Cross-attention lets the decoder condition on encoder outputs.
    """

    def __init__(self, d_model: int, num_heads: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.norm3 = RMSNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.cross_attn = CrossAttention(d_model, num_heads, dropout=dropout)
        self.ffn = FFN(d_model, expansion=expansion, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        self_mask: Optional[torch.Tensor] = None,
        cross_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.dropout(self.self_attn(self.norm1(x), mask=self_mask))
        x = x + self.dropout(self.cross_attn(self.norm2(x), context, mask=cross_mask))
        x = x + self.dropout(self.ffn(self.norm3(x)))
        return x
