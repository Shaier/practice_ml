import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from attention import scaled_dot_product_attention


# ---------------------------------------------------------------------------
# Multi-Query Attention (MQA)
# ---------------------------------------------------------------------------

class MultiQueryAttention(nn.Module):
    """Multi-Query Attention: all query heads share a single K and V head.
    Reduces KV cache memory by num_heads x at inference time.
    Used in: PaLM, Falcon.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)           # projects to (num_heads * d_k)
        self.W_k = nn.Linear(d_model, self.d_k)          # single head
        self.W_v = nn.Linear(d_model, self.d_k)          # single head
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape

        q = self.W_q(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)  # (B, H, T, d_k)
        k = self.W_k(x).unsqueeze(1).expand(-1, self.num_heads, -1, -1)        # (B, H, T, d_k)
        v = self.W_v(x).unsqueeze(1).expand(-1, self.num_heads, -1, -1)        # (B, H, T, d_k)

        out = scaled_dot_product_attention(q, k, v, mask=mask, dropout_p=self.dropout, training=self.training)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(out)


# ---------------------------------------------------------------------------
# Grouped-Query Attention (GQA)
# ---------------------------------------------------------------------------

class GroupedQueryAttention(nn.Module):
    """Grouped-Query Attention: Q heads are split into groups, each group shares one K/V head.
    Interpolates between MHA (num_kv_heads == num_heads) and MQA (num_kv_heads == 1).
    Used in: Llama 2/3, Mistral.
    """

    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int, dropout: float = 0.0):
        super().__init__()
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_heads // num_kv_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, num_heads * self.d_k)
        self.W_k = nn.Linear(d_model, num_kv_heads * self.d_k)
        self.W_v = nn.Linear(d_model, num_kv_heads * self.d_k)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape

        q = self.W_q(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)       # (B, H, T, d_k)
        k = self.W_k(x).view(B, T, self.num_kv_heads, self.d_k).transpose(1, 2)    # (B, G, T, d_k)
        v = self.W_v(x).view(B, T, self.num_kv_heads, self.d_k).transpose(1, 2)    # (B, G, T, d_k)

        # expand K/V so each group of Q heads sees the same K/V head
        k = k.repeat_interleave(self.num_groups, dim=1)                             # (B, H, T, d_k)
        v = v.repeat_interleave(self.num_groups, dim=1)                             # (B, H, T, d_k)

        out = scaled_dot_product_attention(q, k, v, mask=mask, dropout_p=self.dropout, training=self.training)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(out)


# ---------------------------------------------------------------------------
# KV Cache
# ---------------------------------------------------------------------------

class KVCache:
    """Stores and incrementally grows K and V tensors during autoregressive decoding.
    At each decode step, new K/V slices are appended so attention sees full history.
    Used in all decoder-only and encoder-decoder inference loops.
    """

    def __init__(self):
        self.k: Optional[torch.Tensor] = None
        self.v: Optional[torch.Tensor] = None

    def update(self, new_k: torch.Tensor, new_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append new K/V slices and return the full cached K/V.
        Args:
            new_k: (B, H, 1, d_k) — K for the current token
            new_v: (B, H, 1, d_k) — V for the current token
        Returns:
            k: (B, H, T_so_far, d_k)
            v: (B, H, T_so_far, d_k)
        """
        if self.k is None:
            self.k = new_k
            self.v = new_v
        else:
            self.k = torch.cat([self.k, new_k], dim=2)
            self.v = torch.cat([self.v, new_v], dim=2)
        return self.k, self.v

    def reset(self):
        self.k = None
        self.v = None
