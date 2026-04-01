import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------

def make_causal_mask(seq_len: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """Lower-triangular boolean mask. True = keep, False = mask out.
    Shape: (seq_len, seq_len)
    Used in decoder self-attention to prevent attending to future tokens.
    """
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))


def make_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """Boolean mask from sequence lengths. True = real token, False = padding.
    Shape: (batch_size, max_len)
    Args:
        lengths: (batch_size,) number of real tokens per sequence
        max_len: total sequence length including padding
    """
    batch_size = lengths.size(0)
    positions = torch.arange(max_len, device=lengths.device).unsqueeze(0).expand(batch_size, -1)
    return positions < lengths.unsqueeze(1)


def make_combined_mask(lengths: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Combines causal mask and padding mask. True = keep.
    Shape: (batch_size, seq_len, seq_len)
    Used in decoder self-attention when input sequences are padded.
    """
    causal = make_causal_mask(seq_len, device=lengths.device)                  # (seq_len, seq_len)
    padding = make_padding_mask(lengths, seq_len)                               # (batch_size, seq_len)
    padding = padding.unsqueeze(1).expand(-1, seq_len, -1)                      # (batch_size, seq_len, seq_len)
    return causal.unsqueeze(0) & padding                                        # (batch_size, seq_len, seq_len)


# ---------------------------------------------------------------------------
# Core attention op
# ---------------------------------------------------------------------------

def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    """Scaled dot-product attention.
    Args:
        q: (..., seq_q, d_k)
        k: (..., seq_k, d_k)
        v: (..., seq_k, d_v)
        mask: optional boolean tensor (..., seq_q, seq_k), True = keep
    Returns:
        out: (..., seq_q, d_v)

    NOTE: Flash Attention (Dao et al. 2022) is a hardware-aware rewrite of
    this exact operation — same math, but computed in tiles to avoid
    materializing the full (seq, seq) attention matrix in HBM. The result is
    identical; only the memory access pattern changes. Not drillable by hand.
    """
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / d_k ** 0.5               # (..., seq_q, seq_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attn_weights = F.softmax(scores, dim=-1)
    if dropout_p > 0.0 and training:
        attn_weights = F.dropout(attn_weights, p=dropout_p, training=training)
    return torch.matmul(attn_weights, v)                                       # (..., seq_q, d_v)


# ---------------------------------------------------------------------------
# Multi-head attention
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """Standard multi-head self-attention.
    Splits d_model into num_heads parallel heads, each with d_k = d_model // num_heads.
    Projects Q, K, V independently, attends, merges heads, then projects output.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, d_model) → (B, num_heads, T, d_k)"""
        B, T, D = x.shape
        x = x.view(B, T, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, num_heads, T, d_k) → (B, T, d_model)"""
        B, H, T, d_k = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(B, T, H * d_k)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = self._split_heads(self.W_q(x))
        k = self._split_heads(self.W_k(x))
        v = self._split_heads(self.W_v(x))
        out = scaled_dot_product_attention(q, k, v, mask=mask, dropout_p=self.dropout, training=self.training)
        out = self._merge_heads(out)
        return self.W_o(out)


# ---------------------------------------------------------------------------
# Cross-attention
# ---------------------------------------------------------------------------

class CrossAttention(nn.Module):
    """Cross-attention: Q comes from the decoder, K and V come from the encoder.
    Used in encoder-decoder architectures to condition the decoder on encoder outputs.
    Architecture is identical to MHA except Q and context (K/V source) are separate inputs.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        return x.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, H, T, d_k = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * d_k)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:       (B, T_dec, d_model) — decoder query input
            context: (B, T_enc, d_model) — encoder key/value input
            mask:    optional (B, T_dec, T_enc) padding mask over encoder tokens
        """
        q = self._split_heads(self.W_q(x))
        k = self._split_heads(self.W_k(context))
        v = self._split_heads(self.W_v(context))
        out = scaled_dot_product_attention(q, k, v, mask=mask, dropout_p=self.dropout, training=self.training)
        return self.W_o(self._merge_heads(out))