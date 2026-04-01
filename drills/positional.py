import torch
import torch.nn as nn
import math


# ---------------------------------------------------------------------------
# Sinusoidal positional encoding (original transformer, Vaswani et al. 2017)
# ---------------------------------------------------------------------------

def sinusoidal_encoding(seq_len: int, d_model: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """Fixed sinusoidal positional encodings. Not learned, no parameters.
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    Returns:
        (seq_len, d_model)
    """
    pe = torch.zeros(seq_len, d_model, device=device)
    position = torch.arange(0, seq_len, device=device).unsqueeze(1).float()         # (seq_len, 1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device).float() * -(math.log(10000.0) / d_model)
    )                                                                                 # (d_model/2,)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


# ---------------------------------------------------------------------------
# Learned absolute positional embeddings (BERT, GPT-2)
# ---------------------------------------------------------------------------

class LearnedPositionalEmbedding(nn.Module):
    """Trainable position embeddings — each position gets its own embedding vector.
    Simpler than sinusoidal but can't extrapolate beyond max_seq_len seen during training.
    Used in: BERT, GPT-2.
    """

    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            x + positional embeddings: (B, T, d_model)
        """
        B, T, D = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0)                   # (1, T)
        return x + self.embedding(positions)


# ---------------------------------------------------------------------------
# Rotary positional embeddings (RoPE, Su et al. 2021)
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE).
    Encodes position by rotating Q and K vectors in 2D subspaces.
    Key property: dot product of Q_m and K_n depends only on their content and (m-n),
    giving relative position awareness without explicit relative attention.
    Used in: LLaMA, Mistral, GPT-NeoX, PaLM 2.
    """

    def __init__(self, d_k: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        self.d_k = d_k
        inv_freq = 1.0 / (base ** (torch.arange(0, d_k, 2).float() / d_k))          # (d_k/2,)
        self.register_buffer("inv_freq", inv_freq)

        # precompute cos/sin for all positions up to max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()               # (seq_len,)
        freqs = torch.outer(t, self.inv_freq)                                         # (seq_len, d_k/2)
        emb = torch.cat([freqs, freqs], dim=-1)                                       # (seq_len, d_k)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotates the second half of the last dimension to implement the 2D rotation."""
        x1, x2 = x[..., : self.d_k // 2], x[..., self.d_k // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple:
        """Apply rotary embeddings to Q and K.
        Args:
            q: (B, H, T, d_k)
            k: (B, H, T, d_k)
        Returns:
            q_rot, k_rot: same shapes
        """
        T = q.size(2)
        cos = self.cos_cached[:T].unsqueeze(0).unsqueeze(0)                          # (1, 1, T, d_k)
        sin = self.sin_cached[:T].unsqueeze(0).unsqueeze(0)

        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot
