# source: drills/attention.py::MultiHeadAttention
# masked: 1 line(s)

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

        # code here
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
