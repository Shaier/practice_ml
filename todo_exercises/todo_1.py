# source: drills/attention.py::CrossAttention
# masked: 1 line(s)

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
        # code here
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
