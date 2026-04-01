import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Standard FFN
# ---------------------------------------------------------------------------

class FFN(nn.Module):
    """Position-wise Feed-Forward Network (Vaswani et al. 2017).
    Two linear layers with a ReLU in between. Expansion ratio is typically 4x.
    Applied independently to each token position.
    """

    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        d_ff = d_model * expansion
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


# ---------------------------------------------------------------------------
# Gated FFN (SwiGLU / GeGLU)
# ---------------------------------------------------------------------------

class GatedFFN(nn.Module):
    """Gated Feed-Forward Network supporting SwiGLU and GeGLU variants.
    Uses three projections: gate, up, down.
    output = down( act(gate(x)) * up(x) )
    The gate controls how much of the up-projection flows through.

    SwiGLU: act = SiLU (swish)  — used in LLaMA, PaLM, Mistral
    GeGLU:  act = GELU          — used in T5 v1.1, some GPT variants

    d_ff is typically set to 2/3 * 4 * d_model when using gating,
    to keep parameter count comparable to a standard 4x FFN.
    """

    def __init__(self, d_model: int, d_ff: int, activation: str = "swiglu", dropout: float = 0.0):
        super().__init__()
        assert activation in ("swiglu", "geglu"), "activation must be 'swiglu' or 'geglu'"
        self.activation = activation

        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        if self.activation == "swiglu":
            gate = F.silu(gate)
        else:
            gate = F.gelu(gate)
        return self.down_proj(self.dropout(gate * self.up_proj(x)))
