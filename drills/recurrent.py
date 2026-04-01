import torch
import torch.nn as nn
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# RNN Cell
# ---------------------------------------------------------------------------

class RNNCell(nn.Module):
    """Single-step vanilla RNN.
    h_t = tanh(x_t @ W_xh + h_{t-1} @ W_hh + b)
    Hidden state mixes current input with previous hidden state.
    Suffers from vanishing gradients over long sequences.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.W_xh = nn.Linear(input_size, hidden_size)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_size)
            h: (B, hidden_size)
        Returns:
            h_next: (B, hidden_size)
        """
        return torch.tanh(self.W_xh(x) + self.W_hh(h))


# ---------------------------------------------------------------------------
# LSTM Cell
# ---------------------------------------------------------------------------

class LSTMCell(nn.Module):
    """Single-step LSTM (Hochreiter & Schmidhuber 1997).
    Four gates computed from x_t and h_{t-1}:
        i (input gate):  how much new info to write to cell state
        f (forget gate): how much old cell state to keep
        g (cell gate):   candidate values to write
        o (output gate): how much of cell state to expose as h_t
    c_t = f * c_{t-1} + i * g
    h_t = o * tanh(c_t)
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        # one combined projection for all 4 gates — more efficient
        self.W_x = nn.Linear(input_size, 4 * hidden_size)
        self.W_h = nn.Linear(hidden_size, 4 * hidden_size, bias=False)

    def forward(
        self, x: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:     (B, input_size)
            state: (h, c) each (B, hidden_size)
        Returns:
            (h_next, c_next) each (B, hidden_size)
        """
        h, c = state
        gates = self.W_x(x) + self.W_h(h)                                       # (B, 4 * hidden_size)
        i, f, g, o = gates.chunk(4, dim=-1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


# ---------------------------------------------------------------------------
# RNN (unrolled over sequence)
# ---------------------------------------------------------------------------

class RNN(nn.Module):
    """Unrolls an RNNCell over a full sequence."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = RNNCell(input_size, hidden_size)

    def forward(
        self, x: torch.Tensor, h0: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:  (B, T, input_size)
            h0: (B, hidden_size) initial hidden state, zeros if None
        Returns:
            outputs: (B, T, hidden_size) all hidden states
            h_n:     (B, hidden_size) final hidden state
        """
        B, T, _ = x.shape
        h = h0 if h0 is not None else torch.zeros(B, self.hidden_size, device=x.device)
        outputs = []
        for t in range(T):
            h = self.cell(x[:, t, :], h)
            outputs.append(h.unsqueeze(1))
        return torch.cat(outputs, dim=1), h


# ---------------------------------------------------------------------------
# LSTM (unrolled over sequence)
# ---------------------------------------------------------------------------

class LSTM(nn.Module):
    """Unrolls an LSTMCell over a full sequence."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = LSTMCell(input_size, hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x:     (B, T, input_size)
            state: (h0, c0) each (B, hidden_size), zeros if None
        Returns:
            outputs: (B, T, hidden_size) all hidden states
            (h_n, c_n): final hidden and cell states
        """
        B, T, _ = x.shape
        if state is None:
            h = torch.zeros(B, self.hidden_size, device=x.device)
            c = torch.zeros(B, self.hidden_size, device=x.device)
        else:
            h, c = state
        outputs = []
        for t in range(T):
            h, c = self.cell(x[:, t, :], (h, c))
            outputs.append(h.unsqueeze(1))
        return torch.cat(outputs, dim=1), (h, c)
