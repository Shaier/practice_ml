# source: drills/attention.py::make_causal_mask
# masked: 1 line(s)

def make_causal_mask(seq_len: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """Lower-triangular boolean mask. True = keep, False = mask out.
    Shape: (seq_len, seq_len)
    Used in decoder self-attention to prevent attending to future tokens.
    """
    # code here
