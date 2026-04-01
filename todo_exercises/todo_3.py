# source: drills/attention.py::make_padding_mask
# masked: 1 line(s)

def make_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """Boolean mask from sequence lengths. True = real token, False = padding.
    Shape: (batch_size, max_len)
    Args:
        lengths: (batch_size,) number of real tokens per sequence
        max_len: total sequence length including padding
    """
    # code here
    positions = torch.arange(max_len, device=lengths.device).unsqueeze(0).expand(batch_size, -1)
    return positions < lengths.unsqueeze(1)
