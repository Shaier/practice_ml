import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------

def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically stable softmax.
    Subtracts max before exp to prevent overflow. Doesn't change the result
    since the constant cancels in numerator/denominator.
    """
    x_max = x.max(dim=dim, keepdim=True).values
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(dim=dim, keepdim=True)


def swish(x: torch.Tensor) -> torch.Tensor:
    """Swish activation (Ramachandran et al. 2017): x * sigmoid(x).
    Smooth, non-monotonic. Same as SiLU.
    """
    return x * torch.sigmoid(x)


def gelu(x: torch.Tensor) -> torch.Tensor:
    """Gaussian Error Linear Unit (Hendrycks & Gimpel 2016).
    Smooth approximation of ReLU that weights inputs by their probability
    under a Gaussian. Tanh approximation used in GPT-2.
    """
    return 0.5 * x * (1.0 + torch.tanh(
        (2.0 / torch.pi) ** 0.5 * (x + 0.044715 * x ** 3)
    ))


def silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU (Sigmoid Linear Unit) — identical to Swish.
    x * sigmoid(x). Used as the gate activation in SwiGLU.
    """
    return x * torch.sigmoid(x)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Manual cross-entropy loss.
    Steps: log-softmax over vocab → gather log-prob of correct token → negate and mean.
    logits:  (B, vocab_size) or (B, T, vocab_size)
    targets: (B,) or (B, T) integer class indices
    """
    log_probs = F.log_softmax(logits, dim=-1)                                    # (B, vocab) or (B, T, vocab)
    if logits.dim() == 3:
        B, T, V = logits.shape
        log_probs = log_probs.view(B * T, V)
        targets = targets.view(B * T)
    correct_log_probs = log_probs[torch.arange(log_probs.size(0)), targets]      # (B*T,) or (B,)
    return -correct_log_probs.mean()


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Focal loss (Lin et al. 2017) for class imbalance.
    Downweights easy examples (high p_t) so training focuses on hard ones.
    FL(p_t) = -(1 - p_t)^gamma * log(p_t)
    gamma=0 recovers standard cross-entropy. gamma=2 is the typical default.
    logits:  (B, num_classes)
    targets: (B,) integer class indices
    """
    log_probs = F.log_softmax(logits, dim=-1)                                    # (B, C)
    probs = torch.exp(log_probs)                                                  # (B, C)
    B = logits.size(0)
    p_t = probs[torch.arange(B), targets]                                        # (B,)
    log_p_t = log_probs[torch.arange(B), targets]                                # (B,)
    return (-((1 - p_t) ** gamma) * log_p_t).mean()
