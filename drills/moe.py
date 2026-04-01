import torch
import torch.nn as nn
import torch.nn.functional as F
from feedforward import FFN


# ---------------------------------------------------------------------------
# Switch MoE (hard routing — top-1)
# ---------------------------------------------------------------------------

class SwitchMoE(nn.Module):
    """Mixture of Experts with hard top-1 routing (Fedus et al. 2021 — Switch Transformer).
    Each token is routed to exactly one expert based on the highest router logit.
    Only one expert is activated per token — compute cost stays constant regardless of num_experts.

    Router: linear projection → softmax → argmax to select expert.
    The routing score (probability) is multiplied into the expert output to allow
    gradients to flow back through the router.
    """

    def __init__(self, d_model: int, num_experts: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        self.num_experts = num_experts
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([
            FFN(d_model, expansion=expansion, dropout=dropout)
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            (B, T, d_model)
        """
        B, T, D = x.shape
        x_flat = x.view(B * T, D)                                                # (B*T, D)

        router_logits = self.router(x_flat)                                      # (B*T, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)                          # (B*T, num_experts)
        expert_idx = router_probs.argmax(dim=-1)                                 # (B*T,)
        top1_scores = router_probs[torch.arange(B * T), expert_idx].unsqueeze(-1)  # (B*T, 1)

        out = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (expert_idx == i)                                             # (B*T,) bool
            if mask.any():
                out[mask] = expert(x_flat[mask])

        out = out * top1_scores                                                  # scale by routing prob
        return out.view(B, T, D)


# ---------------------------------------------------------------------------
# Soft MoE (weighted average over top-k experts)
# ---------------------------------------------------------------------------

class SoftMoE(nn.Module):
    """Soft Mixture of Experts with top-k routing (Puigcerver et al. 2023).
    Instead of hard routing, each token gets a weighted combination of the top-k expert outputs.
    Weights are the softmax-normalized scores of the top-k experts.
    More expressive than Switch MoE; more compute (k experts run per token).
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        top_k: int = 2,
        expansion: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_experts
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([
            FFN(d_model, expansion=expansion, dropout=dropout)
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            (B, T, d_model)
        """
        B, T, D = x.shape
        x_flat = x.view(B * T, D)                                                # (B*T, D)

        router_logits = self.router(x_flat)                                      # (B*T, num_experts)
        topk_scores, topk_indices = router_logits.topk(self.top_k, dim=-1)      # (B*T, k)
        topk_weights = F.softmax(topk_scores, dim=-1)                            # (B*T, k) normalized

        out = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            expert_ids = topk_indices[:, k]                                      # (B*T,)
            weights = topk_weights[:, k].unsqueeze(-1)                           # (B*T, 1)
            for i, expert in enumerate(self.experts):
                mask = (expert_ids == i)
                if mask.any():
                    out[mask] += weights[mask] * expert(x_flat[mask])

        return out.view(B, T, D)
