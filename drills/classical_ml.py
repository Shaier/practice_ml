import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ---------------------------------------------------------------------------
# Linear Regression
# ---------------------------------------------------------------------------

class LinearRegression(nn.Module):
    """Linear regression: y = xW + b
    Trained with MSE loss. Closed-form solution exists (normal equations)
    but gradient descent is used here for consistency with the rest of the code.
    """

    def __init__(self, in_features: int, out_features: int = 1):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(preds, targets)


# ---------------------------------------------------------------------------
# Logistic Regression
# ---------------------------------------------------------------------------

class LogisticRegression(nn.Module):
    """Binary logistic regression: p = sigmoid(xW + b)
    Outputs a probability. Trained with binary cross-entropy loss.
    Decision boundary is linear in input space.
    """

    def __init__(self, in_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns probabilities in [0, 1]. Shape: (B, 1)"""
        return torch.sigmoid(self.linear(x))

    def loss(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Binary cross-entropy: -[y log p + (1-y) log(1-p)]"""
        return F.binary_cross_entropy(probs.squeeze(-1), targets.float())


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def precision(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Precision = TP / (TP + FP)
    Of all positive predictions, how many are actually positive.
    preds, targets: (N,) binary tensors (0 or 1)
    """
    tp = (preds * targets).sum().float()
    fp = (preds * (1 - targets)).sum().float()
    return tp / (tp + fp + 1e-8)


def recall(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Recall = TP / (TP + FN)
    Of all actual positives, how many did we correctly predict.
    preds, targets: (N,) binary tensors (0 or 1)
    """
    tp = (preds * targets).sum().float()
    fn = ((1 - preds) * targets).sum().float()
    return tp / (tp + fn + 1e-8)


def f1_score(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """F1 = 2 * precision * recall / (precision + recall)
    Harmonic mean of precision and recall. Balances both metrics.
    preds, targets: (N,) binary tensors (0 or 1)
    """
    p = precision(preds, targets)
    r = recall(preds, targets)
    return 2 * p * r / (p + r + 1e-8)


# ---------------------------------------------------------------------------
# K-Means clustering
# ---------------------------------------------------------------------------

def kmeans(
    x: torch.Tensor,
    k: int,
    num_iters: int = 100,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """K-Means clustering with Lloyd's algorithm.
    1. Randomly initialise k centroids from the data points.
    2. Assign each point to the nearest centroid (by L2 distance).
    3. Update each centroid to the mean of its assigned points.
    4. Repeat until convergence or max iterations.

    Args:
        x:         (N, d) data points
        k:         number of clusters
        num_iters: maximum iterations
    Returns:
        centroids:   (k, d) final centroid positions
        assignments: (N,)  cluster index for each point
    """
    N, d = x.shape
    indices = torch.randperm(N, device=x.device)[:k]
    centroids = x[indices].clone()

    for _ in range(num_iters):
        dists = torch.cdist(x, centroids)                                  # (N, k)
        assignments = dists.argmin(dim=-1)                                 # (N,)

        new_centroids = torch.zeros_like(centroids)
        for j in range(k):
            mask = assignments == j
            if mask.any():
                new_centroids[j] = x[mask].mean(dim=0)
            else:
                new_centroids[j] = centroids[j]                            # keep old if empty cluster

        if torch.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return centroids, assignments
