import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# InfoNCE loss
# ---------------------------------------------------------------------------

def info_nce_loss(anchors: torch.Tensor, positives: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """InfoNCE contrastive loss (van den Oord et al. 2018).
    For each anchor, the matching positive is the target; all other items in
    the batch act as negatives (in-batch negatives).
    Intuitively: a classification loss where the correct class is the paired item.

    anchors:   (B, d) L2-normalised embeddings
    positives: (B, d) L2-normalised embeddings — anchor[i] pairs with positive[i]
    """
    anchors = F.normalize(anchors, dim=-1)
    positives = F.normalize(positives, dim=-1)

    logits = torch.matmul(anchors, positives.T) / temperature              # (B, B)
    labels = torch.arange(logits.size(0), device=logits.device)            # [0, 1, ..., B-1]
    return F.cross_entropy(logits, labels)


# ---------------------------------------------------------------------------
# NT-Xent (SimCLR loss)
# ---------------------------------------------------------------------------

def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """Normalised Temperature-scaled Cross Entropy loss (Chen et al. 2020 — SimCLR).
    Two augmented views of the same image form a positive pair.
    All 2B - 2 other embeddings in the batch are negatives.
    Symmetrised: loss computed in both directions and averaged.

    z1, z2: (B, d) L2-normalised embeddings from two augmentation views
    """
    B = z1.size(0)
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    z = torch.cat([z1, z2], dim=0)                                         # (2B, d)
    sim = torch.matmul(z, z.T) / temperature                               # (2B, 2B)

    # mask out self-similarity (diagonal)
    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, float("-inf"))

    # positive pairs: (i, i+B) and (i+B, i)
    labels = torch.cat([
        torch.arange(B, 2 * B, device=z.device),
        torch.arange(B, device=z.device),
    ])                                                                      # (2B,)
    return F.cross_entropy(sim, labels)


# ---------------------------------------------------------------------------
# CLIP-style dual encoder
# ---------------------------------------------------------------------------

class CLIPLoss(nn.Module):
    """Contrastive Language-Image Pre-training loss (Radford et al. 2021).
    Aligns image and text embeddings by maximising similarity of matched pairs
    and minimising similarity of unmatched pairs — symmetrically in both directions.
    Temperature is a learned scalar (log-parameterised to keep it positive).

    image_embeds: (B, d) L2-normalised
    text_embeds:  (B, d) L2-normalised
    """

    def __init__(self, init_temperature: float = 0.07):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(init_temperature)))

    def forward(self, image_embeds: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)
        temperature = self.log_temperature.exp()

        logits = torch.matmul(image_embeds, text_embeds.T) / temperature   # (B, B)
        labels = torch.arange(logits.size(0), device=logits.device)

        loss_i2t = F.cross_entropy(logits, labels)                         # image → text
        loss_t2i = F.cross_entropy(logits.T, labels)                       # text → image
        return (loss_i2t + loss_t2i) / 2


class DualEncoder(nn.Module):
    """Minimal CLIP-style dual encoder.
    Two separate encoder towers (e.g. image + text) project to a shared embedding space.
    Trained with CLIPLoss to align matched pairs.
    """

    def __init__(self, image_encoder: nn.Module, text_encoder: nn.Module, d_embed: int, d_model: int):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.image_proj = nn.Linear(d_model, d_embed, bias=False)
        self.text_proj = nn.Linear(d_model, d_embed, bias=False)
        self.loss_fn = CLIPLoss()

    def forward(self, images: torch.Tensor, texts: torch.Tensor) -> torch.Tensor:
        image_embeds = self.image_proj(self.image_encoder(images))
        text_embeds = self.text_proj(self.text_encoder(texts))
        return self.loss_fn(image_embeds, text_embeds)
