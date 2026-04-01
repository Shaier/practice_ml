import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ---------------------------------------------------------------------------
# Autoencoder
# ---------------------------------------------------------------------------

class Autoencoder(nn.Module):
    """Deterministic autoencoder.
    Encoder compresses input to a bottleneck (latent code).
    Decoder reconstructs input from the latent code.
    Trained with reconstruction loss (MSE or BCE).
    Latent space is unstructured — not suitable for generation (use VAE for that).
    """

    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

    def loss(self, x: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(recon, x)


# ---------------------------------------------------------------------------
# Variational Autoencoder (VAE)
# ---------------------------------------------------------------------------

class VAE(nn.Module):
    """Variational Autoencoder (Kingma & Welling 2013).
    Encoder outputs parameters (mu, log_var) of a diagonal Gaussian q(z|x).
    Sampling uses the reparameterisation trick: z = mu + eps * std, eps ~ N(0,I)
    — this keeps the sample differentiable w.r.t. mu and log_var.
    Decoder reconstructs x from z.

    ELBO loss = reconstruction loss + KL divergence
    KL(q(z|x) || p(z)) = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
    KL regularises the latent space to be close to N(0,I), enabling smooth generation.
    """

    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_log_var(h)

    def reparameterise(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """z = mu + eps * std, eps ~ N(0, I). Differentiable w.r.t. mu and log_var."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterise(mu, log_var)
        recon = self.decode(z)
        return recon, mu, log_var

    def loss(self, x: torch.Tensor, recon: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        recon_loss = F.mse_loss(recon, x, reduction="sum")
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return (recon_loss + kl_loss) / x.size(0)


# ---------------------------------------------------------------------------
# GAN (Generator + Discriminator + training step)
# ---------------------------------------------------------------------------

class Generator(nn.Module):
    """Maps random noise z ~ N(0,I) to data space."""

    def __init__(self, latent_dim: int, output_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Discriminator(nn.Module):
    """Outputs probability that input is real (not generated)."""

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def gan_discriminator_loss(real_scores: torch.Tensor, fake_scores: torch.Tensor) -> torch.Tensor:
    """Discriminator loss: maximise log D(x) + log(1 - D(G(z))).
    Equivalently: binary cross-entropy with real=1, fake=0.
    """
    real_loss = F.binary_cross_entropy(real_scores, torch.ones_like(real_scores))
    fake_loss = F.binary_cross_entropy(fake_scores, torch.zeros_like(fake_scores))
    return real_loss + fake_loss


def gan_generator_loss(fake_scores: torch.Tensor) -> torch.Tensor:
    """Generator loss: maximise log D(G(z)), i.e. fool discriminator into predicting real=1.
    In practice this non-saturating loss (treat fakes as real) trains better than
    minimising log(1 - D(G(z))) because gradients don't vanish early in training.
    """
    return F.binary_cross_entropy(fake_scores, torch.ones_like(fake_scores))


# ---------------------------------------------------------------------------
# Straight-Through Estimator (STE)
# ---------------------------------------------------------------------------

class StraightThroughStep(torch.autograd.Function):
    """Straight-Through Estimator (Bengio et al. 2013).
    Allows gradients to flow through a non-differentiable quantisation/thresholding op.

    Forward: apply the discrete/quantised function (e.g. round, sign, argmax).
    Backward: pretend the operation was the identity — pass gradients straight through.

    Used in: VQ-VAE (vector quantisation), binary neural networks, discrete VAEs.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return x.round()                                                    # discrete op (e.g. quantise)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output                                                  # identity: pass grad unchanged


def straight_through_round(x: torch.Tensor) -> torch.Tensor:
    """Round x to nearest integer, but pass gradients through as if identity."""
    return StraightThroughStep.apply(x)


class VectorQuantiser(nn.Module):
    """Vector Quantisation with Straight-Through gradient (VQ-VAE, van den Oord et al. 2017).
    Maps continuous encoder outputs to the nearest entry in a learned codebook.
    Gradients for the encoder are copied straight from the decoder input (STE).
    The codebook is updated via a separate commitment + embedding loss.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.commitment_cost = commitment_cost
        nn.init.uniform_(self.embedding.weight, -1 / num_embeddings, 1 / num_embeddings)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (B, T, embedding_dim) continuous encoder output
        Returns:
            z_q:  (B, T, embedding_dim) quantised (nearest codebook entry)
            loss: scalar VQ loss
        """
        dists = torch.cdist(z, self.embedding.weight)                      # (B, T, num_embeddings)
        indices = dists.argmin(dim=-1)                                     # (B, T)
        z_q = self.embedding(indices)                                      # (B, T, embedding_dim)

        loss = (
            F.mse_loss(z_q.detach(), z) * self.commitment_cost            # commit encoder to codebook
            + F.mse_loss(z_q, z.detach())                                  # move codebook toward encoder
        )

        z_q = z + (z_q - z).detach()                                       # STE: copy gradients from z_q to z
        return z_q, loss
