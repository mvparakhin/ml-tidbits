"""
TestEmbedModels.py
==============
Deterministic Gaussian Autoencoder -- minimal training example.

Generates strongly non-Gaussian synthetic data in 15-D, trains an autoencoder
whose latent space (8-D) is pushed toward N(0,I) by the Wristband Gaussian
Loss, and prints diagnostics every epoch.

Usage:
    python TestEmbedModels.py
"""

import torch
import torch.nn as nn
import os, sys
# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embed_models.EmbedModels import (
   C_EmbedAttentionModule,
   C_ACN,
   C_InvertibleFlow,
   C_WristbandGaussianLoss,
)

#///////////////////////////////////////////////////////////////////////////////
# Model
#///////////////////////////////////////////////////////////////////////////////

class Encoder(nn.Module):
   def __init__(self, in_dim, embed_dim):
      super().__init__()
      internal = 128
      heads    = 64
      basis    = 256
      self.net = C_EmbedAttentionModule(
         in_dim, internal, embed_dim, basis, heads,
         q_transform = nn.Linear(in_dim, in_dim),
         head_combine = C_ACN(heads * internal, embed_dim, internal, 2),
      )
   def forward(self, x):
      return self.net(x)

class Decoder(nn.Module):
   def __init__(self, embed_dim, out_dim):
      super().__init__()
      internal = 128
      heads    = 64
      basis    = 256
      self.net = C_EmbedAttentionModule(
         embed_dim, internal, out_dim, basis, heads,
         q_transform = None,
         head_combine = C_ACN(heads * internal, out_dim, internal, 2),
      )
   def forward(self, z):
      return self.net(z)

class DeterministicGaussianAutoencoder(nn.Module):
   def __init__(self, in_dim, embed_dim):
      super().__init__()
      self.embed_dim = embed_dim
      self.encoder = Encoder(in_dim, embed_dim)
      self.decoder = Decoder(embed_dim, in_dim)
      self.flow = C_InvertibleFlow(
         embed_dim,
         n_layers   = 4,
         hidden_dim = 32,
         n_blocks   = 2,
         s_max      = 2.,
         permute_mode = "per_pair",
      )

   def forward(self, x):
      z = self.flow(self.encoder(x))              # encode -> flow -> latent
      x_hat = self.decoder(self.flow.inverse(z))  # latent -> inverse flow -> decode
      return x_hat, z

#///////////////////////////////////////////////////////////////////////////////
# Synthetic non-Gaussian data
#///////////////////////////////////////////////////////////////////////////////

def make_non_gaussian_data(n_samples: int, dim: int, n_clusters: int = 5, seed: int = 42):
   """Mixture of shifted/scaled Gaussians with heavy tails and correlations.

   The result is strongly non-Gaussian: multi-modal, skewed, with non-trivial
   covariance structure.
   """
   rng = torch.Generator().manual_seed(seed)

   # Random cluster centres spread out
   centres = torch.randn(n_clusters, dim, generator=rng) * 4.0

   # Per-cluster random linear transform (induces correlations + varying scale)
   transforms = []
   for _ in range(n_clusters):
      A = torch.randn(dim, dim, generator=rng) * 0.6
      transforms.append(A)

   samples_per = n_samples // n_clusters
   parts = []
   for i in range(n_clusters):
      raw = torch.randn(samples_per, dim, generator=rng)
      # Cube some dimensions to add skew / heavy tails
      raw[:, :dim // 3] = raw[:, :dim // 3].sign() * raw[:, :dim // 3].abs().pow(1.5)
      parts.append(raw @ transforms[i].T + centres[i])

   data = torch.cat(parts, dim=0)
   # Shuffle
   perm = torch.randperm(data.size(0), generator=rng)
   return data[perm]

#///////////////////////////////////////////////////////////////////////////////
# Training
#///////////////////////////////////////////////////////////////////////////////

def main():
   device = "cuda" if torch.cuda.is_available() else "cpu"
   print(f"Device: {device}")

   # Hyperparameters
   in_dim      = 15
   embed_dim   = 8
   n_samples   = 100_000
   batch_size  = 1024
   n_epochs    = 40
   lr          = 3e-4
   lambda_rec  = 1.0     # reconstruction weight
   lambda_wb   = 0.1     # wristband weight

   # Data
   data = make_non_gaussian_data(n_samples, in_dim).to(device)
   # Standardise input (zero mean, unit variance per feature) for stable training
   data_mean = data.mean(dim=0)
   data_std  = data.std(dim=0).clamp_min(1e-6)
   data = (data - data_mean) / data_std
   print(f"Data: {data.shape}  (mean~=0, std~=1 per feature after standardisation)")

   # Model
   model = DeterministicGaussianAutoencoder(in_dim, embed_dim).to(device)
   n_params = sum(p.numel() for p in model.parameters())
   print(f"Model parameters: {n_params:,}")

   # Wristband loss -- calibrated for the actual batch/embed dimensions
   wristband = C_WristbandGaussianLoss(
      calibration_shape = (batch_size, embed_dim),
   )
   print(f"Wristband calibration done  (batch={batch_size}, dim={embed_dim})")

   optimiser = torch.optim.Adam(model.parameters(), lr=lr)

   # Training loop
   print(f"\n{'epoch':>5}  {'loss':>9}  {'recon':>9}  {'wb_tot':>9}  {'wb_rep':>9}  {'wb_rad':>9}  {'wb_mom':>9}")
   print("-" * 72)

   for epoch in range(1, n_epochs + 1):
      perm = torch.randperm(data.size(0), device=device)
      epoch_loss = epoch_rec = epoch_wb = 0.
      n_batches = 0

      for i in range(0, data.size(0) - batch_size + 1, batch_size):
         batch = data[perm[i : i + batch_size]]

         x_hat, z = model(batch)

         # Reconstruction (MSE)
         rec_loss = (x_hat - batch).square().mean()

         # Wristband on latent z
         wb = wristband(z)

         loss = lambda_rec * rec_loss + lambda_wb * wb.total

         optimiser.zero_grad()
         loss.backward()
         optimiser.step()

         epoch_loss += loss.item()
         epoch_rec  += rec_loss.item()
         epoch_wb   += wb.total.item()
         n_batches  += 1

      if n_batches > 0:
         el = epoch_loss / n_batches
         er = epoch_rec  / n_batches
         ew = epoch_wb   / n_batches

      # Per-epoch wristband diagnostics on the full data (no grad)
      with torch.no_grad():
         _, z_all = model(data[:batch_size])  # use one batch for diagnostics
         wb_diag = wristband(z_all)

      print(f"{epoch:5d}  {el:9.4f}  {er:9.4f}  {wb_diag.total:9.4f}  "
            f"{wb_diag.rep:9.4f}  {wb_diag.rad:9.4f}  {wb_diag.mom:9.4f}")

   # -- Quick sanity check: are latents approximately Gaussian? --
   print("\n-- Latent statistics (should be ~= 0 mean, ~= 1 std) --")
   with torch.no_grad():
      _, z_final = model(data)
      print(f"  mean: {z_final.mean(0).cpu().tolist()}")
      print(f"  std:  {z_final.std(0).cpu().tolist()}")
      print(f"  overall mean={z_final.mean():.4f}, std={z_final.std():.4f}")

   # -- Round-trip reconstruction quality --
   with torch.no_grad():
      x_hat_all, _ = model(data)
      mse = (x_hat_all - data).square().mean().item()
      print(f"\n  Reconstruction MSE: {mse:.6f}")

   print("\nDone.")

if __name__ == "__main__":
   main()