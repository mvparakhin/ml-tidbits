# Deterministic Conditional Sampling Demo (MNIST top-half | bottom-half).
#
# Demonstrates the Wristband Gaussian Loss approach for deterministic
# Gaussian autoencoders applied to a simple conditional generation task:
#   Given the bottom half of an MNIST digit, sample top halves from the conditional distribution.
#
# Architecture:
#   - Two separate encoders (bottom -> z_b, top -> z_t) using simple FC networks
#   - Two separate invertible flows push embeddings toward N(0,I)
#   - Shared decoder: flow -> linear features -> ConvTranspose upsample -> 28x28
#
# Training losses:
#   (1) Full reconstruction: encode both halves, decode, compare to original
#   (2) Expected reconstruction: encode bottom only,
#       sample z_t ~ N(0,I) K times, average the per-sample losses
#   (3) Wristband Gaussian loss on the joint z = [z_b, z_t]
#
# At inference, only the bottom half is needed: encode it to z_b, sample
# z_t ~ N(0,I), and decode - each sample yields a different plausible top half.

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from torchvision import datasets, transforms
from torchvision.utils import make_grid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embed_models.EmbedModels import C_InvertibleFlow, C_ACN, C_WristbandGaussianLoss
from schedulers.Schedulers import C_CosineAnnealingWarmRestartsDecay


##########################################################################################################################################################################################
def SplitTopBottom(x):
   """Split a (B,1,28,28) MNIST image into top and bottom halves, each (B,1,14,28)."""
   return x[:, :, :14, :], x[:, :, 14:, :]


##########################################################################################################################################################################################
class C_InpaintMNIST(nn.Module):
   """Conditional inpainting model: bottom half -> latent -> full image.

   Two separate encoder+flow paths produce z_b (bottom) and z_t (top).
   A shared decoder reconstructs the full 28x28 image from both latents.
   At test time, z_t is sampled from N(0,I) to generate diverse top halves.
   """

   def __init__(self, d_b: int, d_t: int):
      super().__init__()
      self.d_b = d_b
      self.d_t = d_t
      self.d   = d_b + d_t

      # --- Encoders: flatten half-image (14x28=392) -> latent ---
      self.enc_bottom = C_ACN(14 * 28, d_b, hidden_dim=128,  n_blocks=4)
      self.enc_top    = C_ACN(14 * 28, d_t, hidden_dim=128, n_blocks=4)

      # --- Invertible flows: push embeddings toward N(0,I) ---
      flow_cfg = dict(n_layers=4, hidden_dim=64, n_blocks=4, s_max=2., permute_mode="per_pair")
      self.flow_b = C_InvertibleFlow(d_b, **flow_cfg)
      self.flow_t = C_InvertibleFlow(d_t, **flow_cfg)

      # --- Decoder: latent -> features -> seed map -> upsample to 28x28 ---
      feat = 128  # internal feature width after flow inverse
      self.proj_b = nn.Linear(d_b, feat)
      self.proj_t = nn.Linear(d_t, feat)

      seed_ch, seed_hw = 128, 7  # seed feature map: 64 channels x 7x7
      self.fc = nn.Sequential(
         nn.Linear(2 * feat, seed_ch * seed_hw * seed_hw),
         nn.ReLU(inplace=True),
      )
      self.upsample = nn.Sequential(
         nn.ConvTranspose2d(seed_ch,  64, 4, stride=2, padding=1), nn.ReLU(inplace=True),  # 7x7 -> 14x14
         nn.Conv2d(64, 64, 3, padding=1),                         nn.ReLU(inplace=True),  # refine at 14x14
         nn.ConvTranspose2d(64,       32, 4, stride=2, padding=1), nn.ReLU(inplace=True),  # 14x14 -> 28x28
         nn.Conv2d(32, 32, 3, padding=1),                         nn.ReLU(inplace=True),  # refine at 28x28
         nn.Conv2d(32, 1, 3, padding=1),
      )
      self._seed_ch = seed_ch
      self._seed_hw = seed_hw

   ########################################################################################################################################################################################
   # Encoding
   ########################################################################################################################################################################################
   def EncodeBottom(self, bottom):
      """Encode bottom half -> z_b through encoder + flow."""
      return self.flow_b(self.enc_bottom(bottom.view(-1, 14 * 28)))

   def EncodeTop(self, top):
      """Encode top half -> z_t through encoder + flow."""
      return self.flow_t(self.enc_top(top.view(-1, 14 * 28)))

   ########################################################################################################################################################################################
   # Decoding
   ########################################################################################################################################################################################
   def Decode(self, z_b, z_t):
      """Decode latents (z_b, z_t) -> full 28x28 image."""
      yb = self.proj_b(self.flow_b.inverse(z_b))
      yt = self.proj_t(self.flow_t.inverse(z_t))
      y  = torch.cat([yb, yt], dim=-1)
      h  = self.fc(y).view(-1, self._seed_ch, self._seed_hw, self._seed_hw)
      res = self.upsample(h)
      return torch.sigmoid(res) + 0.01 * (res - res.detach()) # avoiding all-black collupse at the start of training

##########################################################################################################################################################################################
class C_TrainStep(nn.Module):
   """Wraps model + losses + wristband into a single compiled forward pass.

   Returns (total_loss, rec_loss, conditional_loss, wristband_loss).
   """

   def __init__(self, model: C_InpaintMNIST, wristband: C_WristbandGaussianLoss,
                n_of_cf_samples: int, loss_mode: str,
                w_rec: float, w_cf: float, w_wb: float,
                noise_t: float = 0.0, adv_t: float = 0.0):
      super().__init__()
      self.model     = model
      self.wristband = wristband
      self.n_of_cf_samples = n_of_cf_samples
      self.loss_mode = loss_mode
      self.w_rec = w_rec
      self.w_cf  = w_cf
      self.w_wb  = w_wb
      self.noise_t = noise_t
      self.adv_t   = adv_t

   def _PixelLoss(self, a, b):
      """Per-pixel reconstruction loss (L1 or L2)."""
      d = a - b
      if self.loss_mode == "l1":
         return d.abs().mean()
      return d.square().mean()

   def forward(self, x):
      top, bottom = SplitTopBottom(x)
      B = x.size(0)
      K = self.n_of_cf_samples

      z_b = self.model.EncodeBottom(bottom)
      z_t = self.model.EncodeTop(top)

      # --- Optional z_t regularization specifically for exact reconstruction ---
      z_t_rec = z_t
      if self.noise_t > 0.0:
         z_t_rec = z_t + torch.randn_like(z_t) * self.noise_t
      elif self.adv_t > 0.0:
         # Create a detached leaf tensor so gradient tracking is perfectly isolated
         z_in = z_t.detach().requires_grad_(True)
         loss_t = self._PixelLoss(self.model.Decode(z_b.detach(), z_in), x)
         g = torch.autograd.grad(loss_t, z_in, create_graph=False)[0]
         z_t_rec = z_t + self.adv_t * torch.nn.functional.normalize(g, dim=-1)

      # (1) Full reconstruction: both halves encoded, decode, compare
      x_hat = self.model.Decode(z_b, z_t_rec)
      rec = self._PixelLoss(x_hat, x)

      # (2) Conditional reconstruction: encode bottom only, sample z_t ~ N(0,I)
      #     K times, average the LOSS (not the output) over samples
      eps_t   = torch.randn(K, B, self.model.d_t, device=x.device, dtype=z_b.dtype)
      z_b_rep = z_b.unsqueeze(0).expand(K, B, self.model.d_b)
      x_hat_k = self.model.Decode(
         z_b_rep.reshape(K * B, self.model.d_b),
         eps_t.reshape(K * B, self.model.d_t),
      ).reshape(K, B, 1, 28, 28)
      cf = self._PixelLoss(x_hat_k, x.unsqueeze(0))

      # (3) Wristband Gaussian loss on joint latent (uses unmodified z_t explicitly)
      z  = torch.cat([z_b, z_t], dim=-1)
      wb = self.wristband(z).total

      loss = self.w_rec * rec + self.w_cf * cf + self.w_wb * wb
      return loss, rec.detach(), cf.detach(), wb.detach()


##########################################################################################################################################################################################
@torch.no_grad()
def Evaluate(model: C_InpaintMNIST, wristband: C_WristbandGaussianLoss,
             loader: DataLoader, device,
             n_of_cf_samples: int, loss_mode: str,
             w_rec: float, w_cf: float, w_wb: float):
   """Evaluate on a full data loader; returns (total_loss, rec, cf, wb)."""
   model.eval()

   def _pixel_loss(a, b):
      d = a - b
      if loss_mode == "l1":
         return d.abs().mean()
      return d.square().mean()

   sum_rec, sum_cf, sum_wb = 0., 0., 0.
   n_of_samples = 0
   K = n_of_cf_samples

   for x, _ in loader:
      x = x.to(device, non_blocking=True)
      top, bottom = SplitTopBottom(x)
      B = x.size(0)

      z_b = model.EncodeBottom(bottom)
      z_t = model.EncodeTop(top)

      x_hat = model.Decode(z_b, z_t)
      rec = _pixel_loss(x_hat, x)

      eps_t   = torch.randn(K, B, model.d_t, device=device, dtype=z_b.dtype)
      z_b_rep = z_b.unsqueeze(0).expand(K, B, model.d_b)
      x_hat_k = model.Decode(
         z_b_rep.reshape(K * B, model.d_b),
         eps_t.reshape(K * B, model.d_t),
      ).reshape(K, B, 1, 28, 28)
      cf = _pixel_loss(x_hat_k, x.unsqueeze(0))

      z   = torch.cat([z_b, z_t], dim=-1)
      wbl = wristband(z).total

      sum_rec += rec.item() * B
      sum_cf  += cf.item()  * B
      sum_wb  += wbl.item() * B
      n_of_samples += B

   avg_rec = sum_rec / n_of_samples
   avg_cf  = sum_cf  / n_of_samples
   avg_wb  = sum_wb  / n_of_samples
   avg_loss = w_rec * avg_rec + w_cf * avg_cf + w_wb * avg_wb
   model.train()
   return avg_loss, avg_rec, avg_cf, avg_wb


##########################################################################################################################################################################################
@torch.no_grad()
def ShowConditionalSamples(model: C_InpaintMNIST, x, device, n_of_samples: int = 5):
   """For each of `n_of_samples` digits, show the original bottom half with
   `n_of_samples` different sampled top halves (one per column)."""
   model.eval()
   N = n_of_samples
   x = x[:N].to(device)
   _, bottom = SplitTopBottom(x)

   z_b     = model.EncodeBottom(bottom)                         # (N, d_b)
   z_t     = torch.randn(N, N, model.d_t, device=device)        # (cols, rows, d_t)
   z_b_rep = z_b.unsqueeze(0).expand(N, N, model.d_b)           # (cols, rows, d_b)

   x_hat = model.Decode(
      z_b_rep.reshape(N * N, model.d_b),
      z_t.reshape(N * N, model.d_t),
   ).reshape(N, N, 1, 28, 28).permute(1, 0, 2, 3, 4).contiguous()  # (rows, cols, 1, 28, 28)

   # Paste the real bottom half so only the sampled top varies
   x_hat[:, :, :, 14:, :] = bottom.unsqueeze(1)

   grid = make_grid(x_hat.reshape(N * N, 1, 28, 28).clamp(0, 1), nrow=N, padding=2)
   plt.figure(figsize=(6, 6))
   plt.title("Conditional samples: same bottom, different sampled tops")
   plt.imshow(grid[0].cpu().numpy(), cmap="gray")
   plt.axis("off")
   plt.tight_layout()
   plt.show()


##########################################################################################################################################################################################
def _PrintHeader():
   print("         epoch     loss      rec       cf       wb")
   print("        ------   ------   ------   ------   ------")


def _PrintRow(tag: str, epoch: int, loss: float, rec: float, cf: float, wb: float):
   print(f" {tag:>6s} {epoch:5d}   {loss:6.4f}   {rec:6.4f}   {cf:6.4f}   {wb:6.4f}")


##########################################################################################################################################################################################
def Main():
   torch.manual_seed(0)
   torch.backends.cudnn.benchmark = True
   if torch.cuda.is_available():
      torch.backends.cuda.matmul.allow_tf32 = True
      torch.backends.cudnn.allow_tf32       = True

   device = "cuda" if torch.cuda.is_available() else "cpu"
   print(f"Device: {device}\n")

   # -- Data --------------------------------------------------------------
   BATCH = 1024
   train_full = datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor())

   n_of_total = len(train_full)
   n_of_test  = n_of_total // 10
   n_of_train = n_of_total - n_of_test
   train_ds, test_ds = torch.utils.data.random_split(
      train_full, [n_of_train, n_of_test],
      generator=torch.Generator().manual_seed(42),
   )

   loader_cfg = dict(batch_size=BATCH, drop_last=True, num_workers=2,
                     persistent_workers=True, pin_memory=(device == "cuda"), prefetch_factor=4)
   train_loader = DataLoader(train_ds, shuffle=True,  **loader_cfg)
   test_loader  = DataLoader(test_ds,  shuffle=False, **loader_cfg)

   # -- Model -------------------------------------------------------------
   #  d_b=18: latent dims for the bottom half (observed / conditioned on)
   #  d_t=3:  latent dims for the top half (sampled at inference)
   model = C_InpaintMNIST(d_b=18, d_t=3).to(device)
   print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

   # -- Wristband Gaussian loss -------------------------------------------
   #  Calibrated reference statistics for N(0,I) samples of shape (BATCH, d).
   wristband = C_WristbandGaussianLoss(calibration_shape=(BATCH, model.d))
   print(f"Wristband calibrated: shape=({BATCH}, {model.d})\n")

   # -- Training setup ----------------------------------------------------
   N_OF_CYCLES    = 6
   LR             = 2.e-3
   LOSS_MODE      = "l1"
   N_OF_CF_SAMPLES = 32        # K: number of z_t samples for conditional loss
   W_REC, W_CF, W_WB = 1., 0.1, 0.1
   NOISE_T, ADV_T = 0., 0.15   # Regularization to force smoothness in z_t space

   step_fn = C_TrainStep(model, wristband,
                         n_of_cf_samples=N_OF_CF_SAMPLES, loss_mode=LOSS_MODE,
                         w_rec=W_REC, w_cf=W_CF, w_wb=W_WB,
                         noise_t=NOISE_T, adv_t=ADV_T).to(device)
   step_fn = torch.compile(step_fn, mode="default") #"max-autotune-no-cudagraphs") #"default")

   opt = torch.optim.AdamW(model.parameters(), lr=LR)

   n_of_batches = len(train_loader)
   warmup_steps = 20 * n_of_batches

   scheduler = C_CosineAnnealingWarmRestartsDecay(
      opt,
      t_0=4 * n_of_batches,
      t_mult=1.5,
      eta_min=0.05 * LR,
      decay=0.1,
      warmup_steps=warmup_steps,
      warmup_start_factor=0.1,
   )

   # -- Training loop -----------------------------------------------------
   print("=" * 60)
   print(f" WARM-UP PHASE  ({warmup_steps} steps ~= {warmup_steps // n_of_batches} epochs)")
   print("=" * 60)
   _PrintHeader()

   epoch = 0
   global_step = 0
   cycles_done = 0
   warmup_announced = False

   while cycles_done < N_OF_CYCLES:
      model.train()
      epoch += 1
      sum_l = sum_r = sum_c = sum_w = 0.
      n_of_batches_done = 0
      cycle_ended = False

      for x, _ in train_loader:
         x = x.to(device, non_blocking=True)
         loss, rec, cf, wbl = step_fn(x)

         opt.zero_grad(set_to_none=True)
         loss.backward()
         nn.utils.clip_grad_norm_(model.parameters(), 1., foreach=True)
         opt.step()
         scheduler.step()
         global_step += 1

         sum_l += loss.item()
         sum_r += rec.item()
         sum_c += cf.item()
         sum_w += wbl.item()
         n_of_batches_done += 1

         # Announce transition from warm-up to cosine annealing
         if not warmup_announced and global_step >= warmup_steps:
            warmup_announced = True
            n = n_of_batches_done
            _PrintRow("train", epoch, sum_l / n, sum_r / n, sum_c / n, sum_w / n)
            print("=" * 60)
            print(f" COSINE ANNEALING PHASE  ({N_OF_CYCLES} cycles)")
            print("=" * 60)
            _PrintHeader()

         # Cycle boundary: evaluate and log
         if scheduler.just_restarted:
            cycles_done += 1
            cycle_ended = True
            break

      if cycle_ended:
         tl, tr, tc, tw = Evaluate(model, wristband, test_loader, device,
                                   n_of_cf_samples=N_OF_CF_SAMPLES, loss_mode=LOSS_MODE,
                                   w_rec=W_REC, w_cf=W_CF, w_wb=W_WB)
         _PrintRow("test", epoch, tl, tr, tc, tw)
         print(f"        -- end of cycle {cycles_done} --")
      else:
         if n_of_batches_done > 0:
            n = n_of_batches_done
            _PrintRow("train", epoch, sum_l / n, sum_r / n, sum_c / n, sum_w / n)

   # -- Visualize ---------------------------------------------------------
   print("\n" + "=" * 60)
   print(" DONE - generating conditional samples")
   print("=" * 60)
   x_test, _ = next(iter(test_loader))
   ShowConditionalSamples(model, x_test, device)


if __name__ == "__main__":
   Main()