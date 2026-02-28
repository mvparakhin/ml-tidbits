import torch
import torch.nn as nn
import os, sys
# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embed_models.EmbedModels import C_WristbandGaussianLoss

def _PearsonCorrelation(x: torch.Tensor, y: torch.Tensor) -> float:
   x = x.detach().reshape(-1).cpu().to(torch.float64)
   y = y.detach().reshape(-1).cpu().to(torch.float64)
   x = x - x.mean()
   y = y - y.mean()
   denom = torch.sqrt(x.square().sum() * y.square().sum()).clamp_min(1.e-30)
   return float((x * y).sum() / denom)

def _MakeStructuredSpectralTestBatch(seed: int, n: int, d: int, device, dtype) -> torch.Tensor:
   torch.manual_seed(int(seed))
   x = torch.randn(4, n, d, device=device, dtype=dtype)

   # 0: null Gaussian
   # 1: mean-shifted
   x[1] += 0.35

   # 2: anisotropic
   scales = torch.linspace(.5, 1.5, d, device=device, dtype=dtype)
   x[2] *= scales

   # 3: partial near-duplicates / cluster
   x[3, :n // 4] = x[3, 0:1] + .01 * torch.randn_like(x[3, :n // 4])

   return x

def TestWristbandSpectralApproximation(*,
   device: str | torch.device = "cpu",
   dtype: torch.dtype = torch.float32,
   beta: float = 8.,
   k_modes: int = 6,
   n: int = 1024,
   dims: tuple[int, ...] = (16, 64, 128, 256),
   seeds: tuple[int, ...] = (0, 1, 2, 3),
):
   """Diagnostic comparison of exact global/chordal wristband repulsion
   against the spectral approximation.

   This test intentionally uses calibration_shape=None so the raw repulsion
   values are directly comparable.

   Printed columns:
      mean|d_rep|   mean absolute error in raw repulsion
      max|d_rep|    worst absolute error in raw repulsion
      corr(rep)     Pearson correlation of raw repulsion across cases
      mean grad cos average cosine similarity of exact vs spectral gradients
      min grad cos  worst gradient cosine across seeds
      max|d_rad|    should be ~0; otherwise the merge broke radial loss
      max|d_mom|    should be ~0; otherwise the merge broke moment loss
   """
   device = torch.device(device)
   tol_shared = 1.e-6 if dtype == torch.float32 else 1.e-12

   print("Comparing exact wristband loss vs spectral approximation")
   print(f"device={device} dtype={dtype} beta={beta} k_modes={k_modes} n={n}")
   print("   d   mean|d_rep|    max|d_rep|    corr(rep)   mean grad cos   min grad cos    max|d_rad|    max|d_mom|")

   results = {}

   for d in dims:
      exact_rep = C_WristbandGaussianLoss(
         beta=beta,
         angular="chordal",
         reduction="global",
         spectral=False,
         lambda_rad=0.,
         lambda_ang=0.,
         lambda_mom=0.,
         calibration_shape=None,
      )
      spec_rep = C_WristbandGaussianLoss(
         beta=beta,
         angular="chordal",
         reduction="global",
         spectral=True,
         k_modes=k_modes,
         lambda_rad=0.,
         lambda_ang=0.,
         lambda_mom=0.,
         calibration_shape=None,
      )

      # Separate instance with nonzero shared terms, to make sure we did not accidentally change rad or mom in spectral mode.
      exact_full = C_WristbandGaussianLoss(
         beta=beta,
         angular="chordal",
         reduction="global",
         spectral=False,
         lambda_rad=0.1,
         lambda_ang=0.,
         moment="w2",
         lambda_mom=1.,
         calibration_shape=None,
      )
      spec_full = C_WristbandGaussianLoss(
         beta=beta,
         angular="chordal",
         reduction="global",
         spectral=True,
         k_modes=k_modes,
         lambda_rad=0.1,
         lambda_ang=0.,
         moment="w2",
         lambda_mom=1.,
         calibration_shape=None,
      )

      rep_abs_all = []
      rep_exact_all = []
      rep_spec_all = []
      grad_cos_all = []
      rad_max = 0.
      mom_max = 0.

      for seed in seeds:
         x = _MakeStructuredSpectralTestBatch(seed, n, d, device, dtype)

         with torch.no_grad():
            comp_exact = exact_rep._Compute(x)
            comp_spec = spec_rep._Compute(x)

            comp_exact_full = exact_full._Compute(x)
            comp_spec_full = spec_full._Compute(x)

         assert torch.isfinite(comp_exact.rep).all()
         assert torch.isfinite(comp_spec.rep).all()
         assert torch.isfinite(comp_exact_full.rad).all()
         assert torch.isfinite(comp_spec_full.rad).all()
         assert torch.isfinite(comp_exact_full.mom).all()
         assert torch.isfinite(comp_spec_full.mom).all()

         rep_abs_all.append((comp_exact.rep - comp_spec.rep).abs().cpu())
         rep_exact_all.append(comp_exact.rep.cpu())
         rep_spec_all.append(comp_spec.rep.cpu())

         rad_max = max(rad_max, float((comp_exact_full.rad - comp_spec_full.rad).abs().max()))
         mom_max = max(mom_max, float((comp_exact_full.mom - comp_spec_full.mom).abs().max()))

         xa = x.clone().requires_grad_(True)
         xb = x.clone().requires_grad_(True)

         loss_exact = exact_rep(xa).total
         loss_spec = spec_rep(xb).total
         loss_exact.backward()
         loss_spec.backward()

         ga = xa.grad.reshape(-1)
         gb = xb.grad.reshape(-1)

         assert torch.isfinite(ga).all()
         assert torch.isfinite(gb).all()

         cos = torch.dot(ga, gb) / (ga.norm() * gb.norm() + 1.e-12)
         grad_cos_all.append(float(cos.detach().cpu()))

      rep_abs = torch.cat(rep_abs_all)
      rep_exact_vec = torch.cat(rep_exact_all)
      rep_spec_vec = torch.cat(rep_spec_all)
      #diff = rep_spec_vec - rep_exact_vec
      #bias = float(diff.mean())
      #resid = diff - bias
      #print(f"bias(rep_spec - rep_exact)   = {bias: .6e}")
      #print(f"std(diff - bias)             = {float(resid.std(unbiased=True)): .6e}")
      #print(f"max|diff - bias|             = {float(resid.abs().max()): .6e}")

      mean_abs = float(rep_abs.mean())
      max_abs = float(rep_abs.max())
      corr = _PearsonCorrelation(rep_exact_vec, rep_spec_vec)
      grad_cos_mean = float(sum(grad_cos_all) / len(grad_cos_all))
      grad_cos_min = float(min(grad_cos_all))

      print(
         f"{d:4d}   {mean_abs:11.6e}   {max_abs:11.6e}   {corr:10.6f}   "
         f"{grad_cos_mean:13.6f}   {grad_cos_min:12.6f}   {rad_max:11.3e}   {mom_max:11.3e}"
      )

      assert rad_max < tol_shared, f"rad mismatch too large for d={d}: {rad_max}"
      assert mom_max < tol_shared, f"mom mismatch too large for d={d}: {mom_max}"

      results[int(d)] = dict(
         mean_abs_rep=mean_abs,
         max_abs_rep=max_abs,
         corr_rep=corr,
         mean_grad_cos=grad_cos_mean,
         min_grad_cos=grad_cos_min,
         max_abs_rad=rad_max,
         max_abs_mom=mom_max,
      )

   return results

if __name__ == "__main__":
   TestWristbandSpectralApproximation()