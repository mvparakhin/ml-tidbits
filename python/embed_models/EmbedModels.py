"""
Provides:
  - C_EmbedAttentionModule : multi-head softmax Euclidean attention with learnable keys/values
  - C_ACN                  : auto-compressing network
  - C_PermutationLayer     : deterministic permutation (exact inverse)
  - C_AffineCouplingLayer  : RealNVP-style affine coupling (exact inverse)
  - C_InvertibleFlow       : composable invertible normalizing flow
  - W2ToStandardNormalSq   : squared 2-Wasserstein distance to N(0, I)
  - C_WristbandGaussianLoss: wristband-repulsion loss encouraging N(0, I) latents
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import gammaln, ive, iv

__all__ = [
   "C_EmbedAttentionModule",
   "C_ACN",
   "C_PermutationLayer",
   "C_AffineCouplingLayer",
   "C_InvertibleFlow",
   "W2ToStandardNormalSq",
   "C_WristbandGaussianLoss",
   "S_LossComponents",
]

########################################################################################################################
# Helpers
########################################################################################################################

def EpsForDtype(dtype: torch.dtype, large: bool = False) -> float:
   """Return a small epsilon suitable for *dtype*.

   When *large* is True the returned value is sqrt(eps) -- useful as a
   variance floor where machine-epsilon itself would be too tight.
   """
   eps = torch.finfo(dtype).eps
   return math.sqrt(eps) if large else eps


# ---- Spectral Neumann helpers (used by C_WristbandLoss) ----

@dataclass(frozen=True)
class SpectralNeumannCoefficients:
   """Precomputed spectral coefficients for the Neumann target kernel.

   lam_0 : angular eigenvalue for l=0 (constant spherical harmonic).
   lam_1 : angular eigenvalue for l=1 (linear spherical harmonics).
   a_k   : radial Neumann coefficients, shape (K,).
   """
   lam_0: float
   lam_1: float
   a_k: torch.Tensor


def _LogBesselIve(order: float, c: float) -> float:
   """Logarithm of the exponentially-scaled modified Bessel function I_v(c)*e^{-c}.

   scipy.special.ive underflows to 0 when the order v is large (typical at
   high embedding dimension d, since v = d/2 + ell).  No scipy builtin for
   log(ive) exists (scipy/scipy#12607), so we chain three strategies:

   1. ive(v, c): fast and accurate when it doesn't underflow.
   2. iv(v, c): unscaled; finite for moderate v even when ive rounds to 0.
      We subtract c to recover the log-scaled value.
   3. Leading-term asymptotic: I_v(c) ~ (c/2)^v / Gamma(v+1) when v >> c.
      Accurate in the large-d regime where strategies 1-2 both fail.
   """
   val = float(ive(order, c))
   if val > 0.0 and math.isfinite(val):
      return math.log(val)

   val = float(iv(order, c))
   if val > 0.0 and math.isfinite(val):
      return math.log(val) - c

   # Large-order asymptotic: log(I_v(c) * e^{-c}) ~ -c + v*log(c/2) - log(Gamma(v+1))
   return float(-c + order * math.log(c / 2.0) - gammaln(order + 1.0))


def _AngularEigenvalueL(d: int, beta: float, alpha: float, ell: int) -> float:
   """Angular eigenvalue for degree-ell spherical harmonics.

   lam_ell = Gamma(v+1) * (2/c)^v * I_{ell+v}(c) * e^{-c},
   where v = (d-2)/2, c = 2*beta*alpha^2.  Computed in log-domain to avoid
   overflow/underflow at large d.
   """
   if d < 3:
      raise ValueError("Spectral Neumann path requires d >= 3.")
   nu = 0.5 * (d - 2)
   c  = 2.0 * beta * (alpha ** 2)
   log_prefactor = float(gammaln(nu + 1.0) + nu * (math.log(2.0) - math.log(c)))
   log_lambda    = log_prefactor + _LogBesselIve(nu + ell, c)
   if log_lambda < math.log(float(torch.finfo(torch.float64).tiny)):
      return 0.0
   lam = math.exp(log_lambda)
   if not math.isfinite(lam) or lam < 0.0:
      raise FloatingPointError(f"Invalid angular eigenvalue: {lam}.")
   return lam


def _BuildSpectralNeumannCoefficients(
   d: int, beta: float, alpha: float, k_modes: int,
   *, device: torch.device, dtype: torch.dtype,
) -> SpectralNeumannCoefficients:
   """Precompute lam_0, lam_1 and radial Neumann coefficients a_k.

   Called once at construction time.  The radial eigenfunctions on [0,1]
   with Neumann BCs are f_0(t)=1, f_k(t)=cos(k*pi*t), with eigenvalues
   (2*eigenvalues, actually - we are working in unnormalized space)
   a_k = sqrt(pi/beta) * (1 if k==0 else 2*exp(-pi^2*k^2 / (4*beta))).
   """
   if k_modes < 1:
      raise ValueError("k_modes must be >= 1.")
   lam_0 = _AngularEigenvalueL(d, beta, alpha, ell=0)
   lam_1 = _AngularEigenvalueL(d, beta, alpha, ell=1)
   beta_t  = torch.as_tensor(beta, device=device, dtype=dtype)
   k_range = torch.arange(k_modes, device=device, dtype=dtype)
   pref    = torch.sqrt(torch.pi / beta_t)
   a_k = pref * torch.where(
      k_range == 0,
      torch.ones_like(k_range),
      2.0 * torch.exp(-(torch.pi ** 2) * k_range.square() / (4.0 * beta_t)),
   )
   return SpectralNeumannCoefficients(lam_0=lam_0, lam_1=lam_1, a_k=a_k)


#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# C_EmbedAttentionModule
#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class C_EmbedAttentionModule(nn.Module):
   """Multi-head softmax attention with *learnable* keys and values.

   The query is the input ``x`` (optionally transformed by ``q_transform``).
   Keys ``k`` and values ``v`` are stored as trainable parameter tensors of
   shape ``(n_of_heads, n_of_basis, ...)``.

   Parameters
   ----------
   input_dim : int
       Dimensionality of the input (query) vectors.
   hidden_dim : int
       Per-head value dimensionality.
   out_dim : int
       Final output dimensionality (after ``head_combine``).
   n_of_basis : int
       Number of learnable key/value pairs per head.
   n_of_heads : int
       Number of attention heads.
   is_euclidean : bool
       If True (default) use Euclidean attention: logits = <q,k> - 0.5*||k||^2.
       Otherwise use scaled dot-product attention.
   normalize_k : bool
       If True, L2-normalise keys before computing logits.
   q_transform : nn.Module | None
       Optional transform applied to queries before attention.
   head_combine : nn.Module | None
       Optional projection from concatenated head outputs to *out_dim*.
       Defaults to ``nn.Linear(hidden_dim * n_of_heads, out_dim)`` when
       ``n_of_heads > 1``, or ``nn.Identity()`` otherwise.
   affine_experts : bool
       Enable rank-1 affine expert contribution to values: v(q) = v + v_out * (q * v_in).
   affine_init_scale : float
       Scale factor for affine expert initialisation (smaller -> closer to
       identity at init).
   head_temperature : bool
       Enable a learnable per-head temperature multiplier on logits.
   """

   def __init__(self, input_dim, hidden_dim, out_dim, n_of_basis, n_of_heads, *, is_euclidean: bool = True, normalize_k: bool = False, q_transform=None, head_combine=None,
                affine_experts: bool = False, affine_init_scale: float = 0.1, head_temperature: bool = False):
      super(C_EmbedAttentionModule, self).__init__()

      self.n_of_heads = n_of_heads
      self.hidden_dim = hidden_dim
      self.out_dim = out_dim
      self.n_of_basis = n_of_basis
      self.k_norm = 1. / math.sqrt(input_dim)
      self.is_euclidean = is_euclidean
      self.normalize_k = normalize_k

      self.affine_experts = bool(affine_experts)
      self.head_temperature = bool(head_temperature)

      # Trainable keys, values, and per-head scale
      self.k = nn.Parameter(torch.empty((n_of_heads, n_of_basis, input_dim)))
      self.v = nn.Parameter(torch.empty((n_of_heads, n_of_basis, hidden_dim)))
      self.scale = nn.Parameter(torch.zeros((n_of_heads,)))
      self.layer_norm_scale = nn.Parameter(-torch.ones((n_of_basis)))
      nn.init.xavier_normal_(self.k)
      nn.init.xavier_normal_(self.v)

      # Per-head temperature (always allocated; only used when head_temperature=True)
      self.head_temp = nn.Parameter(torch.zeros((n_of_heads,)))

      # Rank-1 affine expert parameters (always allocated; only used when affine_experts=True)
      self.v_in  = nn.Parameter(torch.empty((n_of_heads, n_of_basis, input_dim)))
      self.v_out = nn.Parameter(torch.empty((n_of_heads, n_of_basis, hidden_dim)))
      if self.affine_experts:
         nn.init.xavier_normal_(self.v_in)
         nn.init.xavier_normal_(self.v_out)
         s = float(affine_init_scale)
         if s != 1.:
            with torch.no_grad():
               self.v_in.mul_(s)
               self.v_out.mul_(s)

      # Optional q transformation and head combination
      self.q_transform = q_transform if q_transform is not None else nn.Identity()
      if n_of_heads > 1:
         self.head_combine = head_combine if head_combine is not None else nn.Linear(hidden_dim * n_of_heads, out_dim)
      else:
         self.head_combine = nn.Identity()

   def forward(self, x):
      orig_shape = x.shape
      q = x.reshape(-1, orig_shape[-1])
      q = self.q_transform(q)
      k = self.k * torch.clip(self.k.square().sum(dim=-1, keepdim=True), min=1.19209e-07**2).rsqrt() if self.normalize_k else self.k

      if self.is_euclidean: #looks suboptimal, but it's faster for whatever reason
         cross_term = torch.einsum('bi,hni->bhn', q, k)
         if not self.normalize_k:
            k_term = torch.sum(self.k.square(), dim=-1) * torch.exp(self.scale)[..., None]
            logits = cross_term - 0.5 * k_term[None, ...]
         else:
            logits = cross_term
      else:
         logits = torch.einsum('bi,hni->bhn', q, k) * self.k_norm

      logits = F.layer_norm(logits, normalized_shape=(logits.size(-1),), weight=torch.exp(self.layer_norm_scale))

      if self.head_temperature: # Optional per-head temperature (rank-1 extension)
         logits = logits * torch.exp(self.head_temp)[None, :, None]

      attention_weights = F.softmax(logits, dim=-1)
      output = torch.einsum('bhn,hnd->bhd', attention_weights, self.v)

      if self.affine_experts: # Optional rank-1 affine expert contribution
         s = torch.einsum('bi,hni->bhn', q, self.v_in) # s[b,h,n] = q[b] * v_in[h,n]
         output = output + torch.einsum('bhn,hnd->bhd', attention_weights * s, self.v_out) # add Sum_n a[b,h,n] * s[b,h,n] * v_out[h,n,:]
      output = output.reshape(-1, self.hidden_dim * self.n_of_heads)
      output = self.head_combine(output)

      new_shape = orig_shape[:-1] + (self.out_dim,)
      output = output.reshape(new_shape)
      return output

#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# Auto-Compressing Network
#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class C_ACN(nn.Module):
   """Residual MLP whose blocks are *added* (not chained) before the output
   projection, yielding a compressed-residual architecture."""

   def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 512, n_blocks: int = 2):
      super().__init__()
      self.in_proj = nn.Linear(in_dim, hidden_dim)
      self.blocks = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_blocks)])
      self.out_proj = nn.Linear(hidden_dim, out_dim)

   def forward(self, x):
      a = self.in_proj(x)
      res = a
      for lin in self.blocks:
         a = lin(F.elu(a))
         res = res + a
      return self.out_proj(F.elu(res))

#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# Deterministic permutation layer (exact inverse)
#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class C_PermutationLayer(nn.Module):
   """Fixed permutation of the last dimension.  ``inverse()`` applies the
   exact inverse permutation."""

   def __init__(self, dim: int, perm: torch.Tensor):
      super().__init__()
      dim = int(dim)
      if dim < 1:
         raise ValueError("dim must be >= 1")
      if perm.ndim != 1 or int(perm.numel()) != dim:
         raise ValueError("perm must have shape (dim,)")

      perm = perm.to(dtype=torch.int64).contiguous()
      inv = torch.empty_like(perm)
      inv[perm] = torch.arange(dim, dtype=torch.int64)

      self.dim = dim
      self.register_buffer("perm", perm)
      self.register_buffer("inv_perm", inv)

   def forward(self, x: torch.Tensor):
      if x.shape[-1] != self.dim:
         raise ValueError(f"Expected last dim == {self.dim}, but got {int(x.shape[-1])}")
      return torch.index_select(x, -1, self.perm)

   @torch.jit.export
   def inverse(self, y: torch.Tensor):
      if y.shape[-1] != self.dim:
         raise ValueError(f"Expected last dim == {self.dim}, but got {int(y.shape[-1])}")
      return torch.index_select(y, -1, self.inv_perm)

#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# Invertible flow: RealNVP-style affine coupling (exact inverse)
#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class C_AffineCouplingLayer(nn.Module):
   """Single affine coupling layer.

   - Works for any last-dim ``D >= 1`` (including odd D).
   - Input can be any shape ``(..., D)``.
   - Log-scale ``s`` is bounded via ``tanh`` for numerical stability.
   """

   def __init__(self, dim: int, mask: torch.Tensor, hidden_dim: int = 128, n_blocks: int = 2, s_max: float = 2.0):
      super().__init__()
      dim = int(dim)
      if dim < 1:
         raise ValueError("dim must be >= 1")

      self.dim = dim
      self.s_max = float(s_max)

      if mask.ndim != 1 or int(mask.numel()) != dim:
         raise ValueError("mask must have shape (dim,)")

      m = mask.to(dtype=torch.float32).contiguous()
      self.register_buffer("mask", m)
      self.register_buffer("inv_mask", 1. - m)

      pass_idx  = torch.nonzero(m >= 0.5, as_tuple=False).flatten().to(dtype=torch.int64)
      trans_idx = torch.nonzero(m <  0.5, as_tuple=False).flatten().to(dtype=torch.int64)

      self.register_buffer("pass_idx", pass_idx)
      self.register_buffer("trans_idx", trans_idx)
      self.pass_dim = int(pass_idx.numel())
      self.trans_dim = int(trans_idx.numel())

      if self.pass_dim < 1 or self.trans_dim < 1:
         self.net = None
         self._contig = True
         self._pass_first = True
         return

      # Fast path detection: contiguous split either [pass | trans] or [trans | pass].
      self._contig = False
      self._pass_first = True
      with torch.no_grad():
         a = torch.arange(self.dim, dtype=torch.int64)
         if torch.equal(pass_idx, a[:self.pass_dim]) and torch.equal(trans_idx, a[self.pass_dim:]): # [0..pass_dim-1] are pass, [pass_dim..D-1] are trans
            self._contig = True
            self._pass_first = True
         elif torch.equal(trans_idx, a[:self.trans_dim]) and torch.equal(pass_idx, a[self.trans_dim:]): # [0..trans_dim-1] are trans, [trans_dim..D-1] are pass
            self._contig = True
            self._pass_first = False
      self.net = C_ACN(self.pass_dim, 2 * self.trans_dim, hidden_dim, n_blocks) # Conditioner: (pass_dim) -> (2 * trans_dim)
      # Identity-ish init: start close to identity for stability.
      with torch.no_grad():
         nn.init.zeros_(self.net.out_proj.weight)
         nn.init.zeros_(self.net.out_proj.bias)

   def _ST(self, x_pass: torch.Tensor):
      st = self.net(x_pass)
      s_raw, t = st.chunk(2, dim=-1)
      s = torch.tanh(s_raw) * self.s_max
      return s, t

   def forward(self, x: torch.Tensor):
      if x.shape[-1] != self.dim:
         raise ValueError(f"Expected last dim == {self.dim}, but got {int(x.shape[-1])}")
      if self.net is None:
         return x

      if self._contig:
         if self._pass_first:
            x_pass  = x[..., :self.pass_dim]
            x_trans = x[..., self.pass_dim:]
            s, t = self._ST(x_pass)
            y_trans = x_trans * torch.exp(s) + t
            return torch.cat((x_pass, y_trans), dim=-1)
         else:
            x_trans = x[..., :self.trans_dim]
            x_pass  = x[..., self.trans_dim:]
            s, t = self._ST(x_pass)
            y_trans = x_trans * torch.exp(s) + t
            return torch.cat((y_trans, x_pass), dim=-1)

      x_pass  = torch.index_select(x, -1, self.pass_idx)
      x_trans = torch.index_select(x, -1, self.trans_idx)
      s, t = self._ST(x_pass)
      y_trans = x_trans * torch.exp(s) + t

      y = torch.empty_like(x)
      y.index_copy_(-1, self.pass_idx, x_pass)
      y.index_copy_(-1, self.trans_idx, y_trans)
      return y

   @torch.jit.export
   def inverse(self, y: torch.Tensor):
      if y.shape[-1] != self.dim:
         raise ValueError(f"Expected last dim == {self.dim}, but got {int(y.shape[-1])}")
      if self.net is None:
         return y

      if self._contig:
         if self._pass_first:
            y_pass  = y[..., :self.pass_dim]
            y_trans = y[..., self.pass_dim:]
            s, t = self._ST(y_pass)
            x_trans = (y_trans - t) * torch.exp(-s)
            return torch.cat((y_pass, x_trans), dim=-1)
         else:
            y_trans = y[..., :self.trans_dim]
            y_pass  = y[..., self.trans_dim:]
            s, t = self._ST(y_pass)
            x_trans = (y_trans - t) * torch.exp(-s)
            return torch.cat((x_trans, y_pass), dim=-1)

      y_pass  = torch.index_select(y, -1, self.pass_idx)
      y_trans = torch.index_select(y, -1, self.trans_idx)
      s, t = self._ST(y_pass)
      x_trans = (y_trans - t) * torch.exp(-s)

      x = torch.empty_like(y)
      x.index_copy_(-1, self.pass_idx, y_pass)
      x.index_copy_(-1, self.trans_idx, x_trans)
      return x

#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# Composable invertible flow
#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class C_InvertibleFlow(nn.Module):
   """Stack of affine coupling layers with optional permutations.

   Parameters
   ----------
   dim : int
       Feature dimensionality.
   n_layers : int
       Number of coupling layers.
   hidden_dim : int
       Hidden width inside each coupling conditioner network.
   n_blocks : int
       Number of residual blocks in each conditioner.
   s_max : float
       Tanh saturation bound for log-scale in coupling layers.
   mask_mode : ``"alternating"`` | ``"half"``
       How binary masks are constructed (only used when ``permute_mode="none"``).
   permute_mode : ``"none"`` | ``"per_layer"`` | ``"per_pair"``
       Random permutation strategy between coupling layers.
   permute_seed : int
       Seed for deterministic permutation generation.
   """

   def __init__(self, dim: int, n_layers: int = 6, hidden_dim: int = 128, n_blocks: int = 2, s_max: float = 2.,
                mask_mode: str = "alternating", *, permute_mode: str = "per_pair", permute_seed: int = 1337):
      super().__init__()
      dim = int(dim)
      if dim < 1:
         raise ValueError("dim must be >= 1")

      self.dim = dim
      self.n_layers = int(n_layers)
      self.hidden_dim = int(hidden_dim)
      self.n_blocks = int(n_blocks)
      self.s_max = float(s_max)

      if mask_mode not in ("alternating", "half"):
         raise ValueError("mask_mode must be 'alternating' or 'half'")
      self.mask_mode = mask_mode

      if permute_mode not in ("none", "per_layer", "per_pair"):
         raise ValueError("permute_mode must be 'none', 'per_layer', or 'per_pair'")
      self.permute_mode = permute_mode
      self.permute_seed = int(permute_seed) & 0x7FFFFFFF

      ops: list[nn.Module] = []
      if self.n_layers > 0 and self.dim >= 2:
         idx = torch.arange(self.dim, dtype=torch.int64)
         d1 = self.dim // 2

         # Base half masks in current coordinate order.
         mask0 = torch.zeros((self.dim,), dtype=torch.float32)
         mask0[:d1] = 1.
         mask1 = 1. - mask0

         if self.permute_mode != "none": # Deterministic permutation stream.
            gen = torch.Generator()
            gen.manual_seed(self.permute_seed)

            for i in range(self.n_layers):
               if (self.permute_mode == "per_layer") or (self.permute_mode == "per_pair" and ((int(i) & 1) == 0)):
                  perm = torch.randperm(self.dim, generator=gen)
                  ops.append(C_PermutationLayer(self.dim, perm))

               m = mask0 if ((int(i) & 1) == 0) else mask1 # Always alternate half-mask and its complement.
               ops.append(C_AffineCouplingLayer(self.dim, m, hidden_dim=self.hidden_dim, n_blocks=self.n_blocks, s_max=self.s_max))
         else:
            for i in range(self.n_layers):
               if self.mask_mode == "alternating":
                  m = ((idx + int(i)) & 1).to(dtype=torch.float32)
               else:  # "half"
                  m = mask0 if ((int(i) & 1) == 0) else mask1
               ops.append(C_AffineCouplingLayer(self.dim, m, hidden_dim=self.hidden_dim, n_blocks=self.n_blocks, s_max=self.s_max))

      self.ops = nn.ModuleList(ops)

   def forward(self, x: torch.Tensor):
      if x.shape[-1] != self.dim:
         raise ValueError(f"Expected last dim == {self.dim}, but got {int(x.shape[-1])}")
      y = x
      for op in self.ops:
         y = op(y)
      return y

   @torch.jit.export
   def inverse(self, y: torch.Tensor):
      if y.shape[-1] != self.dim:
         raise ValueError(f"Expected last dim == {self.dim}, but got {int(y.shape[-1])}")
      x = y
      for op in self.ops[::-1]:
         x = op.inverse(x)
      return x

#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# Squared 2-Wasserstein distance to N(0, I)
#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

def W2ToStandardNormalSq(x: torch.Tensor, *, reduction: str = "mean") -> torch.Tensor:
   r"""Squared 2-Wasserstein distance between the Gaussian fit to samples *x*
   and :math:`\mathcal{N}(0, I)`.

   .. math::

      W_2^2 = \|\mu\|^2 + \sum_i (\sqrt{\lambda_i} - 1)^2

   where :math:`\lambda_i` are eigenvalues of the sample covariance of *x*.

   Parameters
   ----------
   x : Tensor
       Shape ``(..., B, d)`` where ``B`` is the number of samples and ``d``
       the feature dimension.
   reduction : ``"none"`` | ``"mean"`` | ``"sum"``

   Returns
   -------
   Tensor
       ``(...)`` when ``reduction="none"``, scalar otherwise.
   """
   if x.ndim < 2:
      raise ValueError(f"Expected x.ndim>=2 with shape (..., B, d), got {tuple(x.shape)}")
   b = x.shape[-2]
   d = x.shape[-1]
   if b < 2:
      raise ValueError("Need B>=2 for covariance (denominator B-1).")

   work_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
   xw = x.to(dtype=work_dtype)

   mu = xw.mean(dim=-2, keepdim=True)          # (..., 1, d)
   xc = xw - mu                                # (..., B, d)
   mu2 = mu.squeeze(-2).square().sum(dim=-1)   # (...,)
   denom = float(b - 1)

   if d <= b: # Choose smaller PSD matrix to eigendecompose
      m = (xc.transpose(-1, -2) @ xc) / denom # Covariance: (..., d, d)
      m_dim = d
   else:
      m = (xc @ xc.transpose(-1, -2)) / denom # Gram: (..., B, B)
      m_dim = b

   m = .5 * (m + m.transpose(-1, -2))

   # Eigenvalues of PSD matrix (sorted). For Gram, these are the nonzero eigenvalues of Sigma.
   eig = torch.linalg.eigvalsh(m)
   eig = eig.clamp_min(0.)

   sqrt_eig = torch.sqrt(eig + EpsForDtype(eig.dtype))
   bw2 = (sqrt_eig - 1.).square().sum(dim=-1)

   # If we used Gram (m_dim=B<d), Sigma has (d-m_dim) additional zero eigenvalues, each contributing (sqrt(0)-1)^2 = 1
   if d > m_dim:
      bw2 = bw2 + (d - m_dim)

   loss = mu2 + bw2 # canonical coefficient on mean term is 1

   if reduction == "none":
      return loss
   elif reduction == "mean":
      return loss.mean()
   elif reduction == "sum":
      return loss.sum()
   else:
      raise ValueError("reduction must be one of {'none','mean','sum'}")

#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# Wristband Gaussian Loss
#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class S_LossComponents(NamedTuple):
   """Named tuple returned by :class:`C_WristbandGaussianLoss`."""
   total: torch.Tensor
   rep: torch.Tensor
   rad: torch.Tensor
   ang: torch.Tensor
   mom: torch.Tensor

class C_WristbandGaussianLoss:
   r"""Batch loss encouraging :math:`x \sim \mathcal{N}(0, I)` via wristband
   repulsion on the (direction, radius) decomposition of samples, with optional
   marginal-uniformity and moment-matching penalties.

   The loss maps each sample to a *wristband* representation ``(u, t)`` where
   ``u`` is the unit direction and ``t = gammainc(d/2, ||x||^2/2)`` is the
   CDF-transformed radius (uniform under the null). Repulsion can be computed
   either exactly with the pairwise reflected kernel, or approximately with a
   spectral Neumann expansion. 

   All component losses are calibrated by Monte-Carlo sampling from the null 
   distribution at construction time, so the returned ``total`` is a
   zero-mean, unit-variance z-score under :math:`\mathcal{N}(0, I)`.

   Parameters
   ----------
   beta : float
       Bandwidth parameter for the Gaussian kernel in the repulsion term.
   alpha : float | None
       Coupling constant between angular and radial scales. ``None`` picks
       a heuristic default that balances the two.
   angular : ``"chordal"`` | ``"geodesic"``
       Metric on the unit sphere for the angular component.
   reduction : ``"per_point"`` | ``"global"``
       Whether repulsion is averaged per-row (per-point) or globally.
   spectral : bool
       If ``True``, use the O(N d K) spectral Neumann approximation for the
       repulsion term. This currently supports only ``angular="chordal"``,
       ``reduction="global"``, ``lambda_ang=0`` and ``d >= 3``.
   k_modes : int
       Number of radial Neumann modes used when ``spectral=True``.
   lambda_rad, lambda_ang, lambda_mom : float
       Weights for the radial-uniformity, angular-uniformity, and moment
       penalty components.
   moment : str
       Moment penalty type. One of ``"mu_only"``, ``"kl_diag"``,
       ``"kl_full"``, ``"jeff_diag"``, ``"jeff_full"``, ``"w2"``.
   calibration_shape : tuple[int, int] | None
       ``(N, D)`` shape for Monte-Carlo calibration. If provided the loss
       components are normalised to zero mean / unit variance under the null.
   calibration_reps : int
       Number of Monte-Carlo repetitions for calibration.
   calibration_device, calibration_dtype
       Device and dtype for calibration samples.

   Example
   -------
   >>> loss_fn = C_WristbandGaussianLoss(calibration_shape=(256, 8))
   >>> z = torch.randn(256, 8)
   >>> lc = loss_fn(z)
   >>> lc.total.backward()
   >>> loss_fn_spec = C_WristbandGaussianLoss(
   ...    spectral=True, reduction="global", calibration_shape=(256, 8)
   ... )
   """

   def __init__(self, *,
      beta: float = 8.,
      alpha: float | None = None,
      angular: str = "chordal",     # "chordal" or "geodesic"
      reduction: str = "per_point", # "per_point" or "global"
      spectral: bool = False,
      k_modes: int = 6,
      lambda_rad: float = 0.1,
      lambda_ang: float = 0.,
      moment: str = "w2",           # "mu_only" | "kl_diag" | "kl_full" | "jeff_diag" | "jeff_full" | "w2"
      lambda_mom: float = 1.,
      calibration_shape: tuple[int, int] | None = None, # (N, D)
      calibration_reps: int = 1024,
      calibration_device: str | torch.device = "cpu",
      calibration_dtype: torch.dtype = torch.float32,
   ):
      if beta <= 0:
         raise ValueError("beta must be > 0")
      if angular not in ("chordal", "geodesic"):
         raise ValueError("angular must be 'chordal' or 'geodesic'")
      if reduction not in ("per_point", "global"):
         raise ValueError("reduction must be 'per_point' or 'global'")
      if moment not in ("mu_only", "kl_diag", "kl_full", "jeff_diag", "jeff_full", "w2"):
         raise ValueError("moment must be 'mu_only', 'kl_diag', 'kl_full', 'jeff_diag', 'jeff_full' or 'w2'")
      if int(k_modes) < 1:
         raise ValueError("k_modes must be >= 1")
      if spectral and angular != "chordal":
         raise ValueError("spectral=True currently supports only angular='chordal'")
      if spectral and reduction != "global":
         raise ValueError("spectral=True currently supports only reduction='global'")
      if spectral and lambda_ang != 0.:
         raise ValueError("spectral=True currently supports only lambda_ang=0")

      self.beta = float(beta)
      self.angular = angular
      self.reduction = reduction
      self.spectral = bool(spectral)
      self.k_modes = int(k_modes)

      if alpha is None:
         if angular == "chordal":
            alpha = math.sqrt(1. / 12.) # heuristic so E[(t_i-t_j)^2] (~1/6) matches E[alpha^2 * chordal^2] (E[chordal^2]=2)
         else:  # geodesic
            alpha = math.sqrt(2. / (3. * math.pi * math.pi)) # heuristic so alpha^2 * (pi/2)^2 ~ 1/6
      self.alpha = float(alpha)
      self.beta_alpha2 = self.beta * (self.alpha * self.alpha)

      self.lambda_rad = float(lambda_rad)
      self.lambda_ang = float(lambda_ang)
      self.moment = moment
      self.lambda_mom = float(lambda_mom)
      self.eps = 1.e-12
      self.clamp_cos = 1.e-6
      self._spectral_cache = {}

      # Calibration statistics (identity transform when not calibrated)
      self.mean_rep = self.mean_rad = self.mean_ang = self.mean_mom = 0.
      self.std_rep = self.std_rad = self.std_ang = self.std_mom = 1.
      self.std_total = 1.

      if calibration_shape is not None:
         self._Calibrate(calibration_shape, calibration_reps, calibration_device, calibration_dtype)

   # ---- helpers ----

   def _GetSpectralCoefficients(self, d: int, device: torch.device, dtype: torch.dtype):
      if d < 3:
         raise ValueError("spectral=True requires d >= 3")
      key = (int(d), str(device), dtype)
      coeffs = self._spectral_cache.get(key)
      if coeffs is None:
         coeffs = _BuildSpectralNeumannCoefficients(
            d=int(d), beta=self.beta, alpha=self.alpha, k_modes=self.k_modes,
            device=device, dtype=dtype,
         )
         self._spectral_cache[key] = coeffs
      return coeffs

   def _MomentPenalty(self, xw: torch.Tensor) -> torch.Tensor:
      batch_shape = xw.shape[:-2]
      mom_pen = xw.new_zeros(batch_shape)
      if self.lambda_mom == 0.:
         return mom_pen

      n = int(xw.shape[-2])
      d = int(xw.shape[-1])
      n_f, d_f = float(n), float(d)
      eps = self.eps

      if self.moment == "w2":
         return W2ToStandardNormalSq(xw, reduction="none") / d_f

      mu = xw.mean(dim=-2)

      if self.moment == "mu_only":
         return mu.square().mean(dim=-1)

      xc = xw - mu[..., None, :]

      if self.moment == "jeff_diag": # 0.5 * Jeffreys = 0.25*( v + 1/v + mu^2 + mu^2/v - 2 )
         var = xc.square().sum(dim=-2) / (n_f - 1.)
         v = var + eps
         inv_v = v.reciprocal()
         mu2 = mu.square()
         return .25 * (v + inv_v + mu2 + mu2 * inv_v - 2.).mean(dim=-1)

      if self.moment == "jeff_full": # 0.5 * Jeffreys (full cov), normalized by d to match per-dim scale: jeff = 0.25*(tr(S)+tr(S^{-1}) + ||mu||^2 + mu^T S^{-1} mu - 2d) / d
         eps_cov = max(eps, 1.e-6) if xw.dtype == torch.float32 else max(eps, float(torch.finfo(xw.dtype).eps))
         cov = (xc.transpose(-1, -2) @ xc) / (n_f - 1.)
         eye = torch.eye(d, device=xw.device, dtype=xw.dtype)
         cov = cov + eps_cov * eye
         chol, _ = torch.linalg.cholesky_ex(cov)
         tr = cov.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
         inv_cov = torch.cholesky_solve(eye, chol)
         tr_inv = inv_cov.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
         mu_col = mu[..., :, None]
         sol_mu = torch.cholesky_solve(mu_col, chol)
         mu_inv_mu = (mu_col * sol_mu).sum(dim=(-2, -1))
         mu2_sum = mu.square().sum(dim=-1)
         return .25 * (tr + tr_inv + mu2_sum + mu_inv_mu - 2. * d_f) / d_f

      if self.moment == "kl_diag": # KL(N(mu, diag(var)) || N(0, I)) averaged per-dim
         var = xc.square().sum(dim=-2) / (n_f - 1.)
         return 0.5 * (var + mu.square() - 1. - torch.log(var + eps)).mean(dim=-1)

      eye = torch.eye(d, device=xw.device, dtype=xw.dtype)
      cov = (xc.transpose(-1, -2) @ xc) / n_f + eps * eye
      chol, _ = torch.linalg.cholesky_ex(cov)
      diag = chol.diagonal(dim1=-2, dim2=-1)
      logdet = 2.0 * torch.log(diag).sum(dim=-1)
      tr = cov.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
      mu2 = mu.square().sum(dim=-1)
      return 0.5 * (tr + mu2 - d_f - logdet) / d_f

   def _WristbandMap(self, xw: torch.Tensor):
      d_f = float(xw.shape[-1])
      s = xw.square().sum(dim=-1).clamp_min(self.eps)   # (..., n)
      u = xw * torch.rsqrt(s)[..., :, None]             # (..., n, d)
      a_df = s.new_tensor(.5 * d_f)
      t = torch.special.gammainc(a_df, .5 * s).clamp(self.eps, 1. - self.eps) # (..., n)
      return u, t

   def _RadialLoss(self, t: torch.Tensor, n_f: float, dtype: torch.dtype) -> torch.Tensor:
      t_sorted, _ = torch.sort(t, dim=-1)
      q = (torch.arange(int(t.shape[-1]), device=t.device, dtype=dtype) + .5) / n_f
      return 12. * (t_sorted - q).square().mean(dim=-1)

   def _AngularExponent(self, u: torch.Tensor) -> torch.Tensor:
      g = (u @ u.transpose(-1, -2)).clamp(-1., 1.)    # (..., n, n)

      if self.angular == "chordal": # chordal^2 = ||u_i-u_j||^2 = 2 - 2g, e_ang = -beta * alpha^2 * chordal^2 = 2*beta*alpha^2*(g-1)
         return (2. * self.beta_alpha2) * (g - 1.)

      theta = torch.acos(g.clamp(-1. + self.clamp_cos, 1. - self.clamp_cos))
      ang2 = theta.square()
      ang2 = ang2 - torch.diag_embed(ang2.diagonal(dim1=-2, dim2=-1)) # zero diag without fill_diagonal_ (works for batched)
      return -self.beta_alpha2 * ang2  # diag 0

   def _AngularUniformity(self, e_ang: torch.Tensor, n_f: float) -> torch.Tensor:
      if self.reduction == "per_point":
         row_sum = torch.exp(e_ang).sum(dim=-1) - 1.
         mean_k = row_sum / (n_f - 1.)
         return torch.log(mean_k + self.eps).mean(dim=-1) / self.beta

      total = torch.exp(e_ang).sum(dim=(-2, -1)) - n_f
      mean_k = total / (n_f * (n_f - 1.))
      return torch.log(mean_k + self.eps) / self.beta

   def _PairwiseRepulsion(self, e_ang: torch.Tensor, t: torch.Tensor, n_f: float) -> torch.Tensor:
      tc = t[..., :, None]
      tr = t[..., None, :]
      diff0 = tc - tr
      diff1 = tc + tr
      diff2 = diff1 - 2.

      if self.reduction == "per_point":
         row_sum  = torch.exp(torch.addcmul(e_ang, diff0, diff0, value=-self.beta)).sum(dim=-1)
         row_sum += torch.exp(torch.addcmul(e_ang, diff1, diff1, value=-self.beta)).sum(dim=-1)
         row_sum += torch.exp(torch.addcmul(e_ang, diff2, diff2, value=-self.beta)).sum(dim=-1)
         row_sum -= 1. # remove only the real self term (=1), keep diagonal mirror terms
         mean_k = row_sum / (3. * n_f - 1.)
         return torch.log(mean_k + self.eps).mean(dim=-1) / self.beta

      total  = torch.exp(torch.addcmul(e_ang, diff0, diff0, value=-self.beta)).sum(dim=(-2, -1))
      total += torch.exp(torch.addcmul(e_ang, diff1, diff1, value=-self.beta)).sum(dim=(-2, -1))
      total += torch.exp(torch.addcmul(e_ang, diff2, diff2, value=-self.beta)).sum(dim=(-2, -1))
      total -= n_f # remove n real-self terms (=1)
      mean_k = total / (3. * n_f * n_f - n_f)
      return torch.log(mean_k + self.eps) / self.beta

   def _SpectralRepulsion(self, u: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
      d = int(u.shape[-1])
      n_f, d_f = float(u.shape[-2]), float(d)
      coeffs = self._GetSpectralCoefficients(d, u.device, u.dtype)

      a_k = coeffs.a_k.to(device=t.device, dtype=t.dtype)
      k_range = torch.arange(int(a_k.shape[0]), device=t.device, dtype=t.dtype)
      cos_mat = torch.cos(torch.pi * t[..., :, None] * k_range)        # (..., n, k)
      c_0k = cos_mat.mean(dim=-2)                                      # (..., k)
      c_1k = (math.sqrt(d_f) / n_f) * (u.transpose(-1, -2) @ cos_mat) # (..., d, k)

      lam_0_t = torch.as_tensor(coeffs.lam_0, device=t.device, dtype=t.dtype)
      lam_1_t = torch.as_tensor(coeffs.lam_1, device=t.device, dtype=t.dtype)

      e_total = lam_0_t * (a_k * c_0k.square()).sum(dim=-1)
      e_total += lam_1_t * (a_k * c_1k.square()).sum(dim=(-2, -1))

      norm_const = torch.clamp_min(lam_0_t * a_k[0], self.eps)
      return torch.log(torch.clamp_min(e_total / norm_const, self.eps)) / self.beta

   # ---- calibration ----

   def _Calibrate(self, shape: tuple[int, int], reps: int, device, dtype):
      n, d = shape
      if n < 2 or d < 1 or reps < 2:
         return

      sum_rep, sum_rad, sum_ang, sum_mom = 0., 0., 0., 0.
      sum2_rep, sum2_rad, sum2_ang, sum2_mom = 0., 0., 0., 0.
      all_rep, all_rad, all_ang, all_mom = [], [], [], []

      with torch.no_grad():
         for _ in range(int(reps)):
            x_gauss = torch.randn(int(n), int(d), device=device, dtype=dtype)
            comp = self._Compute(x_gauss)

            f_rep, f_rad, f_ang, f_mom = float(comp.rep), float(comp.rad), float(comp.ang), float(comp.mom)
            sum_rep += f_rep;  sum2_rep += f_rep * f_rep;  all_rep.append(f_rep)
            sum_rad += f_rad;  sum2_rad += f_rad * f_rad;  all_rad.append(f_rad)
            sum_ang += f_ang;  sum2_ang += f_ang * f_ang;  all_ang.append(f_ang)
            sum_mom += f_mom;  sum2_mom += f_mom * f_mom;  all_mom.append(f_mom)

      reps_f = float(reps)
      bessel = reps_f / (reps_f - 1.)

      self.mean_rep = sum_rep / reps_f
      self.mean_rad = sum_rad / reps_f
      self.mean_ang = sum_ang / reps_f
      self.mean_mom = sum_mom / reps_f

      var_rep = (sum2_rep / reps_f - self.mean_rep * self.mean_rep) * bessel
      var_rad = (sum2_rad / reps_f - self.mean_rad * self.mean_rad) * bessel
      var_ang = (sum2_ang / reps_f - self.mean_ang * self.mean_ang) * bessel
      var_mom = (sum2_mom / reps_f - self.mean_mom * self.mean_mom) * bessel

      eps_cal = float(EpsForDtype(dtype, True))
      self.std_rep = math.sqrt(max(var_rep, eps_cal))
      self.std_rad = math.sqrt(max(var_rad, eps_cal))
      self.std_ang = math.sqrt(max(var_ang, eps_cal))
      self.std_mom = math.sqrt(max(var_mom, eps_cal))

      # Std of the weighted total (for final normalisation)
      sum_total, sum2_total = 0., 0.
      for i in range(int(reps)):
         t_rep = (all_rep[i] - self.mean_rep) / self.std_rep
         t_rad = self.lambda_rad * (all_rad[i] - self.mean_rad) / self.std_rad
         t_ang = self.lambda_ang * (all_ang[i] - self.mean_ang) / self.std_ang
         t_mom = self.lambda_mom * (all_mom[i] - self.mean_mom) / self.std_mom
         total = t_rep + t_rad + t_ang + t_mom
         sum_total += total
         sum2_total += total * total

      mean_total = sum_total / reps_f
      var_total = (sum2_total / reps_f - mean_total * mean_total) * bessel
      self.std_total = math.sqrt(max(var_total, eps_cal))

   # ---- core computation ----

   def _Compute(self, x: torch.Tensor) -> S_LossComponents:
      # x: (..., N, D) where N is #samples, D is feature dim
      if x.ndim < 2:
         raise ValueError(f"Expected x.ndim>=2 with shape (..., N, D), got {tuple(x.shape)}")

      n = int(x.shape[-2])
      d = int(x.shape[-1])
      batch_shape = x.shape[:-2]

      if n < 2 or d < 1:
         z = x.sum(dim=(-2, -1)) * 0.
         return S_LossComponents(z, z, z, z, z)

      wdtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
      xw = x.to(wdtype)
      n_f = float(n)

      mom_pen = self._MomentPenalty(xw)
      u, t = self._WristbandMap(xw)

      rad_loss = xw.new_zeros(batch_shape)
      if self.lambda_rad != 0.:
         rad_loss = self._RadialLoss(t, n_f, wdtype)

      if self.spectral:
         rep_loss = self._SpectralRepulsion(u, t)
         ang_loss = xw.new_zeros(batch_shape)
      else:
         e_ang = self._AngularExponent(u)
         ang_loss = xw.new_zeros(batch_shape)
         if self.lambda_ang != 0.:
            ang_loss = self._AngularUniformity(e_ang, n_f)
         rep_loss = self._PairwiseRepulsion(e_ang, t, n_f)

      return S_LossComponents(rep_loss, rep_loss, rad_loss, ang_loss, mom_pen) # dummy 'total' - faster this way, avoids nested structure in __call__

   # ---- public interface ----

   def __call__(self, x: torch.Tensor) -> S_LossComponents:
      """Compute the calibrated wristband-Gaussian loss.

      Parameters
      ----------
      x : Tensor of shape ``(..., N, D)``
          Batch of samples (``N`` samples of dimension ``D``).

      Returns
      -------
      S_LossComponents
          Named tuple ``(total, rep, rad, ang, mom)`` where ``total`` is the
          scalar to back-propagate and the rest are normalised diagnostics.
      """
      comp = self._Compute(x)

      # Normalize per-group (days), then reduce by mean over all leading dims.
      norm_rep = (comp.rep - self.mean_rep) / self.std_rep
      norm_rad = (comp.rad - self.mean_rad) / self.std_rad
      norm_ang = (comp.ang - self.mean_ang) / self.std_ang
      norm_mom = (comp.mom - self.mean_mom) / self.std_mom

      # Weighted sum of normalized components, divided by total std
      total = (norm_rep + self.lambda_rad * norm_rad + self.lambda_ang * norm_ang + self.lambda_mom * norm_mom) / self.std_total

      return S_LossComponents(total.mean(), norm_rep.mean(), norm_rad.mean(), norm_ang.mean(), norm_mom.mean())