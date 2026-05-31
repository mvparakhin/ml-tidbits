"""
Provides:
  - C_RankingLoss : configurable pairwise ranking loss (RankNet surrogate with optional LambdaRank/DCG weighting)
"""

from __future__ import annotations

import math

import torch

__all__ = ["C_RankingLoss"]

########################################################################################################################
# Helpers
########################################################################################################################

def EpsForDtype(dtype: torch.dtype, large: bool = False) -> float:
   """Return a small epsilon suitable for *dtype* (sqrt(eps) when *large*)."""
   eps = torch.finfo(dtype).eps
   return math.sqrt(eps) if large else eps


########################################################################################################################
# Ranking loss
########################################################################################################################

class C_RankingLoss:
   """Configurable pairwise ranking loss: a RankNet sigmoid surrogate with an optional detached LambdaRank/DCG weight.

   ``scores`` and ``labels`` are ``[..., N]``: the last axis is one group to rank, any leading axes are independent
   groups. For each pair ``(i, j)`` in a group with ``labels_i > labels_j`` the model is pushed toward
   ``scores_i > scores_j``; the pair is weighted by

       [gain != "none": |gain_i - gain_j|] * [discount: |disc_i - disc_j|].

   The value is ``2 * (weighted fraction of inverted pairs)``, which equals ``1 - (weighted Kendall tau)``:
   0 for a perfect ordering, 1 for random, 2 for a fully reversed ordering -- averaged over the groups that have
   at least one orderable pair.

   Parameters
   ----------
   gain : "none" | "label" | "position"
       Pair gain. ``"none"`` weights every orderable pair equally; ``"label"`` uses ``gain_base ** labels_i``
       (classic DCG gain, e.g. ``2 ** relevance``); ``"position"`` uses ``gain_base ** (normalized ideal rank in
       [0, 1])`` -- only the label *order* matters, not its scale.
   discount : bool
       If True, multiply each pair weight by the standard DCG positional-discount gap
       ``|1/log2(model_rank_i + 2) - 1/log2(model_rank_j + 2)|``, recomputed from the current scores.
   gain_base : float
       Base of the gain exponential (``> 1``). 2 reproduces the textbook ``2 ** relevance`` DCG gain.
   rank_scale : float
       Steepness of the surrogate ``sigmoid(-rank_scale * (scores_i - scores_j))`` for the 0/1 inversion.

   Example
   -------
   >>> loss = C_RankingLoss(gain="label", discount=True)
   >>> scores = torch.randn(32, 10, requires_grad=True)   # 32 queries, 10 items each
   >>> labels = torch.randint(0, 5, (32, 10)).float()     # graded relevance
   >>> loss(scores, labels).backward()
   """

   def __init__(self, *, gain: str = "label", discount: bool = True, gain_base: float = 2., rank_scale: float = 2.):
      if gain not in ("none", "label", "position"):
         raise ValueError("gain must be 'none', 'label', or 'position'")
      if gain != "none" and gain_base <= 1.:
         raise ValueError("gain_base must be greater than 1")
      if rank_scale <= 0.:
         raise ValueError("rank_scale must be positive")
      self.gain, self.discount = gain, bool(discount)
      self.gain_base, self.rank_scale = float(gain_base), float(rank_scale)

   def _Rank(self, x: torch.Tensor, ordinal: bool) -> torch.Tensor:
      """Descending rank along the last axis (0 = largest).

      ``ordinal=True`` breaks ties by index (distinct ranks even for equal values, matching argsort) for the score
      discount; ``ordinal=False`` averages tied positions (shared rank) for label-position gain.
      """
      gt = (x.unsqueeze(-2) > x.unsqueeze(-1)).sum(-1).to(x.dtype)
      eq = (x.unsqueeze(-2) == x.unsqueeze(-1)).to(x.dtype)
      return gt + (eq.tril(-1).sum(-1) if ordinal else 0.5 * (eq.sum(-1) - 1.))

   def _Surrogate(self, margin: torch.Tensor, sign: torch.Tensor, hard: bool) -> torch.Tensor:
      """Pairwise surrogate of the signed margin: smooth sigmoid (soft) or 0/1 step (hard); a tied orientation is 0.5."""
      eff = margin * sign
      return torch.heaviside(-eff, eff.new_full((), 0.5)) if hard else torch.sigmoid(-self.rank_scale * eff)

   def __call__(self, scores: torch.Tensor, labels: torch.Tensor, hard: bool = False, reduce: bool = True) -> torch.Tensor:
      """Return ``1 - (gain/discount-weighted Kendall tau)``; ``hard=True`` counts 0/1 inversions instead of the surrogate.

      Each unordered pair is taken once (upper triangle) and oriented by ``sign(label_i - label_j)``; a tied-label pair
      is a neutral ``0.5`` that still counts for ``gain="none"`` (normalized by all pairs) but gets zero weight otherwise.
      ``reduce=True`` (default) returns the scalar mean over groups; ``reduce=False`` returns the per-group value
      (shape = the leading axes), one loss per independent group.
      """
      margin = scores.unsqueeze(-1) - scores.unsqueeze(-2)
      with torch.no_grad():
         labels = labels.to(scores.dtype)                                      # keep the detached weight math in scores.dtype
         sign = (labels.unsqueeze(-1) - labels.unsqueeze(-2)).sign()           # orientation; 0 for tied labels
         if self.gain == "none":
            w = torch.ones_like(margin)
         else:
            if self.gain == "label":
               g = self.gain_base ** labels
            else:
               last = float(labels.shape[-1] - 1)
               g = self.gain_base ** ((last - self._Rank(labels, ordinal=False)) / max(last, 1.))
            w = (g.unsqueeze(-1) - g.unsqueeze(-2)).abs()                      # tied labels -> zero weight
         if self.discount:
            disc = 1. / torch.log2(self._Rank(scores, ordinal=True) + 2.)
            w = w * (disc.unsqueeze(-1) - disc.unsqueeze(-2)).abs()
         w = w.triu(1)                                                         # count each unordered pair once
         den = w.sum(dim=(-2, -1))
      valid = (den > 0).to(scores.dtype)
      per_group = valid * 2. * (self._Surrogate(margin, sign, hard) * w).sum(dim=(-2, -1)) / den.clamp_min(EpsForDtype(scores.dtype))
      if not reduce:
         return per_group
      return per_group.sum() / valid.sum().clamp_min(1.)

   def Contrast(self, a: torch.Tensor, b: torch.Tensor, hard: bool = False, reduce: bool = True) -> torch.Tensor:
      """Symmetric rank-agreement contrast between two prediction vectors ``a`` and ``b`` (both differentiable).

      The mean of ranking ``a`` by ``b``'s order and ``b`` by ``a``'s order, sharing the difference matrices. Uses
      ``rank_scale`` only (no gain/discount); a tied pair is a neutral ``0.5``. ``reduce``/``hard`` behave as in ``__call__``.
      """
      ma = a.unsqueeze(-1) - a.unsqueeze(-2)
      mb = b.unsqueeze(-1) - b.unsqueeze(-2)
      with torch.no_grad():
         sa, sb = ma.sign(), mb.sign()
      n = ma.shape[-1]
      pair = (self._Surrogate(ma, sb, hard) + self._Surrogate(mb, sa, hard)).triu(1)
      per_group = pair.sum(dim=(-2, -1)) / max(0.5 * float(n * (n - 1)), 1.)
      return per_group if not reduce else per_group.mean()
