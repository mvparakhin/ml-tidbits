"""Tests and a training example for C_RankingLoss (python/ranking/RankingLoss.py)."""

import torch
import os, sys
# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ranking.RankingLoss import C_RankingLoss


def TestTauBaselines(*, n: int = 6, n_groups: int = 20000, seed: int = 0):
   """The loss equals 1 - weighted Kendall tau: 0 for a perfect order, 2 for reversed, ~1 for random.

   Verified (hard metric) for every (gain, discount) combination. With the discount on, a random model
   sits slightly above 1 because the discount weight depends on the model's own ranking.
   """
   torch.manual_seed(seed)
   labels = torch.arange(n - 1, -1, -1, dtype=torch.float32)     # strictly decreasing -> a unique ideal order
   perfect = labels.clone()
   reversed_scores = -labels
   random_scores = torch.randn(n_groups, n)
   random_labels = labels.expand(n_groups, n)

   print("TestTauBaselines: 1 - weighted Kendall tau")
   print("   gain      discount   perfect   reversed    random")
   results = {}
   for gain in ("none", "label", "position"):
      for discount in (True, False):
         base = 32. if gain == "position" else 2.
         loss = C_RankingLoss(gain=gain, discount=discount, gain_base=base)
         p = float(loss(perfect, labels, hard=True))
         r = float(loss(reversed_scores, labels, hard=True))
         m = float(loss(random_scores, random_labels, hard=True))
         print(f"   {gain:8s}  {str(discount):5s}      {p:7.4f}   {r:7.4f}   {m:7.4f}")
         assert p < 1.e-6, f"perfect order should be 0, got {p}"
         assert abs(r - 2.) < 1.e-6, f"reversed order should be 2, got {r}"
         assert 0.9 < m < 1.25, f"random order should be near 1, got {m}"
         results[(gain, discount)] = (p, r, m)
   return results


def TestTieHandlingAndEdges():
   """Tied scores keep a gradient (ordinal discount); degenerate groups are excluded; fp16/float64 are safe."""
   print("TestTieHandlingAndEdges")

   # Tied scores must still produce signal: the ordinal score discount breaks the tie.
   scores = torch.zeros(3, requires_grad=True)
   labels = torch.tensor([3., 2., 1.])
   loss = C_RankingLoss(gain="label", discount=True)(scores, labels)
   loss.backward()
   soft = float(loss.detach())
   grad_l1 = float(scores.grad.abs().sum())
   print(f"   tied scores:   soft={soft:.4f} (expect 1.0)   grad_l1={grad_l1:.4f} (expect > 0)")
   assert abs(soft - 1.) < 1.e-6, f"tied scores should give 1.0, got {soft}"
   assert grad_l1 > 0., "tied scores must still produce a gradient"

   # An all-equal-label group has no orderable pair: it is excluded from the mean, not counted as a perfect 0.
   torch.manual_seed(1)
   two = torch.randn(2, 5)
   labels2 = torch.stack([torch.arange(4, -1, -1, dtype=torch.float32), torch.ones(5)])   # group 1 is degenerate
   f = C_RankingLoss(gain="label")
   both = float(f(two, labels2, hard=True))
   valid_only = float(f(two[:1], labels2[:1], hard=True))
   print(f"   degenerate:    both-groups={both:.6f}   valid-only={valid_only:.6f} (must match)")
   assert abs(both - valid_only) < 1.e-6, "degenerate group must be excluded, not averaged as 0"

   # float16: a degenerate input stays finite (eps floor avoids 0/0).
   fp16 = C_RankingLoss(gain="label", discount=True)(
      torch.zeros(5, dtype=torch.float16), torch.ones(5, dtype=torch.float16), hard=True)
   print(f"   fp16 all-equal labels: value={float(fp16)}   finite={bool(torch.isfinite(fp16))}")
   assert torch.isfinite(fp16), "fp16 degenerate group must not be NaN"

   # float64 labels must not promote the float32 loss.
   out = C_RankingLoss(gain="label", discount=True)(torch.randn(4, 5), torch.randint(0, 4, (4, 5)).double())
   print(f"   float64 labels -> loss dtype {out.dtype} (expect torch.float32)")
   assert out.dtype == torch.float32, f"float64 labels promoted the loss to {out.dtype}"
   return dict(tied_soft=soft, tied_grad_l1=grad_l1)


def TestExactValueAndShapes():
   """Hand-computed value and single-group vs batched consistency."""
   print("TestExactValueAndShapes")
   # labels [2,1,0], scores [3,1,2]: ordered pairs (0,1),(0,2) concordant, (1,2) inverted.
   # label gain 2**[2,1,0]=[4,2,1] -> weights |4-2|=2, |4-1|=3, |2-1|=1; only the weight-1 pair is wrong.
   # loss = 2 * (1*1) / (2+3+1) = 1/3.
   got = float(C_RankingLoss(gain="label", discount=False)(torch.tensor([3., 1., 2.]), torch.tensor([2., 1., 0.]), hard=True))
   print(f"   hand-computed: value={got:.6f}   expected={1. / 3.:.6f}")
   assert abs(got - 1. / 3.) < 1.e-6, f"expected 1/3, got {got}"

   torch.manual_seed(2)
   s = torch.randn(5)
   y = torch.tensor([3., 1., 2., 0., 2.])
   f = C_RankingLoss(gain="label", discount=True)
   single = float(f(s, y))
   batched = float(f(s.expand(4, 5), y.expand(4, 5)))
   print(f"   single [N]={single:.6f}   batched [B,N]={batched:.6f} (must match)")
   assert abs(single - batched) < 1.e-6, "single-group and batched results must agree"
   return dict(hand_value=got)


def TestTrainingConverges(*, n_groups: int = 8, n: int = 12, steps: int = 400, lr: float = 0.1, seed: int = 0):
   """Minimizing the loss sorts free scores into the label order: the hard 1 - tau metric falls to ~0."""
   torch.manual_seed(seed)
   labels = torch.randint(0, 5, (n_groups, n)).float()             # graded relevance, ties allowed
   scores = torch.nn.Parameter(0.01 * torch.randn(n_groups, n))    # start near-tied (also exercises the ordinal discount)
   loss_fn = C_RankingLoss(gain="label", discount=True)
   opt = torch.optim.Adam([scores], lr=lr)
   hard_metric = lambda: float(loss_fn(scores.detach(), labels, hard=True))

   start_soft = float(loss_fn(scores, labels).detach())
   start_hard = hard_metric()
   print(f"TestTrainingConverges: {n_groups} groups x {n} items, {steps} steps")
   print(f"   step    0:  soft={start_soft:.4f}   hard(1-tau)={start_hard:.4f}")
   for step in range(1, steps + 1):
      opt.zero_grad(set_to_none=True)
      loss = loss_fn(scores, labels)
      loss.backward()
      opt.step()
      if step % 100 == 0:
         print(f"   step {step:4d}:  soft={float(loss.detach()):.4f}   hard(1-tau)={hard_metric():.4f}")
   final_soft = float(loss_fn(scores, labels).detach())
   final_hard = hard_metric()
   assert final_soft < start_soft, f"soft loss did not decrease ({start_soft:.4f} -> {final_soft:.4f})"
   assert final_hard < 0.02, f"did not converge to a correct ranking, hard={final_hard:.4f}"
   return dict(start_hard=start_hard, final_hard=final_hard, start_soft=start_soft, final_soft=final_soft)


def RunAll():
   results = {}
   results["tau_baselines"] = TestTauBaselines()
   print()
   results["tie_edges"] = TestTieHandlingAndEdges()
   print()
   results["exact_shapes"] = TestExactValueAndShapes()
   print()
   results["training"] = TestTrainingConverges()
   print("\nALL TESTS PASSED")
   return results


if __name__ == "__main__":
   RunAll()
