"""Unit tests for EmbedModels.py building blocks (attention, flows, wristband loss)."""

import torch
import os, sys
# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embed_models.EmbedModels import (
   C_EmbedAttentionModule,
   C_InvertibleFlow,
   C_PermutationLayer,
   C_WristbandGaussianLoss,
   W2ToStandardNormalSq,
)


def TestHeadCombine():
   """Regression: a user-supplied head_combine must be honored even when n_of_heads == 1."""
   print("TestHeadCombine")
   hc = torch.nn.Linear(16, 5)
   att = C_EmbedAttentionModule(8, 16, 5, n_of_basis=4, n_of_heads=1, head_combine=hc)
   assert att.head_combine is hc, "user head_combine ignored for n_of_heads=1"
   out = att(torch.randn(7, 8))
   assert out.shape == (7, 5), f"bad output shape {tuple(out.shape)}"

   # Defaults: Identity for one head, Linear for several
   att1 = C_EmbedAttentionModule(8, 16, 16, n_of_basis=4, n_of_heads=1)
   assert isinstance(att1.head_combine, torch.nn.Identity)
   assert att1(torch.randn(3, 8)).shape == (3, 16)
   att2 = C_EmbedAttentionModule(8, 16, 5, n_of_basis=4, n_of_heads=2)
   assert isinstance(att2.head_combine, torch.nn.Linear)
   assert att2(torch.randn(3, 8)).shape == (3, 5)
   print("   custom head_combine honored; defaults unchanged  OK")
   return dict(ok=True)


def TestAffineExpertInit():
   """Regression: v_in/v_out must not hold torch.empty garbage when affine_experts=False."""
   print("TestAffineExpertInit")
   att = C_EmbedAttentionModule(8, 16, 16, n_of_basis=4, n_of_heads=1)
   assert float(att.v_in.abs().sum()) == 0. and float(att.v_out.abs().sum()) == 0., "v_in/v_out not zeroed"
   all_params = torch.cat([p.detach().flatten() for p in att.parameters()])
   assert torch.isfinite(all_params).all(), "non-finite values among parameters"

   att_on = C_EmbedAttentionModule(8, 16, 16, n_of_basis=4, n_of_heads=1, affine_experts=True)
   assert float(att_on.v_in.abs().sum()) > 0., "affine_experts=True must initialize v_in"
   print("   v_in/v_out zeroed when disabled, initialized when enabled  OK")
   return dict(ok=True)


def TestFlowRoundTrip():
   """C_InvertibleFlow.inverse must exactly undo forward in every permute mode, odd dims included."""
   print("TestFlowRoundTrip")
   torch.manual_seed(0)
   for mode in ("per_pair", "per_layer", "none"):
      for dim in (8, 7):
         flow = C_InvertibleFlow(dim, n_layers=4, hidden_dim=32, permute_mode=mode)
         x = torch.randn(64, dim)
         x_rec = flow.inverse(flow(x))
         err = float((x_rec - x).abs().max())
         assert err < 1.e-5, f"round trip failed (mode={mode}, dim={dim}): {err}"
   perm = C_PermutationLayer(6, torch.randperm(6))
   x = torch.randn(10, 6)
   assert torch.equal(perm.inverse(perm(x)), x), "permutation inverse not exact"
   print("   flow and permutation round trips exact  OK")
   return dict(ok=True)


def TestW2Sanity():
   """W2ToStandardNormalSq: near 0 under the null, grows with a mean shift, and the Gram (d > B)
   branch accounts for the missing zero eigenvalues."""
   print("TestW2Sanity")
   torch.manual_seed(1)
   null = float(W2ToStandardNormalSq(torch.randn(4096, 8)))
   shifted = float(W2ToStandardNormalSq(torch.randn(4096, 8) + 2.))
   assert null < 0.1, f"null W2 too large: {null}"
   assert shifted > 8 * 2. ** 2 * 0.5, f"shifted W2 too small: {shifted}"

   w2_gram = float(W2ToStandardNormalSq(torch.randn(8, 32)))   # B < d: Gram branch
   assert w2_gram >= 32 - 8, f"Gram branch must count (d - B) zero eigenvalues: {w2_gram}"
   print(f"   null={null:.4f}, shifted={shifted:.2f}, gram-branch={w2_gram:.2f}  OK")
   return dict(null=null, shifted=shifted)


def TestWristbandCalibratedExample():
   """Regression: the documented usage pattern must back-propagate (input requires grad)."""
   print("TestWristbandCalibratedExample")
   loss_fn = C_WristbandGaussianLoss(calibration_shape=(64, 8), calibration_reps=32)
   z = torch.randn(64, 8, requires_grad=True)
   lc = loss_fn(z)
   lc.total.backward()
   assert z.grad is not None and torch.isfinite(z.grad).all(), "no finite gradient from calibrated loss"

   try:
      loss_fn(torch.randn(32, 8))
      raise AssertionError("calibrated loss must reject a mismatched batch shape")
   except ValueError:
      pass
   print("   calibrated example back-propagates; shape mismatch rejected  OK")
   return dict(ok=True)


def RunAll():
   results = {}
   results["head_combine"] = TestHeadCombine()
   print()
   results["affine_init"] = TestAffineExpertInit()
   print()
   results["flow_round_trip"] = TestFlowRoundTrip()
   print()
   results["w2"] = TestW2Sanity()
   print()
   results["wristband"] = TestWristbandCalibratedExample()
   print("\nALL TESTS PASSED")
   return results


if __name__ == "__main__":
   RunAll()
