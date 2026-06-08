"""Tests for C_CosineAnnealingWarmRestartsDecay (python/schedulers/Schedulers.py)."""

import math
import torch
import os, sys
# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schedulers.Schedulers import C_CosineAnnealingWarmRestartsDecay


def _MakeScheduler(base_lr=1., **kwargs):
   opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=base_lr)
   return opt, C_CosineAnnealingWarmRestartsDecay(opt, **kwargs)


def TestWarmupRamp():
   """Warmup ramps start_factor -> 1 over warmup_steps; the cosine schedule then starts at its peak."""
   print("TestWarmupRamp")
   opt, sch = _MakeScheduler(t_0=20, t_mult=1.5, decay=0., warmup_steps=10, warmup_start_factor=0.1)
   assert abs(opt.param_groups[0]['lr'] - 0.1) < 1.e-9, "warmup must start at start_factor * base_lr"
   for _ in range(9):
      sch.step()
   assert abs(opt.param_groups[0]['lr'] - 1.) < 1.e-9, "warmup must end at base_lr"
   sch.step()                                                       # first cosine step = peak
   assert abs(opt.param_groups[0]['lr'] - 1.) < 1.e-9, "cosine schedule must start at its peak"
   print("   ramp endpoints and post-warmup peak  OK")
   return dict(ok=True)


def TestExactIntegerBoundaryRestart():
   """Regression: t_0=25, t_mult=1.2 puts the cycle-2 boundary at exactly t=91, but the closed-form
   float lands an epsilon below it. The restart (and its peak LR) must not be skipped, and
   steps_to_cycle_end must never go negative."""
   print("TestExactIntegerBoundaryRestart")
   opt, sch = _MakeScheduler(t_0=25, t_mult=1.2, decay=0.1)
   restart_lr = None
   for t in range(1, 200):
      sch.step()
      assert sch.steps_to_cycle_end >= 0, f"steps_to_cycle_end went negative at t={t}"
      if t == 91:
         assert sch.just_restarted, f"restart must fire at t=91 (cycle={sch.cycle}, t_cur={sch.t_cur})"
         restart_lr = opt.param_groups[0]['lr']
   expected_peak = 0.9 ** 3
   assert abs(restart_lr - expected_peak) < 1.e-9, f"restart peak LR {restart_lr} != {expected_peak}"
   print(f"   restart at t=91 with peak lr={restart_lr:.6f}  OK")
   return dict(restart_lr=restart_lr)


def TestCycleEndWithFloatNoise():
   """Regression: t_0=100, t_mult=1.3 puts the cycle-2 boundary at exactly t=399, but start+t_i
   computes an epsilon above it. is_cycle_end must still fire exactly once per cycle (at t=398)."""
   print("TestCycleEndWithFloatNoise")
   opt, sch = _MakeScheduler(t_0=100, t_mult=1.3, decay=0.1)
   cycle_end_steps = []
   for t in range(1, 500):
      sch.step()
      if sch.is_cycle_end:
         cycle_end_steps.append(t)
   assert cycle_end_steps == [99, 229, 398], f"unexpected cycle-end steps: {cycle_end_steps}"
   print(f"   is_cycle_end fired at {cycle_end_steps}  OK")
   return dict(cycle_end_steps=cycle_end_steps)


def TestInvariantScan(*, steps: int = 3000):
   """Across fractional/integer multipliers (with warmup): counters stay consistent, every restart
   lands within one step of its boundary (regression: float noise used to delay it), and the LR at
   each restart matches the cosine formula with the decayed peak base_lr * (1-decay)^cycle."""
   print("TestInvariantScan")
   configs = [(25, 1.2), (100, 1.3), (200, 1.5), (7, 2.0), (10, 1.0), (50, 1.7)]
   for t0, tm in configs:
      opt, sch = _MakeScheduler(t_0=t0, t_mult=tm, decay=0.05, warmup_steps=10, warmup_start_factor=0.1)
      ends = restarts = 0
      for t in range(1, steps):
         sch.step()
         assert sch.steps_to_cycle_end >= 0, f"negative steps_to_cycle_end at t={t} (t_0={t0}, t_mult={tm})"
         ends += int(sch.is_cycle_end)
         restarts += int(sch.just_restarted)
         if sch.just_restarted:
            # Fractional boundaries put the first step of a cycle up to one step past the peak.
            assert sch.t_cur < 1., f"restart delayed past its boundary: t_cur={sch.t_cur} (t_0={t0}, t_mult={tm}, t={t})"
            expected = (0.95 ** sch.cycle) * 0.5 * (1. + math.cos(math.pi * sch.t_cur / sch.t_i))
            lr = opt.param_groups[0]['lr']
            assert abs(lr - expected) < 1.e-9, f"restart lr {lr} != {expected} (t_0={t0}, t_mult={tm}, t={t})"
      assert restarts > 0, f"no restart observed (t_0={t0}, t_mult={tm})"
      assert abs(ends - restarts) <= 1, f"cycle ends ({ends}) and restarts ({restarts}) diverged (t_0={t0}, t_mult={tm})"
   print(f"   {len(configs)} configs x {steps} steps: counters consistent, restarts on time, decayed peaks exact  OK")
   return dict(configs=len(configs))


def RunAll():
   results = {}
   results["warmup"] = TestWarmupRamp()
   print()
   results["boundary_restart"] = TestExactIntegerBoundaryRestart()
   print()
   results["cycle_end"] = TestCycleEndWithFloatNoise()
   print()
   results["invariants"] = TestInvariantScan()
   print("\nALL TESTS PASSED")
   return results


if __name__ == "__main__":
   RunAll()
