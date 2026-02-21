import math, torch
from torch.optim.lr_scheduler import LRScheduler

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# Fractional t_mult supported via O(1) closed-form cycle lookup.
#
# Public introspection fields (updated after every step()):
#   cycle                 cycle index for the LR that is now set in optimizer (0,1,2,...)
#   prev_cycle            cycle index before the last step() call
#   t_i                   current cycle length (float, in scheduler steps)
#   t_cur                 time within current cycle (float)
#   next_restart_t        continuous time of next restart (float)
#   steps_to_cycle_end    integer steps remaining until the last step before restart (assuming integer stepping)
#   is_cycle_end          True if current LR index is the last step in this cycle (lowest LR before restart)
#   just_restarted        True if the last step() call moved into a new cycle
#
# Decay semantics: decay in [0,1), exponential peak decay per cycle, gamma = (1 - decay)
# init_cycle=2*batches_in_epoch, cycle_mult_factor=1.5, decay=0.02
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class C_CosineAnnealingWarmRestartsDecay(torch.optim.lr_scheduler.LRScheduler):
   def __init__(self, optimizer, t_0, t_mult=2, eta_min=0, decay=0.1, last_epoch=-1, *, warmup_steps: int = 0, warmup_start_factor: float = 0.1):
      self.decay = float(decay)
      if self.decay < 0. or self.decay >= 1.:
         raise ValueError(f"Expected decay in [0, 1), but got {decay}")
      self.gamma = 1. - self.decay

      self.t_0 = float(t_0)
      if self.t_0 <= 0.:
         raise ValueError(f"Expected positive t_0, but got {t_0}")

      self.t_mult = float(t_mult)
      if self.t_mult < 1.:
         raise ValueError(f"Expected t_mult >= 1, but got {t_mult}")

      self.eta_min = float(eta_min)

      # ---- Linear warmup: LR ramps (warmup_start_factor * base_lr) -> base_lr over warmup_steps,
      # then the cosine-restart schedule begins at its peak (schedule is shifted by warmup_steps).
      self.warmup_steps = int(warmup_steps)
      if self.warmup_steps < 0:
         raise ValueError(f"Expected warmup_steps >= 0, but got {warmup_steps}")
      self.warmup_start_factor = float(warmup_start_factor)
      if self.warmup_start_factor <= 0. or self.warmup_start_factor > 1.:
         raise ValueError(f"Expected warmup_start_factor in (0, 1], but got {warmup_start_factor}")

      # Introspection state (populated on the first base step() call in LRScheduler.__init__)
      self.cycle = 0
      self.prev_cycle = 0
      self.t_i = self.t_0
      self.t_cur = 0.
      self.next_restart_t = self.t_0
      self.steps_to_cycle_end = 0
      self.is_cycle_end = False
      self.just_restarted = False

      super(C_CosineAnnealingWarmRestartsDecay, self).__init__(optimizer, last_epoch)

   def _CycleParams(self, t):
      t = float(t)
      if t < 0.:
         return 0, self.t_0, 0., 0.

      if self.t_mult == 1.:
         cycle = int(math.floor(t / self.t_0))
         start_t = float(cycle) * self.t_0
         t_i = self.t_0
         t_cur = t - start_t
         return cycle, t_i, t_cur, start_t

      x = t * (self.t_mult - 1.) / self.t_0 + 1.
      if x < 1.:
         cycle = 0
      else:
         cycle = int(math.floor(math.log(x) / math.log(self.t_mult)))
         if cycle < 0:
            cycle = 0

      t_i = self.t_0 * (self.t_mult ** float(cycle))
      start_t = self.t_0 * ((self.t_mult ** float(cycle) - 1.) / (self.t_mult - 1.))
      t_cur = t - start_t

      # Numerical safety around boundaries
      if t_cur < 0.:
         t_cur = 0.
      if t_cur >= t_i:
         cycle += 1
         t_i = self.t_0 * (self.t_mult ** float(cycle))
         start_t = self.t_0 * ((self.t_mult ** float(cycle) - 1.) / (self.t_mult - 1.))
         t_cur = t - start_t
         if t_cur < 0.:
            t_cur = 0.

      return cycle, t_i, t_cur, start_t

   def get_lr(self):
      t = float(self.last_epoch)
      w = float(getattr(self, "warmup_steps", 0))
      cycle, t_i, t_cur, start_t = self._CycleParams(t - w) # Shift schedule by warmup so cosine starts at peak right after warmup ends.
      start_t = float(start_t + w)


      self.prev_cycle = int(getattr(self, "cycle", 0))
      self.cycle = int(cycle)
      self.just_restarted = (self.cycle != self.prev_cycle)

      self.t_i = float(t_i)
      self.t_cur = float(t_cur)
      self.next_restart_t = float(start_t + t_i)

      next_restart_step = int(math.ceil(self.next_restart_t))
      cycle_end_step = next_restart_step - 1
      cur_step = int(math.floor(t))
      self.steps_to_cycle_end = int(cycle_end_step - cur_step)
      self.is_cycle_end = (self.steps_to_cycle_end == 0)

      # ---- Warmup branch (no restarts during warmup; cosine schedule begins after warmup) ----
      if w > 0. and t < w:
         if self.warmup_steps <= 1:
            warm_scale = 1.
         else: # t in [0, w-1] => scale in [start_factor, 1]
            warm_scale = self.warmup_start_factor + (1. - self.warmup_start_factor) * (t / (w - 1.))
         warm_scale = min(1., max(self.warmup_start_factor, float(warm_scale)))
         lrs = []
         for base_lr in self.base_lrs:
            lr = float(base_lr) * warm_scale
            if lr < self.eta_min:
               lr = self.eta_min
            lrs.append(lr)
         return lrs

      scale = (self.gamma ** float(self.cycle)) if self.gamma != 1. else 1.
      c = math.cos(math.pi * (self.t_cur / self.t_i)) if self.t_i > 0. else 1.

      lrs = []
      for base_lr in self.base_lrs:
         hi = float(base_lr) * scale
         lo = self.eta_min
         if lo > hi:
            lo = hi
         lr = lo + .5 * (hi - lo) * (1. + c)
         lrs.append(lr)

      return lrs
