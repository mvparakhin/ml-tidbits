import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from typing import List, Tuple, Optional
import numpy as np

##################################################################################################################################################
# Helper Functions
##################################################################################################################################################
def EpsForDtype(dtype: torch.dtype, tight: bool = True) -> float:
   """Returns suitable epsilon for numerical stability based on dtype"""
   if dtype in (torch.float16, torch.bfloat16):
      return 1e-6 if tight else 1e-4
   return 1e-12 if tight else 1e-6


##################################################################################################################################################
# UnifiedMonotonicSpline Implementation
##################################################################################################################################################
class UnifiedMonotonicSpline(nn.Module):
   """
   Unified Monotonic Rational Spline layer.
   
   Modes:
   1. Internal Weights (n_of_nodes provided): Single spline (optimized path)
   2. External Weights (n_of_nodes=None): Expects spline_weights in forward (batched path)

   Args:
      n_of_nodes (int, optional): Number of nodes (N).
      inverse (bool): If True, applies the inverse transformation.
      direction (str or int): 'increasing' (1) or 'decreasing' (-1).
      centered (bool): If True (default), spline passes through (0,0). If False, center (x_0, y_0) is learned.
   """
   def __init__(self, n_of_nodes=None, inverse=False, direction='increasing', centered=True):
      super(UnifiedMonotonicSpline, self).__init__()
      self.n_of_nodes_internal = n_of_nodes
      self.inverse = inverse
      self.use_internal_params = (n_of_nodes is not None)
      self.centered = centered
      
      # Validate and store direction
      if direction in ('increasing', 1):
         self.direction_multiplier = 1.
      elif direction in ('decreasing', -1):
         self.direction_multiplier = -1.
      else:
         raise ValueError("direction must be 'increasing' (1) or 'decreasing' (-1)")
      
      if self.use_internal_params:
         if n_of_nodes < 1:
            raise ValueError("n_of_nodes must be >= 1")
         self._InitInternalParams(n_of_nodes)
   
   #////////////////////////////////////////////////////////////////////////////////////

   def _InitInternalParams(self, n):
      """Initialize parameters with leading batch dimension (B=1)"""
      init_val = math.log(0.5)
      
      self.x_pos = nn.Parameter(torch.full((1, 2*n), init_val, dtype=torch.float32))
      self.x_neg = nn.Parameter(torch.full((1, 2*n), init_val, dtype=torch.float32))
      self.y_pos = nn.Parameter(torch.zeros(1, n, dtype=torch.float32))
      self.y_neg = nn.Parameter(torch.zeros(1, n, dtype=torch.float32))
      self.ln_d = nn.Parameter(torch.zeros(1, 2*n + 1, dtype=torch.float32))
      
      if not self.centered:
         self.x_0 = nn.Parameter(torch.zeros(1, 1, dtype=torch.float32))
         self.y_0 = nn.Parameter(torch.zeros(1, 1, dtype=torch.float32))
   
   #////////////////////////////////////////////////////////////////////////////////////
   # Common part for forward and Deriv
   #////////////////////////////////////////////////////////////////////////////////////
   def _PrepareParams(self, input_data, spline_weights):
      """Helper to handle parameter sourcing for both forward and Deriv."""
      # Parameter sourcing
      if self.use_internal_params:
         if spline_weights is not None:
            raise ValueError("Spline initialized with internal nodes; do not provide external weights")
         params = [self.x_pos, self.x_neg, self.y_pos, self.y_neg, self.ln_d]
         if not self.centered:
            params.extend([self.x_0, self.y_0])
      else:
         if spline_weights is None:
            raise ValueError("Spline initialized for external weights; must provide spline_weights")
         
         w = spline_weights.shape[-1]
         leading_input_shape = input_data.shape[:-1]
         b_flat = torch.prod(torch.tensor(leading_input_shape)).item() if input_data.ndim > 1 else 1
         
         # Shape validation and reshaping
         if spline_weights.shape[:-1] == leading_input_shape or (spline_weights.ndim >= 2 and torch.prod(torch.tensor(spline_weights.shape[:-1])).item() == b_flat):
            weights_reshaped = spline_weights.reshape(b_flat, w)
         else:
            raise ValueError(f'External weights shape {spline_weights.shape} incompatible with input leading dims {leading_input_shape}')
         
         # Validate W (8N+1 or 8N+3 if non-centered)
         w_offset = 3 if not self.centered else 1
         if w % 2 != 1 or (w - w_offset) < 8 or (w - w_offset) % 8 != 0:
            center_str = "8*n + 3" if not self.centered else "8*n + 1"
            raise ValueError(f'Weights dimensionality must be {center_str} (n>=1). Got: {w}')
         
         params = self._ParseExternalWeights(weights_reshaped)
      return params

   #////////////////////////////////////////////////////////////////////////////////////

   def forward(self, input_data, spline_weights=None):
      if input_data.numel() == 0:
         return input_data
      
      params = self._PrepareParams(input_data, spline_weights)
      return self._ComputeSpline(input_data, params, calc_deriv=False)

   #////////////////////////////////////////////////////////////////////////////////////
   # Derivative calculation
   #////////////////////////////////////////////////////////////////////////////////////
   def Deriv(self, input_data, spline_weights=None):
      """Calculates the derivative of the spline (or inverse spline) wrt the input."""
      if input_data.numel() == 0:
         return input_data
      
      params = self._PrepareParams(input_data, spline_weights)
      return self._ComputeSpline(input_data, params, calc_deriv=True)
   
   #////////////////////////////////////////////////////////////////////////////////////
   # In Mode 2 we need to unpack parameters being sent in
   #////////////////////////////////////////////////////////////////////////////////////
   def _ParseExternalWeights(self, weights: torch.Tensor) -> List[torch.Tensor]:
      """Parse external weights into spline parameters"""
      w = weights.shape[-1]
      w_offset = 3 if not self.centered else 1
      n = (w - w_offset) // 8
      
      # Align initialization scale with Mode 1
      log2 = torch.tensor(math.log(2.), dtype=weights.dtype, device=weights.device)
      
      x_pos = weights[..., 0:2*n] - log2
      x_neg = weights[..., 2*n:4*n] - log2
      y_pos = weights[..., 4*n:5*n]
      y_neg = weights[..., 5*n:6*n]
      ln_d = weights[..., 6*n:8*n+1]
      
      params = [x_pos, x_neg, y_pos, y_neg, ln_d]
      
      if not self.centered:
         x_0 = weights[..., 8*n+1:8*n+2]
         y_0 = weights[..., 8*n+2:8*n+3]
         params.extend([x_0, y_0])
      
      return params
   
   #////////////////////////////////////////////////////////////////////////////////////
   # Core Functional Logic
   #////////////////////////////////////////////////////////////////////////////////////
   def _ComputeSpline(self, input_data: torch.Tensor, params: List[torch.Tensor], calc_deriv: bool = False) -> torch.Tensor:
      """Calculate knots and apply spline transformation (or derivative)"""
      # Calculate knots
      x, y, w, derivs = self._CalculateKnots(params)
      
      # Prepare inputs for application
      orig_shape = input_data.shape
      b_proc = x.shape[0]  # Batch size of parameters (1 or B_flat)
      
      if b_proc == 1:
         # Mode 1: Single spline applied to many inputs
         v_in = input_data.reshape(-1)
         x_app, y_app, w_app, derivs_app = x.squeeze(0), y.squeeze(0), w.squeeze(0), derivs.squeeze(0)
      else:
         # Mode 2: B_flat splines for B_flat inputs
         v_in = input_data.reshape(b_proc, orig_shape[-1])
         x_app, y_app, w_app, derivs_app = x, y, w, derivs
      
      output = self._ApplySpline(v_in, x_app, y_app, w_app, derivs_app, calc_deriv)
      return output.reshape(orig_shape)
   
   #////////////////////////////////////////////////////////////////////////////////////
   # Knots are calculated every time forward() is called
   #////////////////////////////////////////////////////////////////////////////////////
   def _CalculateKnots(self, params_batched: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
      """
      Calculate spline knots (x, y, w) and derivatives.
      
      The spline is defined by base knots at positions x with weights w_base and derivatives d.
      Between base knots, we insert midpoints with weights w_m calculated to ensure monotonicity.
      """
      # Unpack parameters
      x_pos_raw, x_neg_raw, y_pos_raw, y_neg_raw, ln_d_raw = params_batched[0], params_batched[1], params_batched[2], params_batched[3], params_batched[4]
      
      b = x_pos_raw.shape[0]
      device = x_pos_raw.device
      dtype = x_pos_raw.dtype
      eps_train = EpsForDtype(dtype, False)
      n = y_pos_raw.shape[-1]
      
      # Phase 1: Calculate spacings, weights, and intermediate points
      
      # Exponentials of raw parameters give actual spacings/derivatives
      x_pos_exp = torch.exp(x_pos_raw)
      x_neg_exp = torch.exp(x_neg_raw)
      y_pos_exp = torch.exp(y_pos_raw)
      y_neg_exp = torch.exp(y_neg_raw)
      derivs = torch.exp(ln_d_raw)  # Derivatives at base knots
      
      # Base weights: w = 1/sqrt(d) for rational spline parameterization
      w_base = torch.rsqrt(derivs)  # Size 2N+1
      
      # Calculate lambdas: ratio controlling midpoint position between adjacent base knots
      # Each pair of x spacings defines an interval; lambda is first_spacing/(first+second)
      base_spacings_x_pos = x_pos_exp.view(b, n, 2).sum(dim=-1)
      base_spacings_x_neg = x_neg_exp.view(b, n, 2).sum(dim=-1)
      
      x_m_spacings_pos = x_pos_exp[..., ::2]  # First spacing in each pair
      x_m_spacings_neg = x_neg_exp[..., ::2]
      
      lambdas_pos = x_m_spacings_pos / base_spacings_x_pos.clamp(min=eps_train)
      lambdas_neg = x_m_spacings_neg / base_spacings_x_neg.clamp(min=eps_train)
      
      lambdas = torch.cat([lambdas_neg, lambdas_pos], dim=-1)
      o_lambdas = 1. - lambdas
      
      # Calculate intermediate weights w_m using rational spline formula
      # w_m ensures smooth interpolation between adjacent base knots
      w_prev, w_next = w_base[..., :-1], w_base[..., 1:]
      d_prev, d_next = derivs[..., :-1], derivs[..., 1:]
      
      base_spacings_x = torch.cat([base_spacings_x_neg, base_spacings_x_pos], dim=-1)
      base_spacings_y = torch.cat([y_neg_exp, y_pos_exp], dim=-1)
      
      # w_m = (lambda*w_p*d_p + (1-lambda)*w_n*d_n) * Delta_x / Delta_y
      num_w_m = lambdas * w_prev * d_prev + o_lambdas * w_next * d_next
      w_m = num_w_m * base_spacings_x / base_spacings_y.clamp(min=eps_train)
      
      # Calculate cumulative positions from spacings
      x_pos = torch.cumsum(x_pos_exp, dim=-1)
      x_neg = -torch.cumsum(x_neg_exp.flip(dims=[-1]).contiguous(), dim=-1).flip(dims=[-1])
      y_pos = torch.cumsum(y_pos_exp, dim=-1)
      y_neg = -torch.cumsum(y_neg_exp.flip(dims=[-1]).contiguous(), dim=-1).flip(dims=[-1])
      
      zero_tensor = torch.zeros((b, 1), device=device, dtype=dtype)
      x = torch.cat([x_neg, zero_tensor, x_pos], dim=-1)
      y_base = torch.cat([y_neg, zero_tensor, y_pos], dim=-1)
      
      # Calculate y_m: vertical position of midpoints using weighted average
      y_prev, y_next = y_base[..., :-1], y_base[..., 1:]
      num_y_m = o_lambdas * w_prev * y_prev + lambdas * w_next * y_next
      den_y_m = (o_lambdas * w_prev + lambdas * w_next).clamp(min=eps_train)
      y_m = num_y_m / den_y_m
      
      # Phase 2: Assemble final knot arrays
      
      # Interleave base and midpoint weights
      w = torch.empty((b, 4*n + 1), device=device, dtype=dtype)
      w[..., ::2] = w_base
      w[..., 1::2] = w_m
      
      # Assemble y coordinates (asymmetric interleaving for neg/pos sides)
      y_m_neg, y_m_pos = y_m[..., :n], y_m[..., n:]
      y_neg_i = torch.stack([y_neg, y_m_neg], dim=-1).view(b, 2*n)  # Base, Mid
      y_pos_i = torch.stack([y_m_pos, y_pos], dim=-1).view(b, 2*n)  # Mid, Base
      y = torch.cat([y_neg_i, zero_tensor, y_pos_i], dim=-1)
      
      # Apply centering if needed
      if not self.centered:
         x_0, y_0 = params_batched[5], params_batched[6]
         x = x + x_0
         y = y + y_0
      
      return x, y, w, derivs
   
   #////////////////////////////////////////////////////////////////////////////////////
   # After getting the knots, computing the spline itself
   #////////////////////////////////////////////////////////////////////////////////////
   def _ApplySpline(self, v_in: torch.Tensor, x: torch.Tensor, y: torch.Tensor, w: torch.Tensor, derivs: torch.Tensor, calc_deriv: bool = False) -> torch.Tensor:
      """Apply spline transformation or calculate derivative to input values"""
      is_batched = (x.dim() > 1)
      eps_eval = EpsForDtype(v_in.dtype, False)
      
      # Input transformation (Handle direction for forward pass input): g_out(x) = g_spline(s*x) where s is direction_multiplier
      if not self.inverse and self.direction_multiplier < 0:
         v_in = -v_in
      
      # Find bins using binary search
      search_target = y if self.inverse else x
      indices = torch.searchsorted(search_target, v_in)
      indices = torch.clamp(indices, min=1, max=search_target.shape[-1]-1).long()
      
      # Gather knot values
      if is_batched:
         w_k = torch.gather(w, dim=-1, index=indices - 1)
         w_k_plus_1 = torch.gather(w, dim=-1, index=indices)
         x_k = torch.gather(x, dim=-1, index=indices - 1)
         x_k_plus_1 = torch.gather(x, dim=-1, index=indices)
         y_k = torch.gather(y, dim=-1, index=indices - 1)
         y_k_plus_1 = torch.gather(y, dim=-1, index=indices)
      else:
         w_k = w[indices - 1]
         w_k_plus_1 = w[indices]
         x_k = x[indices - 1]
         x_k_plus_1 = x[indices]
         y_k = y[indices - 1]
         y_k_plus_1 = y[indices]
      
      # Calculate interpolation weights (v1, v2) and Denominator (S or S')
      if self.inverse:
         # Inverse: S' = W_{k+1}(Y_{k+1}-y) + W_k(y-Y_k)
         v1 = w_k_plus_1 * (y_k_plus_1 - v_in)
         v2 = w_k * (v_in - y_k)
      else:
         # Forward: S = W_k(X_{k+1}-x) + W_{k+1}(x-X_k)
         v1 = w_k * (x_k_plus_1 - v_in)
         v2 = w_k_plus_1 * (v_in - x_k)
      
      denominator = torch.clip(v1 + v2, min=eps_eval)

      if calc_deriv:
         # Derivative calculation (symmetric formula for increasing spline)
         # g'(v) = (W_k * W_{k+1} * dY * dX) / S^2
         numerator = w_k * w_k_plus_1 * (y_k_plus_1 - y_k) * (x_k_plus_1 - x_k)
         # We rely on the fact that X/Y are strictly increasing and W>0, so numerator >= 0.
         res = numerator / (denominator * denominator)
      else:
         # Value calculation
         if self.inverse:
            # res = (X_k*v1 + X_{k+1}*v2) / S'
            res = (x_k*v1 + x_k_plus_1*v2) / denominator
         else:
            # res = (Y_k*v1 + Y_{k+1}*v2) / S
            res = (y_k*v1 + y_k_plus_1*v2) / denominator
      
      # Handle extrapolation (Tails)
      if is_batched:
         x_left = x[..., 0:1]; x_right = x[..., -1:]
         y_left = y[..., 0:1]; y_right = y[..., -1:]
         d_left = derivs[..., 0:1]; d_right = derivs[..., -1:]
      else:
         x_left = x[0]; x_right = x[-1]
         y_left = y[0]; y_right = y[-1]
         d_left = derivs[0]; d_right = derivs[-1]
      
      # Linear extrapolation beyond boundaries
      if self.inverse:
         d_left_inv = 1.0 / torch.clip(d_left, min=eps_eval)
         d_right_inv = 1.0 / torch.clip(d_right, min=eps_eval)
         if calc_deriv:
            res_from_left = d_left_inv
            res_from_right = d_right_inv
         else:
            res_from_left = x_left + (v_in - y_left) * d_left_inv
            res_from_right = x_right + (v_in - y_right) * d_right_inv
         mask_target_min = y_left
         mask_target_max = y_right
      else:
         if calc_deriv:
            res_from_left = d_left
            res_from_right = d_right
         else:
            res_from_left = y_left + (v_in - x_left) * d_left
            res_from_right = y_right + (v_in - x_right) * d_right
         mask_target_min = x_left
         mask_target_max = x_right
      
      # Combine interpolation and extrapolation
      mask_inside_min = v_in > mask_target_min
      res = torch.where(mask_inside_min, res, res_from_left)
      mask_inside_max = mask_target_max > v_in
      res = torch.where(mask_inside_max, res, res_from_right)
      
      # Finalization (Handle direction multiplier for output)
      if calc_deriv:
         # Chain rule: multiply by direction multiplier (s). F'(x) = G'(s*x)*s. H'(y) = s*(G^-1)'(y).
         if self.direction_multiplier < 0:
            res = -res
      else:
         # Value finalization (Inverse case: H(y) = s*G^-1(y))
         if self.inverse and self.direction_multiplier < 0:
            res = -res
      return res
   
   #////////////////////////////////////////////////////////////////////////////////////
   # Exporting to text file (Mode 1 only)
   #////////////////////////////////////////////////////////////////////////////////////
   def SaveSplineWeights(self, file_name: str, append: bool = False) -> None:
      """Save parameters in legacy format"""
      if not self.use_internal_params:
         raise RuntimeError("SaveSplineWeights() requires internal-params mode")
      
      params_to_save = [self.x_pos, self.x_neg, self.y_pos, self.y_neg, self.ln_d]
      if not self.centered:
         xy0 = torch.stack([self.x_0.reshape(-1), self.y_0.reshape(-1)], dim=-1)
         params_to_save.append(xy0)
      
      with open(file_name, ("ab" if append else "wb")) as f:
         f.write(b"#VER = 1001\n")
         for t in params_to_save:
            x = t.detach().cpu().numpy().squeeze(0)
            if x.ndim == 0:
               x = np.array([x])
            np.savetxt(f, x.reshape(1, -1), delimiter="\t")

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# Parameter Extraction from a model in Mode 1 for use in Mode 2
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
def ExtractParamsForExternal(model):
   """Extract parameters from Mode 1 model for Mode 2 usage"""
   if not model.use_internal_params:
      raise ValueError("Model must be initialized in internal parameter mode")
   
   # Mode 2 internally subtracts log(2), so add it back here
   dtype = model.x_pos.data.dtype
   device = model.x_pos.data.device
   log2 = torch.tensor(math.log(2.), dtype=dtype, device=device)
   
   params_list = [
      model.x_pos.data.squeeze(0) + log2,
      model.x_neg.data.squeeze(0) + log2,
      model.y_pos.data.squeeze(0),
      model.y_neg.data.squeeze(0),
      model.ln_d.data.squeeze(0)
   ]
   
   if not model.centered:
      params_list.append(model.x_0.data.squeeze(0))
      params_list.append(model.y_0.data.squeeze(0))
   
   return torch.cat(params_list, dim=-1)
