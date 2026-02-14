import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from typing import List, Tuple, Optional, Union
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
   # Class-level type annotations for TorchScript compatibility
   direction_multiplier: float
   x_pos: Optional[torch.Tensor]
   x_neg: Optional[torch.Tensor]
   y_pos: Optional[torch.Tensor]
   y_neg: Optional[torch.Tensor]
   ln_d: Optional[torch.Tensor]
   x_0: Optional[torch.Tensor]
   y_0: Optional[torch.Tensor]
   
   def __init__(self, n_of_nodes: Optional[int] = None, inverse: bool = False, direction: Union[str, int] = 'increasing', centered: bool = True):
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
      
      # Initialize all potential attributes for TorchScript compatibility
      self.x_pos = None
      self.x_neg = None
      self.y_pos = None
      self.y_neg = None
      self.ln_d = None
      self.x_0 = None
      self.y_0 = None

      if self.use_internal_params:
         if n_of_nodes is None:
            # Should be unreachable, but helps TorchScript narrow the type
            raise ValueError("Internal error: n_of_nodes is None despite use_internal_params being True.")
         if n_of_nodes < 1:
            raise ValueError("n_of_nodes must be >= 1")
         self._InitInternalParams(n_of_nodes)
   
   #////////////////////////////////////////////////////////////////////////////////////

   def _InitInternalParams(self, n: int):
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
   def _PrepareParams(self, input_data: torch.Tensor, spline_weights: Optional[torch.Tensor]) -> List[torch.Tensor]:
      """Helper to handle parameter sourcing for both forward and Deriv."""
      # Parameter sourcing
      if self.use_internal_params:
         if spline_weights is not None:
            raise ValueError("Spline initialized with internal nodes; do not provide external weights")

         # Unwrap optional tensors to get non-optional tensors
         x_pos_unwrapped = torch.jit._unwrap_optional(self.x_pos)
         x_neg_unwrapped = torch.jit._unwrap_optional(self.x_neg)
         y_pos_unwrapped = torch.jit._unwrap_optional(self.y_pos)
         y_neg_unwrapped = torch.jit._unwrap_optional(self.y_neg)
         ln_d_unwrapped = torch.jit._unwrap_optional(self.ln_d)

         params: List[torch.Tensor] = [x_pos_unwrapped, x_neg_unwrapped, y_pos_unwrapped, y_neg_unwrapped, ln_d_unwrapped]
         
         if not self.centered:
            x_0_tensor = torch.jit._unwrap_optional(self.x_0)
            y_0_tensor = torch.jit._unwrap_optional(self.y_0)
            params.extend([x_0_tensor, y_0_tensor])
         return params
      else:
         if spline_weights is None:
            raise ValueError("Spline initialized for external weights; must provide spline_weights")
         
         # Scriptable shape handling for Mode 2
         if input_data.ndim < 1:
            raise ValueError("Input data must be at least 1-dimensional.")

         # Calculate flattened batch dimension of input (B_flat)
         b_flat = input_data.numel() // input_data.shape[-1]
         
         w = spline_weights.shape[-1]

         if spline_weights.ndim < 1:
            raise ValueError("Spline weights must be at least 1-dimensional.")

         # Calculate flattened batch dimension of weights
         b_flat_weights = spline_weights.numel() // w

         # Shape validation and reshaping
         leading_input_sizes = input_data.shape[:-1]
         leading_weight_sizes = spline_weights.shape[:-1]

         if leading_weight_sizes == leading_input_sizes or b_flat_weights == b_flat:
            weights_reshaped = spline_weights.reshape(b_flat, w)
         else:
            raise ValueError('External weights shape incompatible with input leading dimensions.')
         
         # Validate W (8N+1 or 8N+3 if non-centered)
         w_offset = 3 if not self.centered else 1
         if w % 2 != 1 or (w - w_offset) < 8 or (w - w_offset) % 8 != 0:
            raise ValueError('Weights dimensionality validation failed (must be 8*n+1 or 8*n+3).')
         
         params = self._ParseExternalWeights(weights_reshaped)
         return params

   #////////////////////////////////////////////////////////////////////////////////////
   # forward always returns a Tuple for scriptability.
   #////////////////////////////////////////////////////////////////////////////////////
   def forward(self, input_data: torch.Tensor, spline_weights: Optional[torch.Tensor] = None, return_deriv: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
      if input_data.numel() == 0:
         # Return (empty_value, empty_derivative)
         return input_data, torch.empty_like(input_data)
      
      params = self._PrepareParams(input_data, spline_weights)
      
      # Call the internal computation method. Always request value; optionally request derivative.
      return self._ComputeSpline(input_data, params, calc_value=True, calc_deriv=return_deriv)

   #////////////////////////////////////////////////////////////////////////////////////
   # Derivative calculation
   #////////////////////////////////////////////////////////////////////////////////////
   def Deriv(self, input_data: torch.Tensor, spline_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
      """Calculates the derivative of the spline (or inverse spline) wrt the input."""
      if input_data.numel() == 0:
         return input_data
      
      params = self._PrepareParams(input_data, spline_weights)
      
      # Call the internal computation method, requesting only the derivative.
      _, deriv = self._ComputeSpline(input_data, params, calc_value=False, calc_deriv=True)
      return deriv
   
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
   def _ComputeSpline(self, input_data: torch.Tensor, params: List[torch.Tensor], calc_value: bool, calc_deriv: bool) -> Tuple[torch.Tensor, torch.Tensor]:
      """Calculate knots and apply spline transformation (or derivative)"""
      # Calculate knots (Shared computation)
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
      
      # Call the scriptable application logic
      val_out, deriv_out = self._ApplySpline(v_in, x_app, y_app, w_app, derivs_app, calc_value, calc_deriv)
      
      # Handle reshaping and empty tensor returns for scriptability consistency
      if calc_value:
         val_reshaped = val_out.reshape(orig_shape)
      else:
         val_reshaped = torch.zeros(0, dtype=input_data.dtype, device=input_data.device)

      if calc_deriv:
         deriv_reshaped = deriv_out.reshape(orig_shape)
      else:
         deriv_reshaped = torch.zeros(0, dtype=input_data.dtype, device=input_data.device)
         
      return val_reshaped, deriv_reshaped
   
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
      eps_train = 1e-6
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
      # Optimization: Use identity w*d = 1/w.
      # Original: num_w_m = lambdas * w_prev * d_prev + o_lambdas * w_next * d_next
      # Optimized: num_w_m = lambdas / w_prev + o_lambdas / w_next
      num_w_m = lambdas / w_prev + o_lambdas / w_next
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
      
      # Assemble y coordinates using indexed assignment for efficiency
      y_m_neg, y_m_pos = y_m[..., :n], y_m[..., n:]

      y = torch.empty((b, 4*n + 1), device=device, dtype=dtype)
      # Center point Y_0 (Index 2N). Initialize to 0 (offset applied later if non-centered).
      y[..., 2*n] = 0.
      
      # Negative side (Indices 0 to 2N-1)
      y[..., 0:2*n:2] = y_neg     # Base knots
      y[..., 1:2*n:2] = y_m_neg   # Mid knots
      
      # Positive side (Indices 2N+1 to 4N)
      y[..., 2*n+1::2] = y_m_pos  # Mid knots
      y[..., 2*n+2::2] = y_pos    # Base knots
      
      # Apply centering if needed
      if not self.centered:
         x_0, y_0 = params_batched[5], params_batched[6]
         x = x + x_0
         y = y + y_0
      
      return x, y, w, derivs
   
   #////////////////////////////////////////////////////////////////////////////////////
   # After getting the knots, computing the spline itself (Scriptable Core)
   #////////////////////////////////////////////////////////////////////////////////////
   def _ApplySpline(self, v_in: torch.Tensor, x: torch.Tensor, y: torch.Tensor, w: torch.Tensor, derivs: torch.Tensor, calc_value: bool, calc_deriv: bool) -> Tuple[torch.Tensor, torch.Tensor]:
      """Apply spline transformation and/or calculate derivative in a scriptable manner."""
      is_batched = (x.dim() > 1)
      eps_eval = 1e-6
      
      # Input transformation (Handle direction for forward pass input): g_out(x) = g_spline(s*x) where s is direction_multiplier
      if not self.inverse and self.direction_multiplier < 0:
         v_in = -v_in
      
      # Find bins using binary search (Shared computation)
      search_target = y if self.inverse else x
      indices = torch.searchsorted(search_target, v_in)
      indices = torch.clamp(indices, min=1, max=search_target.shape[-1]-1).long()
      
      # Gather knot values (Shared computation)
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
      
      # Calculate interpolation weights (v1, v2) and Denominator (S or S') (Shared computation)
      if self.inverse:
         # Inverse: S' = W_{k+1}(Y_{k+1}-y) + W_k(y-Y_k)
         v1 = w_k_plus_1 * (y_k_plus_1 - v_in)
         v2 = w_k * (v_in - y_k)
      else:
         # Forward: S = W_k(X_{k+1}-x) + W_{k+1}(x-X_k)
         v1 = w_k * (x_k_plus_1 - v_in)
         v2 = w_k_plus_1 * (v_in - x_k)

      denominator = torch.clip(v1 + v2, min=eps_eval)

      # Optimization: Reorganize interpolation using FMA structure (A + w*B).
      # Let beta = v2 / denominator. res = Y_k + beta*(Y_{k+1}-Y_k).
      beta = v2 / denominator

      # Initialize interpolation result variables as Tensors (required for TorchScript).
      res_val_interp = torch.zeros_like(denominator)
      res_deriv_interp = torch.zeros_like(denominator)

      # Calculate deltas (dY, dX) if needed. Initialize for TorchScript type stability.
      delta_x = torch.zeros_like(denominator)
      delta_y = torch.zeros_like(denominator)
      
      if calc_value or calc_deriv:
         delta_x = x_k_plus_1 - x_k
         delta_y = y_k_plus_1 - y_k

      # Calculate interpolation results conditionally.
      if calc_value:
         # Value calculation using FMA structure
         if self.inverse:
            # res = X_k + beta * dX
            res_val_interp = x_k + beta * delta_x
         else:
            # res = Y_k + beta * dY
            res_val_interp = y_k + beta * delta_y

      if calc_deriv:
         # Derivative calculation: g'(v) = (W_k * W_{k+1} * dY * dX) / S^2
         numerator = w_k * w_k_plus_1 * delta_y * delta_x
         res_deriv_interp = numerator / (denominator * denominator)
      
      # Handle extrapolation (Tails) (Shared setup)
      # Use slicing [0:1] and [-1:] to maintain dimensionality, which is safer for TorchScript broadcasting.
      if is_batched:
         x_left = x[..., 0:1]; x_right = x[..., -1:]
         y_left = y[..., 0:1]; y_right = y[..., -1:]
         d_left = derivs[..., 0:1]; d_right = derivs[..., -1:]
      else:
         x_left = x[0:1]; x_right = x[-1:]
         y_left = y[0:1]; y_right = y[-1:]
         d_left = derivs[0:1]; d_right = derivs[-1:]
      
      # Initialize extrapolation variables as Tensors.
      res_val_from_left = torch.zeros_like(v_in)
      res_val_from_right = torch.zeros_like(v_in)
      res_deriv_from_left = torch.zeros_like(v_in)
      res_deriv_from_right = torch.zeros_like(v_in)
      
      # Linear extrapolation beyond boundaries
      if self.inverse:
         d_left_inv = 1.0 / torch.clip(d_left, min=eps_eval)
         d_right_inv = 1.0 / torch.clip(d_right, min=eps_eval)
         if calc_deriv:
            res_deriv_from_left = d_left_inv.expand_as(v_in)
            res_deriv_from_right = d_right_inv.expand_as(v_in)
         if calc_value:
            res_val_from_left = x_left + (v_in - y_left) * d_left_inv
            res_val_from_right = x_right + (v_in - y_right) * d_right_inv
         mask_target_min = y_left
         mask_target_max = y_right
      else:
         if calc_deriv:
            res_deriv_from_left = d_left.expand_as(v_in)
            res_deriv_from_right = d_right.expand_as(v_in)
         if calc_value:
            res_val_from_left = y_left + (v_in - x_left) * d_left
            res_val_from_right = y_right + (v_in - x_right) * d_right
         mask_target_min = x_left
         mask_target_max = x_right
      
      # Combine interpolation and extrapolation
      res_val = torch.zeros_like(v_in)
      res_deriv = torch.zeros_like(v_in)

      mask_inside_min = v_in > mask_target_min
      
      if calc_value:
         res_val = torch.where(mask_inside_min, res_val_interp, res_val_from_left)
      if calc_deriv:
         res_deriv = torch.where(mask_inside_min, res_deriv_interp, res_deriv_from_left)
         
      mask_inside_max = mask_target_max > v_in
      if calc_value:
         res_val = torch.where(mask_inside_max, res_val, res_val_from_right)
      if calc_deriv:
         res_deriv = torch.where(mask_inside_max, res_deriv, res_deriv_from_right)
      
      # Finalization (Handle direction multiplier for output)
      if calc_deriv:
         # Chain rule: multiply by direction multiplier (s). F'(x) = G'(s*x)*s. H'(y) = s*(G^-1)'(y).
         if self.direction_multiplier < 0:
            res_deriv = -res_deriv

      if calc_value:
         # Value finalization (Inverse case: H(y) = s*G^-1(y))
         if self.inverse and self.direction_multiplier < 0:
            res_val = -res_val

      return res_val, res_deriv
   
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
      # Ensure parameters are 1D before concatenation
      x0_data = model.x_0.data.squeeze(0)
      y0_data = model.y_0.data.squeeze(0)
      if x0_data.ndim == 0: x0_data = x0_data.unsqueeze(0)
      if y0_data.ndim == 0: y0_data = y0_data.unsqueeze(0)

      params_list.append(x0_data)
      params_list.append(y0_data)
   
   return torch.cat(params_list, dim=-1)