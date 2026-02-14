import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from math import prod
import matplotlib.pyplot as plt
import time
import os, numpy as np
import sys

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mlrsplines.LRSplines import *

##################################################################################################################################################
# Test Harness
##################################################################################################################################################

def CubeTarget(x): return x**3
def NegCubeTarget(x): return -(x**3)
def TanhTarget(x): return torch.tanh(2*x)
def DecreasingTarget(x): return -torch.tanh(2*x)
def NonCenteredTarget(x): return (x-3.)**3 + 5.

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

def TrainSplineMode1(target_func, title, device, n_nodes=8, epochs=3000, lr=0.01, 
                     x_range=(-3, 3), inverse=False, direction='increasing', centered=True):
   print(f"\n--- Training Mode 1 (Inv={inverse}, Dir={direction}, Centered={centered}) for {title} ---")
   torch.manual_seed(42)
   
   # Generate training data
   n_samples = 256
   x_data = torch.linspace(x_range[0], x_range[1], n_samples).reshape(-1, 1).to(device)
   y_data = target_func(x_data).to(device)
   input_train, target_train = (y_data, x_data) if inverse else (x_data, y_data)
   
   # Setup model
   model = UnifiedMonotonicSpline(n_of_nodes=n_nodes, inverse=inverse, 
                                  direction=direction, centered=centered).to(device)
   optimizer = torch.optim.Adam(model.parameters(), lr=lr)
   criterion = nn.MSELoss()
   
   # Script the model for training
   scripted_model = torch.jit.script(model)
   
   # Training loop - use scripted model for training
   scripted_model.train()
   start_time = time.time()
   for epoch in range(epochs):
      optimizer.zero_grad()
      output_pred, _ = scripted_model(input_train)
      loss = criterion(output_pred, target_train)
      
      if torch.isnan(loss):
         print(f"NaN loss detected. Stopping.")
         return None
      
      loss.backward()
      optimizer.step()
      
      if (epoch + 1) % (epochs // 5) == 0 or epoch == epochs - 1:
         print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')
   
   print(f"Training time: {time.time() - start_time:.2f}s. Final Loss: {loss.item():.6f}")
   
   if not centered and model.use_internal_params:
      if model.x_0 is not None and model.y_0 is not None:
         print(f"Learned Center: ({model.x_0.item():.4f}, {model.y_0.item():.4f})")
   
   # Return the original model (not scripted) so SaveSplineWeights can be called
   return model

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

def EvaluateAndPlot(ax, model, target_func, title, x_range, device):
   x_plot = torch.linspace(x_range[0]-0.5, x_range[1]+0.5, 500).reshape(-1, 1)
   y_true_plot = target_func(x_plot)
   
   model.eval()
   with torch.no_grad():
      if model.inverse:
         y_input_plot = y_true_plot
         x_pred_plot, _ = model(y_input_plot.to(device))
         x_to_plot, y_to_plot = x_pred_plot, y_input_plot
      else:
         y_pred_plot, _ = model(x_plot.to(device))
         x_to_plot, y_to_plot = x_plot, y_pred_plot
   
   ax.plot(x_plot.cpu().numpy(), y_true_plot.cpu().numpy(), 
           label='True Function', color='blue', linewidth=2, alpha=0.6)
   ax.plot(x_to_plot.cpu().numpy(), y_to_plot.cpu().numpy(), 
           label='Spline Approximation', color='red', linestyle='--', linewidth=2)
   ax.set_title(title)
   ax.legend()
   ax.grid(True)

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

def TestMode2Roundtrip(trained_model, device):
   """Test Mode 2 equivalence and inverse operation"""
   direction_str = 'Increasing' if trained_model.direction_multiplier > 0 else 'Decreasing'
   print(f"\nTesting Roundtrip: Dir={direction_str}, Centered={trained_model.centered}, Inverse={trained_model.inverse}")
   
   # Initialize Mode 2 layers
   config = {'n_of_nodes': None, 'direction': trained_model.direction_multiplier, 'centered': trained_model.centered}
   spline_mode2_equivalent = UnifiedMonotonicSpline(**config, inverse=trained_model.inverse).to(device)
   spline_mode2_opposite = UnifiedMonotonicSpline(**config, inverse=not trained_model.inverse).to(device)
   
   params = ExtractParamsForExternal(trained_model)
   
   # Test Mode 2 matches Mode 1
   x_test = torch.linspace(-3., 3., 15).reshape(-1, 1).to(device)
   params_batch = params.repeat(15, 1)
   
   spline_mode2_equivalent.eval()
   trained_model.eval()
   
   with torch.no_grad():
      y_pred_mode2, _ = spline_mode2_equivalent(x_test, params_batch)
      y_pred_mode1, _ = trained_model(x_test)
   
   match = torch.allclose(y_pred_mode2, y_pred_mode1, atol=1e-6)
   print(f"Mode 2 matches Mode 1: {match}")
   if not match:
      print(f"Max diff: {torch.max(torch.abs(y_pred_mode2 - y_pred_mode1)).item()}")
   assert match, "Mode 2 equivalence failed"
   
   # Test roundtrip
   x_test_rt = torch.linspace(-4., 4., 50).reshape(-1, 1).to(device)
   params_batch_rt = params.repeat(50, 1)
   
   spline_mode2_opposite.eval()
   with torch.no_grad():
      y_transformed, _ = spline_mode2_equivalent(x_test_rt, params_batch_rt)
      x_recovered, _ = spline_mode2_opposite(y_transformed, params_batch_rt)
   
   match = torch.allclose(x_test_rt, x_recovered, atol=1e-4)
   max_diff = torch.max(torch.abs(x_test_rt - x_recovered)).item()
   print(f"Roundtrip accurate: {match}. Max diff: {max_diff:.6f}")
   assert match, "Roundtrip failed"

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# Helper function to safely extract internal parameters for testing purposes
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
def GetInternalParamsForTest(m):
   params = []
   if m.x_pos is not None: params.append(m.x_pos)
   if m.x_neg is not None: params.append(m.x_neg)
   if m.y_pos is not None: params.append(m.y_pos)
   if m.y_neg is not None: params.append(m.y_neg)
   if m.ln_d is not None: params.append(m.ln_d)
   if not m.centered:
      if m.x_0 is not None: params.append(m.x_0)
      if m.y_0 is not None: params.append(m.y_0)
   return params

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# Test Function for Derivatives
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
def TestDerivatives(model, device, x_range=(-3, 3)):
   """Test analytical derivatives against autograd and verify F'(x)*H'(y)=1."""
   
   if not model.use_internal_params:
      # These tests rely on internal parameters for setup and comparison.
      return

   direction_str = 'Increasing' if model.direction_multiplier > 0 else 'Decreasing'
   print(f"\nTesting Derivatives: Dir={direction_str}, Centered={model.centered}, Inverse={model.inverse}")

   n_samples = 100
   model.eval()

   # 1. Test analytical vs autograd derivative for the current model configuration
   print("Testing Analytical vs Autograd derivative.")

   # Determine the input range for testing. This is crucial for inverse models.
   if model.inverse:
      # If inverse (H(y)=x), the input is y. We need the range of y values covered by the spline.
      with torch.no_grad():
         params_list = GetInternalParamsForTest(model)
         # Ensure parameters are on the correct device before calling internal methods
         params_list = [p.to(device) for p in params_list]
            
         # _CalculateKnots returns the knots of G.
         _, y_knots, _, _ = model._CalculateKnots(params_list)
         y_min = y_knots.min().item()
         y_max = y_knots.max().item()

      # Input y covers the range of G(x), plus some extrapolation room.
      input_test = torch.linspace(y_min - 1.0, y_max + 1.0, n_samples).reshape(-1, 1).to(device)
      
   else:
      # If forward (F(x)=y), the input is x. We use the provided x_range plus extrapolation room.
      input_test = torch.linspace(x_range[0]-0.5, x_range[1]+0.5, n_samples).reshape(-1, 1).to(device)

   # Create a version for autograd testing
   input_test_grad = input_test.clone().detach().requires_grad_(True)

   # Analytical derivative (using the Deriv method)
   with torch.no_grad():
      d_analytical = model.Deriv(input_test)

   # Autograd derivative
   output, _ = model(input_test_grad)
   # Calculate d(output)/d(input) using autograd. 
   output.sum().backward()
   d_autograd = input_test_grad.grad

   # Compare results
   # Autograd might accumulate small numerical errors (float32 precision).
   match = torch.allclose(d_analytical, d_autograd, atol=1e-5)
   max_diff = torch.max(torch.abs(d_analytical - d_autograd)).item()
   print(f"Analytical vs Autograd match: {match}. Max diff: {max_diff:.6f}")

   assert match, f"Derivative test failed: Analytical does not match Autograd. Max diff: {max_diff:.6f}"

   # 2. Test relationship between forward and inverse derivatives (F'(x) * H'(y) = 1)
   # This uses Mode 2 to easily create both forward and inverse splines with the same parameters.

   print("Testing Forward/Inverse derivative relationship (Mode 2 roundtrip).")
   config = {'n_of_nodes': None, 'direction': model.direction_multiplier, 'centered': model.centered}
   spline_forward = UnifiedMonotonicSpline(**config, inverse=False).to(device)
   spline_inverse = UnifiedMonotonicSpline(**config, inverse=True).to(device)
   params = ExtractParamsForExternal(model)

   # We test this relationship using inputs 'x' in the forward direction.
   x_test_rel = torch.linspace(x_range[0]-0.5, x_range[1]+0.5, n_samples).reshape(-1, 1).to(device)
   params_batch = params.repeat(n_samples, 1)

   spline_forward.eval()
   spline_inverse.eval()

   with torch.no_grad():
      # Utilize the optimized path: Calculate F(x) and F'(x) simultaneously
      y_test_rel, dy_dx = spline_forward(x_test_rel, params_batch, return_deriv=True)

      # Calculate inverse derivative H'(y) at y values
      dx_dy = spline_inverse.Deriv(y_test_rel, params_batch)

   # Check if F'(x) * H'(y) = 1
   product = dy_dx * dx_dy
   target = torch.ones_like(product)

   # Use a slightly higher tolerance as numerical errors might accumulate.
   match = torch.allclose(product, target, atol=1e-4)
   max_diff = torch.max(torch.abs(product - target)).item()
   print(f"F'(x) * H'(y) == 1: {match}. Max diff: {max_diff:.6f}")

   assert match, f"Derivative relationship test failed: F'(x) * H'(y) != 1. Max diff: {max_diff:.6f}"

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# Test Function for forward(return_deriv=True)
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
def TestForwardWithDeriv(model, device, x_range=(-3, 3)):
   """Test the optimized forward pass that returns both value and derivative."""
   
   if not model.use_internal_params:
      return

   direction_str = 'Increasing' if model.direction_multiplier > 0 else 'Decreasing'
   print(f"\nTesting forward(return_deriv=True): Dir={direction_str}, Centered={model.centered}, Inverse={model.inverse}")

   n_samples = 50
   model.eval()

   # Determine the input range
   if model.inverse:
      with torch.no_grad():
         params_list = GetInternalParamsForTest(model)
         params_list = [p.to(device) for p in params_list]
         _, y_knots, _, _ = model._CalculateKnots(params_list)
         y_min = y_knots.min().item()
         y_max = y_knots.max().item()
      input_test = torch.linspace(y_min - 0.5, y_max + 0.5, n_samples).reshape(-1, 1).to(device)
   else:
      input_test = torch.linspace(x_range[0]-0.5, x_range[1]+0.5, n_samples).reshape(-1, 1).to(device)

   # 1. Calculate using the optimized path (return_deriv=True)
   with torch.no_grad():
      start_time_opt = time.time()
      val_both, deriv_both = model(input_test, return_deriv=True)
      time_opt = time.time() - start_time_opt

   # 2. Calculate separately using standard paths
   with torch.no_grad():
      start_time_sep = time.time()
      val_separate, deriv_empty = model(input_test, return_deriv=False)
      deriv_separate = model.Deriv(input_test)
      time_sep = time.time() - start_time_sep

   # Verify that the derivative is empty when not requested
   assert deriv_empty.numel() == 0, "Derivative should be empty when return_deriv=False"

   # 3. Compare results
   match_val = torch.allclose(val_both, val_separate, atol=1e-6)
   max_diff_val = torch.max(torch.abs(val_both - val_separate)).item()
   print(f"Value match (Both vs Separate): {match_val}. Max diff: {max_diff_val:.6f}")
   assert match_val, "Value mismatch in optimized path"

   match_deriv = torch.allclose(deriv_both, deriv_separate, atol=1e-6)
   max_diff_deriv = torch.max(torch.abs(deriv_both - deriv_separate)).item()
   print(f"Derivative match (Both vs Separate): {match_deriv}. Max diff: {max_diff_deriv:.6f}")
   assert match_deriv, "Derivative mismatch in optimized path"
   
   print(f"Timing (Optimized vs Separate): {time_opt*1000:.4f}ms vs {time_sep*1000:.4f}ms")

   # 4. Test Mode 2 (External Weights)
   print("Testing Mode 2 forward(return_deriv=True).")
   config = {'n_of_nodes': None, 'direction': model.direction_multiplier, 'centered': model.centered}
   spline_mode2 = UnifiedMonotonicSpline(**config, inverse=model.inverse).to(device)
   params = ExtractParamsForExternal(model)
   params_batch = params.repeat(n_samples, 1)

   spline_mode2.eval()
   with torch.no_grad():
      val_mode2, deriv_mode2 = spline_mode2(input_test, params_batch, return_deriv=True)

   match_val_m2 = torch.allclose(val_mode2, val_both, atol=1e-6)
   match_deriv_m2 = torch.allclose(deriv_mode2, deriv_both, atol=1e-6)

   print(f"Mode 2 Value match: {match_val_m2}")
   print(f"Mode 2 Derivative match: {match_deriv_m2}")

   assert match_val_m2, "Mode 2 Value mismatch."
   assert match_deriv_m2, "Mode 2 Derivative mismatch."

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# Test Function for TorchScript Compatibility
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
def TestScriptability(model, device):
   """Test if the model can be compiled with TorchScript."""
   
   if not model.use_internal_params:
      return

   direction_str = 'Increasing' if model.direction_multiplier > 0 else 'Decreasing'
   print(f"\nTesting TorchScript compatibility: Dir={direction_str}, Centered={model.centered}, Inverse={model.inverse}")
   
   sample_input = torch.randn(2, 1).to(device)
   model.eval()

   # 1. Test scripting the module itself (Mode 1)
   try:
      scripted_model = torch.jit.script(model)
      print("Mode 1 scripting: Success.")
      
      # Verify execution (Eager vs Scripted)
      with torch.no_grad():
         # Test standard forward (return_deriv=False)
         out_eager_std, deriv_eager_empty = model(sample_input, return_deriv=False)
         out_script_std, deriv_script_empty = scripted_model(sample_input, None, False)
         
         # Test forward with derivative (return_deriv=True)
         out_eager_both, deriv_eager_both = model(sample_input, return_deriv=True)
         out_script_both, deriv_script_both = scripted_model(sample_input, None, True)
      
      match_std = torch.allclose(out_eager_std, out_script_std)
      match_empty = (deriv_eager_empty.numel() == 0) and (deriv_script_empty.numel() == 0)
      match_val = torch.allclose(out_eager_both, out_script_both)
      match_deriv = torch.allclose(deriv_eager_both, deriv_script_both)
      
      print(f"Mode 1 execution match (Std/EmptyDeriv/Val/Deriv): {match_std} / {match_empty} / {match_val} / {match_deriv}")
      assert match_std and match_empty and match_val and match_deriv, "Mode 1 scripted execution mismatch."

   except Exception as e:
      print(f"Mode 1 scripting: Failed.")
      print(e)
      assert False, "Mode 1 scripting failed."

   # 2. Test scripting Mode 2 (External weights)
   config = {'n_of_nodes': None, 'direction': model.direction_multiplier, 'centered': model.centered}
   spline_mode2 = UnifiedMonotonicSpline(**config, inverse=model.inverse).to(device)
   spline_mode2.eval()

   params = ExtractParamsForExternal(model)
   sample_weights = params.repeat(2, 1)

   try:
      scripted_mode2 = torch.jit.script(spline_mode2)
      print("Mode 2 scripting: Success.")

      with torch.no_grad():
         out_eager, deriv_eager = spline_mode2(sample_input, sample_weights, return_deriv=True)
         out_script, deriv_script = scripted_mode2(sample_input, sample_weights, True)
         
      match_val = torch.allclose(out_eager, out_script)
      match_deriv = torch.allclose(deriv_eager, deriv_script)
      print(f"Mode 2 execution match (Val/Deriv): {match_val} / {match_deriv}")
      assert match_val and match_deriv, "Mode 2 scripted execution mismatch."

   except Exception as e:
      print(f"Mode 2 scripting: Failed.")
      print(e)
      assert False, "Mode 2 scripting failed."

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

def TestTrainingGradients(device, n_nodes=5, centered=True, direction=1):
   """Test gradient flow in Mode 2"""
   direction_str = 'Increasing' if direction > 0 else 'Decreasing'
   print(f"\n--- Testing Gradient Flow (Mode 2, Centered={centered}, Dir={direction_str}) ---")
   
   torch.manual_seed(42)
   b = 4  # Batch size
   d = 1  # Input dimension
   w = 8 * n_nodes + 1 + (0 if centered else 2)
   
   # Setup model
   model = UnifiedMonotonicSpline(n_of_nodes=None, inverse=False, centered=centered, direction=direction).to(device)
   criterion = nn.MSELoss()
   
   # Initialize trainable parameters
   input_params = torch.randn(b, d, device=device, requires_grad=True)
   weight_params = torch.randn(b, w, device=device, requires_grad=True)
   with torch.no_grad():
      weight_params *= 0.01
   
   target_y = torch.tensor([[1.], [0.5], [-0.5], [-1.]], device=device)
   
   # Training step
   optimizer = torch.optim.Adam([input_params, weight_params], lr=0.01)
   model.train()
   
   # Test standard forward gradient flow
   optimizer.zero_grad()
   y_pred, _ = model(input_params, weight_params)
   loss = criterion(y_pred, target_y)
   loss.backward()
   
   # Verify gradients
   has_input_grad = input_params.grad is not None and torch.norm(input_params.grad) > 1e-9
   has_weight_grad = weight_params.grad is not None and torch.norm(weight_params.grad) > 1e-9
   
   print(f"Gradient flow (Standard) to Input Data: {has_input_grad}")
   print(f"Gradient flow (Standard) to External Weights: {has_weight_grad}")
   
   assert has_input_grad, "No gradient for Input Data (Standard)"
   assert has_weight_grad, "No gradient for External Weights (Standard)"
   
   # Test gradient flow when using return_deriv=True (gradients must flow through the value)
   optimizer.zero_grad()
   y_pred_both, deriv_pred = model(input_params, weight_params, return_deriv=True)
   loss_both = criterion(y_pred_both, target_y)
   loss_both.backward()

   has_input_grad_both = input_params.grad is not None and torch.norm(input_params.grad) > 1e-9
   has_weight_grad_both = weight_params.grad is not None and torch.norm(weight_params.grad) > 1e-9

   print(f"Gradient flow (return_deriv=True) to Input Data: {has_input_grad_both}")
   print(f"Gradient flow (return_deriv=True) to External Weights: {has_weight_grad_both}")

   assert has_input_grad_both, "No gradient for Input Data (return_deriv=True)"
   assert has_weight_grad_both, "No gradient for External Weights (return_deriv=True)"
   
   if not centered:
      grad_x0 = weight_params.grad[..., -2]
      grad_y0 = weight_params.grad[..., -1]
      has_x0_grad = torch.norm(grad_x0) > 1e-9
      has_y0_grad = torch.norm(grad_y0) > 1e-9
      print(f"Gradient flow to x_0: {has_x0_grad}")
      print(f"Gradient flow to y_0: {has_y0_grad}")
      assert has_x0_grad and has_y0_grad, "No gradient for center parameters"

##################################################################################################################################################
# Main 
##################################################################################################################################################
if __name__ == '__main__':
   device = torch.device("cpu")
   print("Running tests on CPU")
   
   n_nodes = 3
   
   # Train Mode 1 instances
   print("\n" + "="*60)
   print("Training Mode 1 Splines")
   print("="*60)
   
   model_tanh = TrainSplineMode1(TanhTarget, "y = tanh(2x)", device, n_nodes=n_nodes, epochs=2000)
   model_tanh.SaveSplineWeights("spline_tanh.txt")
   model_decreasing = TrainSplineMode1(DecreasingTarget, "y = -tanh(2x)", device, n_nodes=n_nodes, epochs=2000, direction='decreasing')
   model_decreasing.SaveSplineWeights("spline_decreasing.txt")
   model_non_centered = TrainSplineMode1(NonCenteredTarget, "y = (x-3)^3+5", device, n_nodes=n_nodes+1, epochs=4000, x_range=(-4, 7), centered=False)
   model_non_centered.SaveSplineWeights("spline_non_centered.txt")
   model_inverse = TrainSplineMode1(CubeTarget, "x = cbrt(y)", device, n_nodes=n_nodes, epochs=2000, x_range=(-2, 2), inverse=True)
   model_inverse.SaveSplineWeights("spline_inverse.txt")
   model_inverse_decreasing = TrainSplineMode1(NegCubeTarget, "x = cbrt(-y)", device, n_nodes=n_nodes, epochs=2000, x_range=(-2, 2), inverse=True, direction=-1)
   model_inverse_decreasing.SaveSplineWeights("spline_inverse_decreasing.txt")
   
   models = [model_tanh, model_decreasing, model_non_centered, model_inverse, model_inverse_decreasing]
   # Corresponding X ranges used during training (relevant for derivative tests)
   ranges = [(-3, 3), (-3, 3), (-4, 7), (-2, 2), (-2, 2)]
   
   # Run tests
   print("\n" + "="*60)
   print("Test Group 1: Mode 2 Equivalence and Roundtrip")
   print("="*60)
   for model in models:
      if model: TestMode2Roundtrip(model, device)
   
   print("\n" + "="*60)
   print("Test Group 2: Derivatives (Analytical vs Autograd and F' * H' = 1)")
   print("="*60)
   for model, x_range in zip(models, ranges):
      if model: TestDerivatives(model, device, x_range)

   print("\n" + "="*60)
   print("Test Group 3: Optimized Forward (Value + Derivative)")
   print("="*60)
   for model, x_range in zip(models, ranges):
      if model: TestForwardWithDeriv(model, device, x_range)

   # Test Group for Scriptability
   print("\n" + "="*60)
   print("Test Group 4: TorchScript Compatibility")
   print("="*60)
   for model in models:
      if model: TestScriptability(model, device)

   print("\n" + "="*60)
   print("Test Group 5: Gradient Flow (Mode 2 Training)")
   print("="*60)
   TestTrainingGradients(device, n_nodes=5, centered=True, direction=1)
   TestTrainingGradients(device, n_nodes=5, centered=False, direction=1)
   TestTrainingGradients(device, n_nodes=5, centered=True, direction=-1)
   TestTrainingGradients(device, n_nodes=5, centered=False, direction=-1)
   
   # Plot results
   fig, axes = plt.subplots(2, 3, figsize=(18, 10))
   (ax1, ax2, ax3), (ax4, ax5, ax6) = axes
   
   if model_tanh:
      EvaluateAndPlot(ax1, model_tanh, TanhTarget, 'Mode 1: Increasing, Centered', (-3, 3), device)
   if model_decreasing:
      EvaluateAndPlot(ax2, model_decreasing, DecreasingTarget, 'Mode 1: Decreasing, Centered', (-3, 3), device)
   if model_non_centered:
      EvaluateAndPlot(ax3, model_non_centered, NonCenteredTarget, 'Mode 1: Increasing, Non-Centered', (-5, 8), device)
      if not model_non_centered.centered and model_non_centered.x_0 is not None and model_non_centered.y_0 is not None:
         with torch.no_grad():
            x0 = model_non_centered.x_0.item()
            y0 = model_non_centered.y_0.item()
            ax3.scatter([3.], [5.], color='blue', marker='x', s=100, label='True Center (3, 5)', zorder=5)
            ax3.scatter([x0], [y0], color='green', marker='o', s=100, label=f'Learned ({x0:.2f}, {y0:.2f})', zorder=5)
            ax3.legend()
   
   if model_inverse:
      EvaluateAndPlot(ax4, model_inverse, CubeTarget, 'Mode 1 Inverse: Increasing', (-2, 2), device)
   if model_inverse_decreasing:
      EvaluateAndPlot(ax5, model_inverse_decreasing, NegCubeTarget, 'Mode 1 Inverse: Decreasing', (-2, 2), device)
   
   ax6.axis('off')
   plt.tight_layout()
   plt.show()
   
   print("\nAll tests completed successfully")