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
   
   # Training loop
   model.train()
   start_time = time.time()
   for epoch in range(epochs):
      optimizer.zero_grad()
      output_pred = model(input_train)
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
      print(f"Learned Center: ({model.x_0.item():.4f}, {model.y_0.item():.4f})")
   
   return model

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

def EvaluateAndPlot(ax, model, target_func, title, x_range, device):
   x_plot = torch.linspace(x_range[0]-0.5, x_range[1]+0.5, 500).reshape(-1, 1)
   y_true_plot = target_func(x_plot)
   
   model.eval()
   with torch.no_grad():
      if model.inverse:
         y_input_plot = y_true_plot
         x_pred_plot = model(y_input_plot.to(device))
         x_to_plot, y_to_plot = x_pred_plot, y_input_plot
      else:
         y_pred_plot = model(x_plot.to(device))
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
      y_pred_mode2 = spline_mode2_equivalent(x_test, params_batch)
      y_pred_mode1 = trained_model(x_test)
   
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
      y_transformed = spline_mode2_equivalent(x_test_rt, params_batch_rt)
      x_recovered = spline_mode2_opposite(y_transformed, params_batch_rt)
   
   match = torch.allclose(x_test_rt, x_recovered, atol=1e-4)
   max_diff = torch.max(torch.abs(x_test_rt - x_recovered)).item()
   print(f"Roundtrip accurate: {match}. Max diff: {max_diff:.6f}")
   assert match, "Roundtrip failed"

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
      # We inspect the knots of the internal increasing spline (G).
      with torch.no_grad():
         params_list = [model.x_pos, model.x_neg, model.y_pos, model.y_neg, model.ln_d]
         if not model.centered:
            params_list.extend([model.x_0, model.y_0])
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

   # Analytical derivative (using the new Deriv method)
   with torch.no_grad():
      d_analytical = model.Deriv(input_test)

   # Autograd derivative
   output = model(input_test_grad)
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
      # Calculate forward derivative F'(x)
      dy_dx = spline_forward.Deriv(x_test_rel, params_batch)

      # Calculate corresponding y values F(x)
      y_test_rel = spline_forward(x_test_rel, params_batch)

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
   optimizer.zero_grad()
   
   y_pred = model(input_params, weight_params)
   loss = criterion(y_pred, target_y)
   loss.backward()
   
   # Verify gradients
   has_input_grad = input_params.grad is not None and torch.norm(input_params.grad) > 1e-9
   has_weight_grad = weight_params.grad is not None and torch.norm(weight_params.grad) > 1e-9
   
   print(f"Gradient flow to Input Data: {has_input_grad}")
   print(f"Gradient flow to External Weights: {has_weight_grad}")
   
   assert has_input_grad, "No gradient for Input Data"
   assert has_weight_grad, "No gradient for External Weights"
   
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
   
   # New Test Group for Derivatives
   print("\n" + "="*60)
   print("Test Group 2: Derivatives (Analytical vs Autograd and F' * H' = 1)")
   print("="*60)
   for model, x_range in zip(models, ranges):
      if model: TestDerivatives(model, device, x_range)

   print("\n" + "="*60)
   print("Test Group 3: Gradient Flow (Mode 2 Training)")
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
      if not model_non_centered.centered:
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