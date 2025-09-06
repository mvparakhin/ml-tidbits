#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <functional>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <cassert>
#include <iomanip>
#include <utility>

#include "../mlrsplines/LRSplines.h" // Assuming the latest C++ implementation is saved here
using namespace ns_base;

// Define the type and precision for testing
using t_test_val = double;
constexpr t_test_val gc_test_tolerance = 1e-6;

// Aliases for the two modes
using t_spline_internal = T_LRSplines<t_test_val>;
using t_spline_external = T_LRSplinesInput<t_test_val>;

//#########################################################################################################################################################################################

// Finite differences derivative approximation using Richardson extrapolation
template<typename T_Func>
double NumericalDerivative(T_Func func, double x, double h, double* p_err = nullptr) {
   constexpr size_t c_max_iterations = 10;
   constexpr double c_contraction_factor = 1.4;
   constexpr double c_contraction_squared = c_contraction_factor * c_contraction_factor;
   constexpr double c_large_error = 1.0e30;
   constexpr double c_safety_factor = 2.0;
   
   std::vector<std::vector<double>> tableau(c_max_iterations, std::vector<double>(c_max_iterations));
   
   if (h == 0.0) {
      if (p_err) *p_err = c_large_error;
      return 0.0;
   }
   
   double step_size = h;
   double best_estimate = 0.0;
   double min_error = c_large_error;
   
   // Initial approximation using central differences
   tableau[0][0] = (func(x + step_size) - func(x - step_size)) / (2.0 * step_size);
   if (p_err) *p_err = c_large_error;
   
   for (size_t i = 1; i < c_max_iterations; ++i) {
      step_size /= c_contraction_factor;
      
      // Compute new approximation with smaller step
      tableau[0][i] = (func(x + step_size) - func(x - step_size)) / (2.0 * step_size);
      
      // Richardson extrapolation to eliminate higher-order error terms
      double extrapolation_factor = c_contraction_squared;
      for (size_t j = 1; j <= i; ++j) {
         tableau[j][i] = (tableau[j-1][i] * extrapolation_factor - tableau[j-1][i-1]) / 
                        (extrapolation_factor - 1.0);
         extrapolation_factor *= c_contraction_squared;
         
         // Estimate error as maximum difference from neighbors
         double error_estimate = std::max(std::abs(tableau[j][i] - tableau[j-1][i]), 
                                         std::abs(tableau[j][i] - tableau[j-1][i-1]));
         
         // Update best estimate if error improved
         if (error_estimate <= min_error) {
            min_error = error_estimate;
            best_estimate = tableau[j][i];
            if (p_err) *p_err = error_estimate;
         }
      }
      
      // Stop if error is no longer improving significantly
      if (std::abs(tableau[i][i] - tableau[i-1][i-1]) >= c_safety_factor * min_error)
         break;
   }
   
   return best_estimate;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Helper to load parameters from a V1001 (PyTorch) file for External mode testing
// 
// NOTE ON PARAMETERIZATION:
// The C++ implementation's CalculateKnots function always computes exp(P - log(2))
// The Internal mode's TextLoad function adds log(2) upon loading V1001 files
// To use External mode with V1001 files, we must also add log(2) to the X parameters here
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<t_test_val> LoadExternalWeights(const std::string& filename) {
   std::ifstream file(filename);
   if (!file)
      throw std::runtime_error("Cannot open file (ensure it is in the working directory): " + filename);

   std::vector<std::vector<t_test_val>> data;
   std::string line;

   // Read data, skipping headers
   while (std::getline(file, line)) {
      if (line.empty() || line.rfind("#VER", 0) == 0) continue;

      std::vector<t_test_val> values;
      std::stringstream ss(line);
      t_test_val value;
      while (ss >> value)
         values.push_back(value);
      if (!values.empty())
         data.push_back(values);
   }

   if (data.size() < 5)
      throw std::runtime_error("Incomplete data in file: " + filename);

   // Convert P_internal (file) to P_legacy (external buffer)
   // This constant must match the one used in the C++ implementation
   const t_test_val c_log2 = t_test_val(0.6931471805599453);

   // Adjust x_pos (data[0]) and x_neg (data[1])
   for (auto& v : data[0]) v += c_log2;
   for (auto& v : data[1]) v += c_log2;

   // Flatten the data in the order expected by ProcessExternalParams:
   // x_pos, x_neg, y_pos, y_neg, ln_d, [x_0, y_0]
   std::vector<t_test_val> weights;
   for(const auto& section : data)
      weights.insert(weights.end(), section.begin(), section.end());

   return weights;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Helper for assertion
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void AssertNear(t_test_val a, t_test_val b, const std::string& msg, bool relative = false, double tolerance = gc_test_tolerance) {
   if ((!relative && std::abs(a - b) > tolerance) || (relative && std::abs(a - b) > tolerance * std::max(std::abs(a), std::abs(b)))) {
      std::cerr << "ASSERTION FAILED: " << msg << "\n"
                << "  Value A: " << a << "\n  Value B: " << b << std::endl;
      throw std::runtime_error("Test failed assertion.");
   }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Test Runner
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void RunTest(const std::string& name, std::function<void()> test_func) {
   try {
      test_func();
      std::cout << "[PASSED] " << name << std::endl;
   } catch (const std::exception& e) {
      // std::abort() is used here to stop execution immediately upon failure
      std::cerr << "[FAILED] " << name << "\n  Reason: " << e.what() << std::endl;
      std::abort();
   }
}

// Define test inputs spanning interpolation and extrapolation regions
const std::vector<t_test_val> gc_test_inputs = {-5., -1.5, -0.5, 0., 0.5, 1.5, 5.};

//#########################################################################################################################################################################################

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Primary test function: Verifies consistency across all modes (Internal, External Cached, External Fly)
// and checks the fundamental properties (monotonicity, inverse identity, derivatives)
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void TestConsistency(const std::string& filename, int direction, bool is_centered) {

   // 1. Internal Mode (T_LRSplines)
   // Initialize with the correct direction. TextLoad handles centering detection and parameter loading
   t_spline_internal spline_int(is_centered, direction);
   spline_int.TextLoad(filename);

   const auto& container = std::as_const(spline_int).Container();
   double total_err = 0.;
   for (auto x_val : gc_test_inputs) {
      auto grad = spline_int.CalculateGradients(x_val);
      for (size_t group_id = 0; group_id < container.NOfGroups(); ++group_id) {
         if (spline_int.IsCentered() && N_MLRSParamType(group_id) == lrsX0Y0)
            continue;
         for (size_t param_id = 0; param_id < container.size(N_MLRSParamType(group_id)); ++param_id) {
            t_spline_internal::S_GradVerifier f(spline_int, group_id, param_id, x_val, false);
            double err = 0;
            double analytical_grad = grad(N_MLRSParamType(group_id), param_id);
            double numerical_grad = NumericalDerivative(f, container(N_MLRSParamType(group_id), param_id), 0.001, &err);
            total_err += fabs(analytical_grad - numerical_grad);
            //std::cout << group_id << " " << param_id << " " << analytical_grad << " " << numerical_grad << " " << err << std::endl;
         }
      }

      grad = spline_int.CalculateInverseGradients(x_val); //here it is really y_val
      for (size_t group_id = 0; group_id < container.NOfGroups(); ++group_id) {
         if (spline_int.IsCentered() && N_MLRSParamType(group_id) == lrsX0Y0)
            continue;
         for (size_t param_id = 0; param_id < container.size(N_MLRSParamType(group_id)); ++param_id) {
            t_spline_internal::S_GradVerifier f(spline_int, group_id, param_id, x_val, true);
            double err = 0;
            double analytical_grad = grad(N_MLRSParamType(group_id), param_id);
            double numerical_grad = NumericalDerivative(f, container(N_MLRSParamType(group_id), param_id), 0.001, &err);
            total_err += fabs(analytical_grad - numerical_grad);
            //std::cout << group_id << " " << param_id << " " << analytical_grad << " " << numerical_grad << " " << err << std::endl;
         }
      }
   }
   std::cout << std::scientific;
   std::cout << "\nTotal grad error = " << total_err << "\n" << std::endl;
   std::cout << std::fixed;
   AssertNear(total_err, 0., "Total grad error", false, 0.0005);

   // 2. External Mode Setup
   auto weights = LoadExternalWeights(filename);
   const t_test_val* p_params = weights.data();
   size_t n_params = weights.size();

   // 3. External Mode - On-the-fly (T_LRSplinesInput)
   // Must be initialized with the correct centering matching the parameters
   t_spline_external spline_fly(is_centered, direction);

   // 4. External Mode - Cached (T_LRSplinesInput)
   t_spline_external spline_cached(is_centered, direction);
   spline_cached.UpdateCache(p_params, n_params);

   // 5. External Mode - Constructor with params (if applicable)
   std::unique_ptr<t_spline_external> p_spline_ctor;
   if (is_centered && direction == 1)
      p_spline_ctor = std::make_unique<t_spline_external>(p_params, n_params);

   t_test_val prev_y = (direction == 1) ? -std::numeric_limits<t_test_val>::infinity() : std::numeric_limits<t_test_val>::infinity();

   // Define lambdas for numerical differentiation (w.r.t input), using Internal mode as the reference.
   auto func_forward = [&](t_test_val x) { return spline_int.Calc(x); };
   auto func_inverse = [&](t_test_val y) { return spline_int.CalcInv(y); };

   // --- Main Consistency Loop (Values and Derivatives w.r.t Input) ---
   for (t_test_val x : gc_test_inputs) {
      
      // --- Calculate Values (y) ---
      t_test_val y_int = spline_int.Calc(x);
      t_test_val y_fly = spline_fly.Calc(p_params, n_params, x);
      t_test_val y_cached = spline_cached.Calc(x);

      // --- Calculate Forward Derivatives (dy/dx) ---
      t_test_val dy_dx_int = spline_int.CalcDeriv(x);
      t_test_val dy_dx_fly = spline_fly.CalcDeriv(p_params, n_params, x);
      t_test_val dy_dx_cached = spline_cached.CalcDeriv(x);

      // --- Calculate Inverse Derivatives (dx/dy) ---
      // Evaluated at y = f(x)
      t_test_val dx_dy_int = spline_int.CalcInvDeriv(y_int);
      t_test_val dx_dy_fly = spline_fly.CalcInvDeriv(p_params, n_params, y_fly);
      t_test_val dx_dy_cached = spline_cached.CalcInvDeriv(y_cached);

      // --------------------------------------------------------------------------------
      // A. Consistency Checks (Across Modes) - Should match exactly (use standard tolerance)
      // --------------------------------------------------------------------------------
      AssertNear(y_int, y_fly, "Consistency (Value): Internal vs Fly");
      AssertNear(y_int, y_cached, "Consistency (Value): Internal vs Cached");
      AssertNear(dy_dx_int, dy_dx_fly, "Consistency (Deriv): Internal vs Fly");
      AssertNear(dy_dx_int, dy_dx_cached, "Consistency (Deriv): Internal vs Cached");
      AssertNear(dx_dy_int, dx_dy_fly, "Consistency (InvDeriv): Internal vs Fly");
      AssertNear(dx_dy_int, dx_dy_cached, "Consistency (InvDeriv): Internal vs Cached");

      if (p_spline_ctor) {
         AssertNear(y_int, p_spline_ctor->Calc(x), "Consistency (Value): Internal vs Constructor");
         AssertNear(dy_dx_int, p_spline_ctor->CalcDeriv(x), "Consistency (Deriv): Internal vs Constructor");
         // Evaluate constructor spline inverse derivative at y_int for comparison
         AssertNear(dx_dy_int, p_spline_ctor->CalcInvDeriv(y_int), "Consistency (InvDeriv): Internal vs Constructor");
      }

      if (p_spline_ctor)
         AssertNear(y_int, p_spline_ctor->Calc(x), "Consistency: Internal vs Constructor");

      // Check forward/inverse consistency (x = CalcInv(Calc(x)))
      t_test_val x_inv_int = spline_int.CalcInv(y_int);
      t_test_val x_inv_cached = spline_cached.CalcInv(y_cached);

      AssertNear(x, x_inv_int, "Inverse Consistency: Internal");
      AssertNear(x, x_inv_cached, "Inverse Consistency: Cached");

      // Check monotonicity
      if (direction == 1) {
         assert(y_int >= prev_y);
         assert(dy_dx_int >= 0.); // Increasing spline must have non-negative derivative
      } else {
         assert(y_int <= prev_y);
         assert(dy_dx_int <= 0.); // Decreasing spline must have non-positive derivative
      }
      prev_y = y_int;

      // Check Inverse Function Theorem (dy/dx * dx/dy = 1)
      // Using multiplication avoids division by zero issues in saturation regions.
      t_test_val identity_product = dy_dx_int * dx_dy_int;
      // We expect the product to be 1. If the product is very small (near 0), it implies saturation, 
      // where both derivatives are near 0, which is also valid behavior.
      // Only assert it's 1 if the product is significantly different from 0.
      if (std::abs(identity_product) > gc_test_tolerance) {
          AssertNear(identity_product, 1.0, "Inverse Function Theorem (dy/dx * dx/dy = 1)");
      }

      // --------------------------------------------------------------------------------
      // C. Numerical Verification (Analytical vs Finite Differences) - Use derivative tolerance
      // --------------------------------------------------------------------------------
      
      // Using a step size of 0.01 for numerical approximation w.r.t input
      t_test_val dy_dx_num = NumericalDerivative(func_forward, x, 0.0001);
      AssertNear(dy_dx_int, dy_dx_num, "Numerical Verification: Forward Derivative (dy/dx)", true, 0.0001);

      t_test_val dx_dy_num = NumericalDerivative(func_inverse, y_int, 0.0001);
      AssertNear(dx_dy_int, dx_dy_num, "Numerical Verification: Inverse Derivative (dx/dy)", true, 0.0001);
   }
   std::cout << std::endl; // Formatting after gradient error output
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Test specific API behaviors: Initialization errors and Move Semantics
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void TestAPIFeatures() {
   // 1. Uninitialized Cache Error (External Mode)
   t_spline_external spline_uninit;
   bool caught = false;
   
   // Check Calc
   try { spline_uninit.Calc(1.); } catch (const LRSplinesException& e) { if (e.code() == 1) caught = true; }
   assert(caught && "Failed to throw on uninitialized external access (Calc).");

   // Check CalcInv
   caught = false;
   try { spline_uninit.CalcInv(1.); } catch (const LRSplinesException& e) { if (e.code() == 1) caught = true; }
   assert(caught && "Failed to throw on uninitialized external access (CalcInv).");

   // Check CalcDeriv
   caught = false;
   try { spline_uninit.CalcDeriv(1.); } catch (const LRSplinesException& e) { if (e.code() == 1) caught = true; }
   assert(caught && "Failed to throw on uninitialized external access (CalcDeriv).");

   // Check CalcInvDeriv
   caught = false;
   try { spline_uninit.CalcInvDeriv(1.); } catch (const LRSplinesException& e) { if (e.code() == 1) caught = true; }
   assert(caught && "Failed to throw on uninitialized external access (CalcInvDeriv).");


   // 2. Uninitialized Error (Internal Mode)
   t_spline_internal spline_uninit_int;
   caught = false;
   try { spline_uninit_int.Calc(1.); } catch (const LRSplinesException& e) { if (e.code() == 1) caught = true; }
   assert(caught && "Failed to throw on uninitialized internal access (Calc).");

   // Check CalcDeriv
   caught = false;
   try { spline_uninit_int.CalcDeriv(1.); } catch (const LRSplinesException& e) { if (e.code() == 1) caught = true; }
   assert(caught && "Failed to throw on uninitialized internal access (CalcDeriv).");


   // 3. Move Semantics (External Mode)
   auto weights = LoadExternalWeights("./data/spline_tanh.txt");
   t_spline_external src_ext(weights.data(), weights.size());
   t_test_val y_src = src_ext.Calc(1.);
   t_test_val dy_dx_src = src_ext.CalcDeriv(1.);

   t_spline_external dest_ext = std::move(src_ext);
   AssertNear(dest_ext.Calc(1.), y_src, "Move External: Destination Calc valid");
   AssertNear(dest_ext.CalcDeriv(1.), dy_dx_src, "Move External: Destination CalcDeriv valid");

   // Source must be invalid (empty cache)
   caught = false;
   try {
      src_ext.Calc(1.);
   } catch (const LRSplinesException& e) {
      if (e.code() == 1) caught = true;
   }
   assert(caught && "Source external object not invalidated after move.");

   // 4. Move Semantics (Internal Mode)
   t_spline_internal src_int;
   src_int.TextLoad("./data/spline_tanh.txt");
   y_src = src_int.Calc(1.);
   dy_dx_src = src_int.CalcDeriv(1.);

   t_spline_internal dest_int = std::move(src_int);
   AssertNear(dest_int.Calc(1.), y_src, "Move Internal: Destination Calc valid");
   AssertNear(dest_int.CalcDeriv(1.), dy_dx_src, "Move Internal: Destination CalcDeriv valid");

   // Source must be invalid
   caught = false;
   try {
      src_int.Calc(1.);
   } catch (const LRSplinesException& e) {
      if (e.code() == 1) caught = true;
   }
   assert(caught && "Source internal object not invalidated after move.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Main test function
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int TestLRSplines() {
   // Update title to reflect the extended tests
   std::cout << "Running T_UnifiedMonotonicSpline Concise Tests (Including Derivatives)..." << std::endl;
   std::cout << std::fixed << std::setprecision(8);

   // NOTE: These tests require the spline_*.txt files to be present in the working directory

   // Update test names to reflect the inclusion of derivative checks

   // Test standard configuration (Centered, Increasing)
   RunTest("Consistency & Derivatives: spline_tanh.txt (Increasing)", [](){
      TestConsistency("./data/spline_tanh.txt", 1, true);
   });

   // Test decreasing configuration (Centered, Decreasing)
   RunTest("Consistency & Derivatives: spline_decreasing.txt (Decreasing)", [](){
      TestConsistency("./data/spline_decreasing.txt", -1, true);
   });

   // Test non-centered configuration (Non-Centered, Increasing)
   RunTest("Consistency & Derivatives: spline_non_centered.txt (Non-Centered)", [](){
      // Must indicate 'false' for centering when initializing External mode configuration
      TestConsistency("./data/spline_non_centered.txt", 1, false);
   });

   // Test inverse files (Ensures robustness of Calc/CalcInv/Deriv logic across different training data)
   RunTest("Consistency & Derivatives: spline_inverse.txt (Increasing)", [](){
      TestConsistency("./data/spline_inverse.txt", 1, true);
   });

   RunTest("Consistency & Derivatives: spline_inverse_decreasing.txt (Decreasing)", [](){
      TestConsistency("./data/spline_inverse_decreasing.txt", -1, true);
   });

   // Test specific API behaviors
   RunTest("API Features (Init Errors, Move Semantics)", TestAPIFeatures);

   std::cout << "\nAll tests passed successfully!" << std::endl;
   return 0;
}

// Main entry point
int main() {
   return TestLRSplines();
}
