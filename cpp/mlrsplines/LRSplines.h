#ifndef LR_SPLINES_H
#define LR_SPLINES_H

#include <vector>
#include <array>
#include <algorithm>
#include <cmath>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <type_traits>

namespace ns_base {
   
   // Simple exception class to replace the EXCEPT_MAP system
   class LRSplinesException : public std::runtime_error {
   public:
      LRSplinesException(int code, const std::string& file, int line, const std::string& msg = "") 
         : std::runtime_error(FormatMessage(code, file, line, msg)), code_(code) {}
      
      int code() const { return code_; }
      
   private:
      int code_;
      
      static std::string FormatMessage(int code, const std::string& file, int line, const std::string& msg) {
         std::ostringstream oss;
         oss << "LRSplines Error " << code << " at " << file << ":" << line;
         switch(code) {
            case 0: oss << " - Unknown error"; break;
            case 1: oss << " - Invariant is broken or spline not initialized"; break;
            case 2: oss << " - Incorrect parameters (e.g., size mismatch for external weights)"; break;
            case 3: oss << " - Incorrect file format or version"; break;
            case 4: oss << " - Invalid configuration (e.g., direction or centering conflict)"; break;
         }
         if (!msg.empty()) oss << " [" << msg << "]";
         return oss.str();
      }
   };

   template<typename T>
   constexpr T Epsilon() {
      if constexpr (std::is_same_v<T, float>)
         return 1e-6f;
      else
         return 1e-6;
   }

   enum N_SplineMode {
      smInternal,
      smExternal
   };

   //#########################################################################################################################################################################################

   template <class T, template<class> class T_Container>
   struct S_MLRSCacheStorage {
      using t_param_arr = T_Container<T>;

      t_param_arr x, y, w;
      T d_left{T(0.)}, d_right{T(0.)};

      // Explicitly define move operations to ensure the source cache is invalidated
      S_MLRSCacheStorage() = default;
      S_MLRSCacheStorage(const S_MLRSCacheStorage&) = default;
      S_MLRSCacheStorage& operator=(const S_MLRSCacheStorage&) = default;

      S_MLRSCacheStorage(S_MLRSCacheStorage&& rhs) noexcept : x(std::move(rhs.x)), y(std::move(rhs.y)), w(std::move(rhs.w)),
         d_left(rhs.d_left), d_right(rhs.d_right) {
         rhs.x.clear(); // Invalidate the source object
      }

      S_MLRSCacheStorage& operator=(S_MLRSCacheStorage&& rhs) noexcept {
         if (this != &rhs) {
            x = std::move(rhs.x);
            y = std::move(rhs.y);
            w = std::move(rhs.w);
            d_left = rhs.d_left;
            d_right = rhs.d_right;
            rhs.x.clear(); // Invalidate the source object
         }
         return *this;
      }
   };

   ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

   enum N_MLRSParamType {
      lrsXPos = 0,
      lrsXNeg,
      lrsYPos,
      lrsYNeg,
      lrsLnD,
      lrsX0Y0,
      lrsLast
   };

   template <class P_Container>
   struct S_MLRSUnconstrainedParams {
      using t_param_arr = P_Container;
      using t_val = typename P_Container::value_type;

      t_param_arr x_pos, x_neg, y_pos, y_neg, ln_d;
      t_val x_0{t_val(0.)}, y_0{t_val(0.)};

      S_MLRSUnconstrainedParams() = default;
      explicit S_MLRSUnconstrainedParams(size_t n) { Init(n); }
      S_MLRSUnconstrainedParams(const S_MLRSUnconstrainedParams&) = default;
      S_MLRSUnconstrainedParams& operator=(const S_MLRSUnconstrainedParams&) = default;
      S_MLRSUnconstrainedParams(S_MLRSUnconstrainedParams&& rhs) noexcept :
         x_pos(std::move(rhs.x_pos)), x_neg(std::move(rhs.x_neg)),
         y_pos(std::move(rhs.y_pos)), y_neg(std::move(rhs.y_neg)),
         ln_d(std::move(rhs.ln_d)),
         x_0(rhs.x_0), y_0(rhs.y_0) {
         rhs.x_pos.clear(); // Invalidate the source object
      }
      S_MLRSUnconstrainedParams& operator=(S_MLRSUnconstrainedParams&& rhs) noexcept {
         if (this != &rhs) {
            x_pos = std::move(rhs.x_pos);
            x_neg = std::move(rhs.x_neg);
            y_pos = std::move(rhs.y_pos);
            y_neg = std::move(rhs.y_neg);
            ln_d = std::move(rhs.ln_d);
            x_0 = rhs.x_0;
            y_0 = rhs.y_0;
            rhs.x_pos.clear(); // Invalidate the source object
         }
         return *this;
      }

      void Init(size_t n) {
         x_pos.resize(2*n, t_val(0.)); x_neg.resize(2*n, t_val(0.));
         y_pos.resize(n, t_val(0.)); y_neg.resize(n, t_val(0.));
         ln_d.resize(2*n + 1, t_val(0.));
         x_0 = y_0 = t_val(0.);
      }
      size_t N() const noexcept { return y_pos.size(); }
      size_t NOfGroups() const noexcept { return size_t(lrsLast); }
      size_t size(N_MLRSParamType group_id) const {
         switch(group_id) {
         case lrsXPos: return x_pos.size();
         case lrsXNeg: return x_neg.size();
         case lrsYPos: return y_pos.size();
         case lrsYNeg: return y_neg.size();
         case lrsLnD: return ln_d.size();
         case lrsX0Y0: return 2;
         default:
            throw LRSplinesException(2, __FILE__, __LINE__, std::to_string(size_t(group_id)));
         }
      }

      t_val& operator()(N_MLRSParamType group_id, size_t param_id) {
         switch(group_id) {
         case lrsXPos: return x_pos[param_id];
         case lrsXNeg: return x_neg[param_id];
         case lrsYPos: return y_pos[param_id];
         case lrsYNeg: return y_neg[param_id];
         case lrsLnD: return ln_d[param_id];
         case lrsX0Y0:
            if (param_id>1)
               throw LRSplinesException(2, __FILE__, __LINE__, std::to_string(param_id));
            return param_id? y_0 : x_0;
         default:
            throw LRSplinesException(2, __FILE__, __LINE__, std::to_string(size_t(group_id)));
         }
      }
      const t_val& operator()(N_MLRSParamType group_id, size_t param_id) const {
         return const_cast<S_MLRSUnconstrainedParams*>(this)->operator()(group_id, param_id);
      }
   };

   ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

   template <class T>
   struct S_MLRSInternalData : private S_MLRSCacheStorage<T, std::vector>, S_MLRSUnconstrainedParams<std::vector<T>> {
      using t_base1 = S_MLRSCacheStorage<T, std::vector>;
      using t_base2 = S_MLRSUnconstrainedParams<std::vector<T>>;
      using typename t_base1::t_param_arr;
      using t_base1::x;
      using t_base1::y;
      using t_base1::w;
      using t_base1::d_left;
      using t_base1::d_right;

      using t_base2::x_pos;
      using t_base2::x_neg;
      using t_base2::y_pos;
      using t_base2::y_neg;
      using t_base2::ln_d;
      using t_base2::x_0;
      using t_base2::y_0;


      using t_base2::N;
      using t_base2::NOfGroups;
      using t_base2::size;
      using t_base2::operator();
   };

   //#########################################################################################################################################################################################

   template<class P_Type, N_SplineMode P_Mode = smInternal>
   class T_UnifiedMonotonicSpline : private std::conditional_t<P_Mode == smInternal, S_MLRSInternalData<P_Type>, S_MLRSCacheStorage<P_Type, std::vector>> {
   private:
      // Used by ApplySplineUnified to select the calculation type
      enum N_CalcType { ctValue, ctDeriv };

   public:
      struct S_GradVerifier {
         T_UnifiedMonotonicSpline* p_spline;
         size_t group_id, param_id;
         double in_val;
         bool inverse;

         S_GradVerifier(T_UnifiedMonotonicSpline& spline, size_t group_id_t, size_t param_id_t, double in_val_t, bool inverse_t) : p_spline(&spline), group_id(group_id_t), 
                                                                                                                              param_id(param_id_t), in_val(in_val_t), inverse(inverse_t) {
         }

         double operator()(double param_v) {
            // This utility requires modification of internal parameters, available only in Internal mode.
            if constexpr (P_Mode == smInternal) {
               double saved = p_spline->Container()(N_MLRSParamType(group_id), param_id);
               p_spline->Container()(N_MLRSParamType(group_id), param_id) = param_v;
               p_spline->UpdateDerivedInfo();
               double res = inverse ? p_spline->CalcInv(in_val) : p_spline->Calc(in_val);
               p_spline->Container()(N_MLRSParamType(group_id), param_id) = saved;
               p_spline->UpdateDerivedInfo();
               return res;
            }
            else {
               static_assert(P_Mode == smInternal, "S_GradVerifier is intended for Internal mode (T_LRSplines) as it modifies internal parameters.");
               return 0.;
            }
         }
      };
      using t_base = std::conditional_t<P_Mode == smInternal, S_MLRSInternalData<P_Type>, S_MLRSCacheStorage<P_Type, std::vector>>;

   public:
      using t_val = P_Type;
      using typename t_base::t_param_arr;
      using t_params = S_MLRSUnconstrainedParams<t_param_arr>;

   private:
      static constexpr t_val c_log2 = t_val(0.6931471805599453);
      using t_buffer = std::vector<t_val>;

      bool m_centered = true;
      t_val m_direction_multiplier = t_val(1.);

      template <typename T>
      struct S_BufferAccessor {
         T* const base_ptr;
         const size_t n;

         S_BufferAccessor(T* const b, size_t n_val) : base_ptr(b), n(n_val) {}

         T* x_pos_exp() noexcept { return base_ptr; }
         T* x_neg_exp() noexcept { return base_ptr + 2*n; }
         T* y_pos_exp() noexcept { return base_ptr + 4*n; }
         T* y_neg_exp() noexcept { return base_ptr + 5*n; }
         T* derivs()    noexcept { return base_ptr + 6*n; }
      };

   public:
      T_UnifiedMonotonicSpline(bool centered = true, int direction = 1) {
         Init(centered, direction);
      }
      T_UnifiedMonotonicSpline(const t_val* p_params, size_t n_of_params) {
         if constexpr (P_Mode == smExternal) {
            Init(true, 1);
            UpdateCache(p_params, n_of_params);
         }
         else {
            static_assert(P_Mode == smExternal, "Constructor with (params, size) is intended for External mode (T_LRSplinesInput).");
         }
      }
      const auto& Container() const noexcept { return *(t_base*)(this); }
      bool IsCentered() const noexcept { return m_centered; }
      bool IsDecreasing() const noexcept { return m_direction_multiplier < 0; }

      /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // Unified Initialization
      /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      void Init(bool centered, int direction) {
         m_centered = centered;
         if (direction == 1)
            m_direction_multiplier = t_val(1.);
         else if (direction == -1)
            m_direction_multiplier = t_val(-1.);
         else
            throw LRSplinesException(4, __FILE__, __LINE__, "Direction must be 1 or -1");

         if constexpr (P_Mode == smInternal) {
            if (!this->x_pos.empty())
               UpdateDerivedInfo();
         }
      }

      /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // Mode 1: Loading from file (unified function)
      /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      void TextLoad(const std::string& file_name) {
         if constexpr (P_Mode != smInternal) {
            static_assert(P_Mode == smInternal, "TextLoad is only available in Internal mode.");
            return;
         }
         std::ifstream fin(file_name);
         if (!fin.is_open())
            throw LRSplinesException(3, __FILE__, __LINE__, file_name + ": Cannot open file.");
            
         std::string first_line;
         if (!std::getline(fin, first_line) || first_line.empty())
            throw LRSplinesException(3, __FILE__, __LINE__, file_name + ": File is empty or unreadable.");

         size_t version = 1000;
         if (first_line.find("#VER = ") == 0) {
            version = std::stoul(first_line.substr(7));
            // Version header consumed, read next line
         } else {
            // No version header, put back the line by resetting and re-reading
            fin.clear();
            fin.seekg(0);
         }

         auto read_line = [&](t_param_arr& arr, size_t expected_size = 0, bool is_optional = false) -> bool {
            std::string str;
            if (!std::getline(fin, str)) {
               if (!is_optional)
                  throw LRSplinesException(3, __FILE__, __LINE__, file_name);
               else
                  return false;
            }
            std::istringstream iss(str);
            t_val v;
            arr.resize(0);
            while (iss >> v)
               arr.push_back(v);
            if (arr.empty() || (expected_size && arr.size()!=expected_size)) {
               if (!is_optional)
                  throw LRSplinesException(3, __FILE__, __LINE__, file_name);
               else
                  return false;
            }
            return true;
         };

         read_line(this->x_pos);
         size_t size_2n = this->x_pos.size();
         if (size_2n < 2 || size_2n % 2 != 0)
            throw LRSplinesException(3, __FILE__, __LINE__, file_name + ": x_pos size must be even and >= 2.");
         size_t n = size_2n / 2;

         read_line(this->x_neg, size_2n); //adjusting Internal mode at the load, so that everywhere else we uniformly divide by 2, avoiding extra buffers
         for (size_t i=0; i<size_2n; ++i) {
            this->x_pos[i] += c_log2;
            this->x_neg[i] += c_log2;
         }
         read_line(this->y_pos, n);
         read_line(this->y_neg, n);
         read_line(this->ln_d, size_2n + 1);

         t_param_arr center_param;
         if (read_line(center_param, 2, true)) {
            this->x_0 = center_param[0];
            this->y_0 = center_param[1];
            m_centered = false;
         }
         else {
            m_centered = true;
            this->x_0 = this->y_0 = t_val(0.);
         }
         UpdateDerivedInfo();
      }

      /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // Precomputing the knots for external mode (if the input is not going to be changing for a while)
      /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      void UpdateCache(const t_val* p_params, size_t n_of_params) {
         if constexpr (P_Mode == smExternal)
            std::tie(this->x, this->y, this->w, this->d_left, this->d_right) = ProcessExternalParams(p_params, n_of_params);
         else
            static_assert(P_Mode == smExternal, "UpdateCache(params, ...) is only available in External mode (T_LRSplinesInput).");
      }

      /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // Mode 1: Calculation (using internal cache)
      /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      t_val Calc(t_val x_in) const {
         if (this->x.empty())
            throw LRSplinesException(1, __FILE__, __LINE__, "Spline is not initialized.");
         return ApplySplineUnified(x_in, this->x, this->y, this->w, this->d_left, this->d_right, false, ctValue);
      }

      t_val CalcInv(t_val y_in) const {
         if (this->x.empty())
            throw LRSplinesException(1, __FILE__, __LINE__, "Spline not initialized.");
         return ApplySplineUnified(y_in, this->x, this->y, this->w, this->d_left, this->d_right, true, ctValue);
      }

      t_val CalcDeriv(t_val x_in) const {
         if (this->x.empty())
            throw LRSplinesException(1, __FILE__, __LINE__, "Spline is not initialized.");
         return ApplySplineUnified(x_in, this->x, this->y, this->w, this->d_left, this->d_right, false, ctDeriv);
      }

      t_val CalcInvDeriv(t_val y_in) const {
         if (this->x.empty())
            throw LRSplinesException(1, __FILE__, __LINE__, "Spline not initialized.");
         return ApplySplineUnified(y_in, this->x, this->y, this->w, this->d_left, this->d_right, true, ctDeriv);
      }

      /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // Mode 2: Calculation (On-the-fly)
      /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      t_val Calc(const t_val* p_params, size_t n_of_params, t_val x_in) const {
         if constexpr (P_Mode == smExternal) {
            auto [x, y, w, d_left, d_right] = ProcessExternalParams(p_params, n_of_params);
            return ApplySplineUnified(x_in, x, y, w, d_left, d_right, false, ctValue);
         }
         else {
            static_assert(P_Mode == smExternal, "Calc(params, ...) is only available in External mode.");
            return t_val(0.);
         }
      }

      t_val CalcInv(const t_val* p_params, size_t n_of_params, t_val y_in) const {
         if constexpr (P_Mode == smExternal) {
            auto [x, y, w, d_left, d_right] = ProcessExternalParams(p_params, n_of_params);
            return ApplySplineUnified(y_in, x, y, w, d_left, d_right, true, ctValue);
         }
         else {
            static_assert(P_Mode == smExternal, "CalcInv(params, ...) is only available in External mode.");
            return t_val(0.);
         }
      }

      t_val CalcDeriv(const t_val* p_params, size_t n_of_params, t_val x_in) const {
         if constexpr (P_Mode == smExternal) {
            auto [x, y, w, d_left, d_right] = ProcessExternalParams(p_params, n_of_params);
            return ApplySplineUnified(x_in, x, y, w, d_left, d_right, false, ctDeriv);
         }
         else {
            static_assert(P_Mode == smExternal, "CalcDeriv(params, ...) is only available in External mode.");
            return t_val(0.);
         }
      }

      t_val CalcInvDeriv(const t_val* p_params, size_t n_of_params, t_val y_in) const {
         if constexpr (P_Mode == smExternal) {
            auto [x, y, w, d_left, d_right] = ProcessExternalParams(p_params, n_of_params);
            return ApplySplineUnified(y_in, x, y, w, d_left, d_right, true, ctDeriv);
         }
         else {
            static_assert(P_Mode == smExternal, "CalcInvDeriv(params, ...) is only available in External mode.");
            return t_val(0.);
         }
      }

      ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // Mode 1: Calculates the derivatives of the spline output w.r.t. the unconstrained parameters (Internal Mode)
      ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      t_params CalculateGradients(t_val v_in) const {
         if constexpr (P_Mode == smInternal) {
            if (this->x_pos.empty() || this->x.empty())
               throw LRSplinesException(1, __FILE__, __LINE__, "Spline not initialized or cache outdated.");

            const size_t n = this->N();

            // Call the unified implementation using internal storage
            return CalculateGradientsUnified(
               v_in, n,
               this->x_pos.data(), this->x_neg.data(),
               this->y_pos.data(), this->y_neg.data(),
               this->ln_d.data(),
               this->x, this->y, this->w
            );
         }
         else {
            // In External mode, parameters (P) are not stored internally.
            static_assert(P_Mode == smInternal, "CalculateGradients(v_in) is only available in Internal mode. Use CalculateGradients(params, size, v_in) for External mode.");
            return {};
         }
      }

      ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // Mode 2: Calculates the derivatives of the spline output w.r.t. the unconstrained parameters (External Mode, On-the-fly)
      ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      t_params CalculateGradients(const t_val* p_params, size_t n_of_params, t_val v_in) const {
         if constexpr (P_Mode == smExternal) {
            
            // 1. Process parameters to validate and calculate knots (X, Y, W).
            // 2. Extract N and pointers (Duplicates extraction logic from ProcessExternalParams, as pointers are needed).
            auto [n, p_x_pos, p_x_neg, p_y_pos, p_y_neg, p_ln_d, x_0, y_0] = UnpackParams(p_params, n_of_params);
            auto [x, y, w, d_left, d_right] = CalculateKnots(n, p_x_pos, p_x_neg, p_y_pos, p_y_neg, p_ln_d, x_0, y_0);

            // 3. Calculate Gradients using the unified function.
            return CalculateGradientsUnified(
               v_in, n,
               p_x_pos, p_x_neg,
               p_y_pos, p_y_neg,
               p_ln_d,
               x, y, w
            );
         }
         else {
            static_assert(P_Mode == smExternal, "CalculateGradients(params, ...) is only available in External mode.");
            return {};
         }
      }
      
      ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // Mode 1: Calculates the derivatives of the INVERSE spline output w.r.t. parameters (Internal Mode)
      // Uses Implicit Function Theorem: dx/dP = - (dy/dP) / (dy/dx)
      ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      t_params CalculateInverseGradients(t_val y_in) const {
         if constexpr (P_Mode == smInternal) {
            if (this->x.empty())
               throw LRSplinesException(1, __FILE__, __LINE__, "Spline not initialized.");

            // 1. Calculate x = g(y). We need x to evaluate forward functions at the correct point.
            t_val x_val = this->CalcInv(y_in);

            // 2. Calculate forward gradients: dy/dP evaluated at x.
            t_params grads = this->CalculateGradients(x_val);

            // 3. Calculate forward derivative: dy/dx evaluated at x. We use the forward derivative for robust division and saturation check.
            t_val deriv = this->CalcDeriv(x_val);

            // 4. Apply Implicit Function Theorem and scale robustly.
            ApplyImplicitGradientScale(grads, deriv);
            return grads;
         }
         else {
            static_assert(P_Mode == smInternal, "CalculateInverseGradients(y_in) is only available in Internal mode. Use the version with parameters for External mode.");
            return {};
         }
      }

      ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // Mode 2: Calculates the derivatives of the INVERSE spline output w.r.t. parameters (External Mode, On-the-fly). Optimized to calculate knots only once.
      ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      t_params CalculateInverseGradients(const t_val* p_params, size_t n_of_params, t_val y_in) const {
         if constexpr (P_Mode == smExternal) {
            // 1. Setup: Unpack parameters and calculate knots (X, Y, W) once for efficiency.
            auto [n, p_x_pos, p_x_neg, p_y_pos, p_y_neg, p_ln_d, x_0_val, y_0_val] = UnpackParams(p_params, n_of_params);
            auto [x_knots, y_knots, w_knots, d_left, d_right] = CalculateKnots(n, p_x_pos, p_x_neg, p_y_pos, p_y_neg, p_ln_d, x_0_val, y_0_val);

            // 2. Calculate x = g(y) using the precomputed knots via the unified internal function.
            t_val x_val = ApplySplineUnified(y_in, x_knots, y_knots, w_knots, d_left, d_right, true, ctValue);

            // 3. Calculate forward gradients: dy/dP at x using the unified internal function.
            t_params grads = CalculateGradientsUnified(
                        x_val, n,
                        p_x_pos, p_x_neg,
                        p_y_pos, p_y_neg,
                        p_ln_d,
                        x_knots, y_knots, w_knots
                  );

            // 4. Calculate forward derivative: dy/dx at x. Note: 'inverse' parameter is false here as we want the forward derivative.
            t_val deriv = ApplySplineUnified(x_val, x_knots, y_knots, w_knots, d_left, d_right, false, ctDeriv);
            
            // 5. Apply Implicit Function Theorem and scale robustly.
            ApplyImplicitGradientScale(grads, deriv);
            return grads;
         }
         else {
            static_assert(P_Mode == smExternal, "CalculateInverseGradients(params, ...) is only available in External mode.");
            return {};
         }
      }
      
   private:
      auto& Container() noexcept { return *(t_base*)(this); }
      
      /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////      
      // Helper function to apply scaling based on Implicit Function Theorem: dx/dP = - (dy/dP) / (dy/dx)
      // Calculates factor = -1 / deriv. Handles saturation (deriv ≈ 0) robustly.
      /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      void ApplyImplicitGradientScale(t_params& p, t_val deriv) const {
          // Use the robust epsilon (e.g., 1e-6) to detect saturation.
          const t_val eps = Epsilon<t_val>();
      
          if (std::abs(deriv) < eps) {
              // Saturation region (dy/dx ≈ 0). Inverse gradients approach infinity.
              // We zero out the gradients for numerical stability, a common practice in optimization.
              const size_t n = p.N();
              // Init(n) correctly resizes (if necessary) and resets all elements to zero.
              p.Init(n); 
              return;
          }
      
          // Calculate the scaling factor: -1 / (dy/dx)
          t_val scale = t_val(-1.) / deriv;
      
          // Apply the factor to all gradient components (dy/dP)
          // We iterate manually as S_MLRSUnconstrainedParams does not have a global operator*= in the provided baseline.
          for (auto& v : p.x_pos) v *= scale;
          for (auto& v : p.x_neg) v *= scale;
          for (auto& v : p.y_pos) v *= scale;
          for (auto& v : p.y_neg) v *= scale;
          for (auto& v : p.ln_d) v *= scale;
          
          // x_0 and y_0 must also be scaled if the spline is not centered.
          p.x_0 *= scale;
          p.y_0 *= scale;
      }

      /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // Core Gradient Calculation Logic (Unified, Mode Agnostic)
      /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      t_params CalculateGradientsUnified(
         t_val v_in,
         size_t n,
         const t_val* p_x_pos, const t_val* p_x_neg,
         const t_val* p_y_pos, const t_val* p_y_neg,
         const t_val* p_ln_d,
         const t_param_arr& x, const t_param_arr& y, const t_param_arr& w
      ) const {
         // 1. Setup and Constants
         const size_t n_of_2n = 2 * n;
         const size_t n_of_nodes = 4 * n;
         const t_val eps = Epsilon<t_val>();

         // Helper for safe clamping (consistent with forward pass clamps)
         auto safe = [&](t_val z) { return std::max(z, eps); };

         t_params grads(n);

         // 2. Recalculate Intermediate Exponentials (p_k, h_i, D_t)
         t_buffer buffer;
         buffer.resize(8*n + 1);
         S_BufferAccessor<t_val> acc(buffer.data(), n);

         // Compute exponentials. Must mirror CalculateKnots logic: always subtract c_log2.
         // This correctly handles both Internal parameters (where c_log2 was added during load) and External parameters.
         for (size_t k = 0; k < n_of_2n; ++k) {
            acc.x_pos_exp()[k] = std::exp(p_x_pos[k] - c_log2); // p_k+
            acc.x_neg_exp()[k] = std::exp(p_x_neg[k] - c_log2); // p_k-
         }
         for (size_t i = 0; i < n; ++i) {
            acc.y_pos_exp()[i] = std::exp(p_y_pos[i]); // h_i+
            acc.y_neg_exp()[i] = std::exp(p_y_neg[i]); // h_i-
         }
         for (size_t t = 0; t <= n_of_2n; ++t)
            acc.derivs()[t] = std::exp(p_ln_d[t]); // D_t

         // Handle direction multiplier. If direction < 0, g_out(v) = g_spline(-v)
         t_val v_eval = v_in;
         if (m_direction_multiplier < 0)
            v_eval = -v_in;
         // NOTE: No global grad_scale is needed for parameter derivatives in forward mode

         // 3. Find Active Segment and Calculate Derivatives
         if (v_eval <= x[0]) {
            // 3.1. Left Tail
            const t_val d_left = acc.derivs()[0];
            if (!m_centered)
               grads.y_0 = 1., grads.x_0 = -d_left;
            grads.ln_d[0] = d_left * (v_eval - x[0]);
            for (size_t k = 0; k < n_of_2n; ++k)
               grads.x_neg[k] = d_left * acc.x_neg_exp()[k];
            for (size_t i = 0; i < n; ++i)
               grads.y_neg[i] = -acc.y_neg_exp()[i];

         }
         else if (v_eval >= x[n_of_nodes]) {
            // 3.2. Right Tail
            const t_val d_right = acc.derivs()[n_of_2n];
            if (!m_centered)
               grads.y_0 = 1., grads.x_0 = -d_right;
            grads.ln_d[n_of_2n] = d_right * (v_eval - x[n_of_nodes]);
            for (size_t k = 0; k < n_of_2n; ++k)
               grads.x_pos[k] = -d_right * acc.x_pos_exp()[k];
            for (size_t i = 0; i < n; ++i)
               grads.y_pos[i] = acc.y_pos_exp()[i];

         }
         else {
            // 3.3. Interior
            auto upper_it = std::upper_bound(x.begin(), x.end(), v_eval);
            size_t j = std::distance(x.begin(), upper_it);

            // Calculate g(v) and Local Adjoints
            const t_val a = w[j-1] * (x[j] - v_eval);
            const t_val b = w[j] * (v_eval - x[j-1]);
            // Ensure s matches the forward pass clamp (ApplySplineUnified)
            const t_val s = safe(a + b);
            // Optimization: Reuse inverse
            const t_val inv_s = t_val(1.) / s;
            const t_val g = (y[j-1] * a + y[j] * b) * inv_s;

            const t_val adj_y_jm1 = a * inv_s;
            const t_val adj_y_j = b * inv_s;
            const t_val adj_x_jm1 = -w[j] * (y[j] - g) * inv_s;
            const t_val adj_x_j = w[j-1] * (y[j-1] - g) * inv_s;
            const t_val adj_w_jm1 = (y[j-1] - g) * (x[j] - v_eval) * inv_s;
            const t_val adj_w_j = (y[j] - g) * (v_eval - x[j-1]) * inv_s;

            // Center Coordinates
            if (!m_centered) {
               grads.x_0 = adj_x_jm1 + adj_x_j;
               grads.y_0 = (a + b) * inv_s;
            }

            // Helper lambda to apply the chain rule for a node s (j-1 or j)
            auto apply_chain_rule_for_node = [&](size_t s, t_val adj_x_s, t_val adj_y_s, t_val adj_w_s) {
               const bool is_even = (s % 2 == 0);

               // 1. Contribution via X-grid (T_X) - Optimized Tight Loops
               // Positive side: k <= s - (2n + 1). Requires s > 2n
               if (s > n_of_2n) {
                  // kmax is the largest index k. Since s <= 4n, kmax <= 2n-1
                  size_t kmax = s - (n_of_2n + 1);
                  for (size_t k = 0; k <= kmax; ++k)
                     grads.x_pos[k] += adj_x_s * acc.x_pos_exp()[k];
               }
               // Negative side: k >= s. Requires s < 2n
               if (s < n_of_2n) {
                  for (size_t k = s; k < n_of_2n; ++k)
                     grads.x_neg[k] -= adj_x_s * acc.x_neg_exp()[k];
               }

               // 2. Contribution via W-grid and Y-grid (Even vs Odd)
               if (is_even) {
                  // EVEN NODE (Knot)
                  size_t t = s / 2;
                  // W-grid (T_W,even)
                  grads.ln_d[t] += adj_w_s * (t_val(-0.5) * w[s]);

                  // Y-grid (T_Y, even) - Prefix sums
                  for (size_t q = 0; q < n; ++q) {
                     const size_t ci_plus = n_of_2n + 2*(q+1);
                     const size_t ci_minus = 2*q;
                     if (s >= ci_plus)  grads.y_pos[q] += adj_y_s * acc.y_pos_exp()[q];
                     if (s <= ci_minus) grads.y_neg[q] -= adj_y_s * acc.y_neg_exp()[q];
                  }
               }
               else {
                  // ODD NODE (Midpoint)
                  // Determine side (sigma) and bin index (i), derivative index (t)
                  bool is_pos = (s > n_of_2n);
                  size_t i = is_pos ? (s - n_of_2n - 1) / 2 : (s - 1) / 2;
                  size_t t = is_pos ? n + i : i;

                  const t_val* p_p_exp = is_pos ? acc.x_pos_exp() : acc.x_neg_exp();
                  const t_val* p_h_exp = is_pos ? acc.y_pos_exp() : acc.y_neg_exp();
                  t_val* p_grad_x = is_pos ? grads.x_pos.data() : grads.x_neg.data();
                  t_val* p_grad_y = is_pos ? grads.y_pos.data() : grads.y_neg.data();

                  // Local variables for the bin (ensure clamps match forward pass)
                  const t_val p0 = p_p_exp[2*i], p1 = p_p_exp[2*i+1];
                  const t_val delta_x = safe(p0 + p1);
                  const t_val delta_y = safe(p_h_exp[i]);
                  const t_val lambda = p0 / delta_x;

                  const t_val alpha = std::sqrt(acc.derivs()[t]);
                  const t_val beta = std::sqrt(acc.derivs()[t+1]);

                  // Weights at surrounding knots and interpolation factor A
                  const t_val a_w = w[s-1], b_w = w[s+1];
                  const t_val a_factor = safe((t_val(1.) - lambda) * a_w + lambda * b_w);
                  // Optimization: Calculate inv_a once
                  const t_val inv_a = t_val(1.) / a_factor;

                  // Optimization: Precompute interpolation weights and dYs/dlambda
                  const t_val ws_l = (t_val(1.)-lambda) * a_w * inv_a;
                  const t_val ws_r = lambda * b_w * inv_a;
                  const t_val dys_dlambda = (a_w * b_w) * (y[s+1] - y[s-1]) * (inv_a * inv_a);

                  // --- Contribution via W_s (T_W and T_Mid via W) ---
                  // W_s w.r.t. m
                  p_grad_y[i] += adj_w_s * (-w[s]);
                  // W_s w.r.t. l (Simplified form)
                  p_grad_x[2*i] += adj_w_s * (alpha * p0 / delta_y);
                  p_grad_x[2*i+1] += adj_w_s * (beta * p1 / delta_y);
                  // W_s w.r.t. r
                  grads.ln_d[t] += adj_w_s * t_val(0.5) * (delta_x / delta_y) * lambda * alpha;
                  grads.ln_d[t+1] += adj_w_s * t_val(0.5) * (delta_x / delta_y) * (t_val(1.)-lambda) * beta;

                  // --- Contribution via Y_s (T_Mid via Y) ---
                  // Y_s w.r.t. l
                  const t_val dlam = dys_dlambda * lambda * (t_val(1.)-lambda);
                  p_grad_x[2*i] += adj_y_s * dlam;
                  p_grad_x[2*i+1] -= adj_y_s * dlam;

                  // Y_s w.r.t. r
                  grads.ln_d[t] += adj_y_s * (-t_val(0.5) * a_w * (t_val(1.)-lambda) * (y[s-1] - y[s]) * inv_a);
                  grads.ln_d[t+1] += adj_y_s * (-t_val(0.5) * b_w * lambda * (y[s+1] - y[s]) * inv_a);

                  // --- Contribution via Y-grid (T_Y, odd) - Prefix sums using interpolation weights
                  for (size_t q = 0; q < n; ++q) {
                     // Positive side
                     const size_t ci_plus = n_of_2n + 2*(q+1);
                     t_val cplus = 0;
                     // s-1 >= 0 and s+1 <= M are guaranteed by the definition of the interior segment
                     if (s-1 >= ci_plus) cplus += ws_l * acc.y_pos_exp()[q];
                     if (s+1 >= ci_plus) cplus += ws_r * acc.y_pos_exp()[q];
                     grads.y_pos[q] += adj_y_s * cplus;

                     // Negative side
                     const size_t ci_minus = 2*q;
                     t_val cminus = 0;
                     if (s-1 <= ci_minus) cminus -= ws_l * acc.y_neg_exp()[q];
                     if (s+1 <= ci_minus) cminus -= ws_r * acc.y_neg_exp()[q];
                     grads.y_neg[q] += adj_y_s * cminus;
                  }
               }
            };

            // Apply the chain rule for both active nodes j-1 and j
            apply_chain_rule_for_node(j-1, adj_x_jm1, adj_y_jm1, adj_w_jm1);
            apply_chain_rule_for_node(j, adj_x_j, adj_y_j, adj_w_j);
         }
         return grads;
      }

      /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // Recalculating knots from stored internal parameters
      /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      void UpdateDerivedInfo() {
         if constexpr (P_Mode == smInternal) {
            if (this->x_pos.empty())
               throw LRSplinesException(1, __FILE__, __LINE__, "Internal parameters not loaded or incomplete.");

            std::tie(this->x, this->y, this->w, this->d_left, this->d_right) = CalculateKnots(
               this->y_pos.size(),
               this->x_pos.data(), this->x_neg.data(),
               this->y_pos.data(), this->y_neg.data(),
               this->ln_d.data(),
               this->x_0, this->y_0
            );
         }
      }

      /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // Just renaming pointers, really
      /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      std::tuple<size_t, const t_val*, const t_val*, const t_val*, const t_val*, const t_val*, t_val, t_val> UnpackParams(const t_val* p_params, size_t n_of_params) const {
         size_t w_offset = m_centered ? 1 : 3;
         if (n_of_params % 2 != 1 || n_of_params < 8 + w_offset || (n_of_params - w_offset) % 8 != 0) {
            std::string expected = m_centered ? "8*N+1" : "8*N+3";
            throw LRSplinesException(2, __FILE__, __LINE__, "Incorrect number of parameters. Expected " + expected + " (N>=1).");
         }

         size_t n = (n_of_params - w_offset) / 8;
         const t_val* p_x_pos = p_params;
         const t_val* p_x_neg = &p_params[2*n];
         const t_val* p_y_pos = &p_params[4*n];
         const t_val* p_y_neg = &p_params[5*n];
         const t_val* p_ln_d = &p_params[6*n];
         t_val x_0 = m_centered ? t_val(0.) : p_params[8*n + 1];
         t_val y_0 = m_centered ? t_val(0.) : p_params[8*n + 2];

         return std::tuple(n, p_x_pos, p_x_neg, p_y_pos, p_y_neg, p_ln_d, x_0, y_0);
      }

      /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // Mode 2: External Parameter Processing
      /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      std::tuple<t_param_arr, t_param_arr, t_param_arr, t_val, t_val> ProcessExternalParams(const t_val* p_params, size_t n_of_params) const {
         auto [n, p_x_pos, p_x_neg, p_y_pos, p_y_neg, p_ln_d, x_0, y_0] = UnpackParams(p_params, n_of_params);
         
         return CalculateKnots(
            n,
            p_x_pos, p_x_neg,
            p_y_pos, p_y_neg,
            p_ln_d,
            x_0, y_0
         );
      }

      /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // Core Knot Calculation
      /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      std::tuple<t_param_arr, t_param_arr, t_param_arr, t_val, t_val> CalculateKnots(
         size_t n,
         const t_val* p_x_pos, const t_val* p_x_neg,
         const t_val* p_y_pos, const t_val* p_y_neg,
         const t_val* p_ln_d,
         t_val x_0, t_val y_0) const {
         const size_t size_2n = 2 * n;
         const size_t total_nodes = 4 * n + 1;
         const t_val eps = Epsilon<t_val>();

         t_param_arr x(total_nodes), y(total_nodes), w(total_nodes);

         t_buffer buffer;
         buffer.resize(8*n + 1);
         S_BufferAccessor<t_val> acc(buffer.data(), n);

         for (size_t i = 0; i < size_2n; ++i) {
            acc.x_pos_exp()[i] = std::exp(p_x_pos[i] - c_log2); //always subtracting c_log2 here, it is added in the Internal mode at the load
            acc.x_neg_exp()[i] = std::exp(p_x_neg[i] - c_log2);
         }
         for (size_t i = 0; i < n; ++i) {
            acc.y_pos_exp()[i] = std::exp(p_y_pos[i]);
            acc.y_neg_exp()[i] = std::exp(p_y_neg[i]);
         }
         for (size_t i = 0; i <= size_2n; ++i) {
            acc.derivs()[i] = std::exp(p_ln_d[i]);
            w[2*i] = t_val(1) / std::sqrt(acc.derivs()[i]);
         }

         t_val sum = 0;
         for (size_t i = 0; i < size_2n; ++i) {
            sum += acc.x_pos_exp()[i];
            x[size_2n + 1 + i] = sum;
         }
         sum = 0;
         for (size_t i = 0; i < n; ++i) {
            sum += acc.y_pos_exp()[i];
            y[2*(i + n + 1)] = sum;
         }
         sum = 0;
         for (size_t i = 0; i < size_2n; ++i) {
            sum += acc.x_neg_exp()[size_2n - 1 - i];
            x[size_2n - 1 - i] = -sum;
         }
         sum = 0;
         for (size_t i = 0; i < n; ++i) {
            sum += acc.y_neg_exp()[n - 1 - i];
            y[2*(n - 1 - i)] = -sum;
         }

         y[2*n] = t_val(0.);

         for (size_t i = 0; i < n; ++i) {
            t_val dx1_pos = acc.x_pos_exp()[2*i];
            t_val lambda_pos = dx1_pos / std::max(dx1_pos + acc.x_pos_exp()[2*i+1], eps);

            t_val dx1_neg = acc.x_neg_exp()[2*i];
            t_val lambda_neg = dx1_neg / std::max(dx1_neg + acc.x_neg_exp()[2*i+1], eps);

            size_t idx_pos = n + i;
            t_val w_pos = w[2*idx_pos];
            t_val w_pos_next = w[2*(idx_pos+1)];
            w[2*(n+i) + 1] = (lambda_pos * w_pos * acc.derivs()[idx_pos] + (1 - lambda_pos) * w_pos_next * acc.derivs()[idx_pos+1]) *
               (dx1_pos + acc.x_pos_exp()[2*i+1]) / std::max(acc.y_pos_exp()[i], eps);

            t_val w_neg = w[2*i];
            t_val w_neg_next = w[2*(i+1)];
            w[2*i + 1] = (lambda_neg * w_neg * acc.derivs()[i] + (1 - lambda_neg) * w_neg_next * acc.derivs()[i+1]) *
               (dx1_neg + acc.x_neg_exp()[2*i+1]) / std::max(acc.y_neg_exp()[i], eps);

            y[size_2n + 1 + 2*i] = ((1 - lambda_pos) * w_pos * y[2*(n+i)] + lambda_pos * w_pos_next * y[2*(n+i+1)]) /
               std::max((1 - lambda_pos) * w_pos + lambda_pos * w_pos_next, eps);

            y[2*i + 1] = ((1 - lambda_neg) * w_neg * y[2*i] + lambda_neg * w_neg_next * y[2*(i+1)]) /
               std::max((1 - lambda_neg) * w_neg + lambda_neg * w_neg_next, eps);
         }

         x[size_2n] = t_val(0.);

         if (x_0 != t_val(0.))
            for (size_t i = 0; i < total_nodes; ++i)
               x[i] += x_0;
         if (y_0 != t_val(0.))
            for (size_t i = 0; i < total_nodes; ++i)
               y[i] += y_0;

         return std::make_tuple(x, y, w, acc.derivs()[0], acc.derivs()[size_2n]);
      }

      /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // Core Spline Application Logic (Unified for Value and Derivative)
      /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      t_val ApplySplineUnified(t_val v_in, const t_param_arr& x, const t_param_arr& y, const t_param_arr& w, t_val d_left, t_val d_right, bool inverse, N_CalcType calc_type) const {
         const size_t n_of_nodes = x.size();
         if (n_of_nodes == 0)
            throw LRSplinesException(2, __FILE__, __LINE__, "Spline knots are empty or not initialized.");

         const t_val eps = Epsilon<t_val>();

         // 1. Input transformation (for forward mode with negative direction: g_out(x) = g_spline(-x))
         if (!inverse && m_direction_multiplier < 0)
            v_in = -v_in;

         const t_param_arr& search = inverse ? y : x;
         const t_param_arr& output = inverse ? x : y;

         // 2. Finalization (handling direction multiplier for output)
         auto finalize = [&](t_val res) {
               if (calc_type == ctDeriv) {
                  // Derivative: Chain rule applies multiplication by m_direction_multiplier (1 or -1).
                  // Forward: F'(x) = G'(-x)*(-1). Inverse: H'(y) = -G_inv'(y).
                  return res * m_direction_multiplier;
               }
               // ctValue: Matches original ApplySpline logic.
            return inverse && m_direction_multiplier < 0 ? -res : res;
         };

         // 3. Boundary conditions (tails)
         if (v_in <= search[0]) {
               // Slope (of the internal increasing spline)
               t_val factor = inverse ? t_val(1.)/std::max(d_left, eps) : d_left;
               if (calc_type == ctDeriv)
                  return finalize(factor);
               // ctValue
            return finalize(output[0] + (v_in - search[0]) * factor);
         }

         if (v_in >= search.back()) {
               // Slope (of the internal increasing spline)
               t_val factor = inverse ? t_val(1.)/std::max(d_right, eps) : d_right;
               if (calc_type == ctDeriv)
                  return finalize(factor);
               // ctValue
            return finalize(output.back() + (v_in - search.back()) * factor);
         }

         // 4. Interior calculation
         auto it = std::upper_bound(search.begin(), search.end(), v_in);
         size_t idx = std::distance(search.begin(), it);

         if (idx == 0 || idx == n_of_nodes)
            throw LRSplinesException(1, __FILE__, __LINE__, "Invariant broken: upper_bound outside expected range.");

         // Knot values (k = idx-1, k+1 = idx)
         t_val w_k = w[idx - 1], w_k1 = w[idx];
         t_val x_k = x[idx - 1], x_k1 = x[idx];
         t_val y_k = y[idx - 1], y_k1 = y[idx];

         t_val v1, v2;

         // Calculate interpolation weights (v1, v2) which form the denominator.
         // The weights used depend on whether we are in forward or inverse mode, matching the original implementation.
         if (inverse) {
               // Inverse: D_inv(y) = w_{k+1}*(y_{k+1}-y) + w_k*(y-y_k)
            v1 = w_k1 * (y_k1 - v_in);
            v2 = w_k * (v_in - y_k);
         } else {
               // Forward: D(x) = w_k*(x_{k+1}-x) + w_{k+1}*(x-x_k)
            v1 = w_k * (x_k1 - v_in);
            v2 = w_k1 * (v_in - x_k);
         }

         t_val denominator = std::max(v1 + v2, eps);

         if (calc_type == ctDeriv) {
               // Derivative formula (symmetric for forward/inverse):
               t_val numerator = w_k * w_k1 * (y_k1 - y_k) * (x_k1 - x_k);
               return finalize(numerator / (denominator * denominator));
         }

         // ctValue
         if (inverse) {
               // Inverse: x = (x_k*v1 + x_{k+1}*v2) / D
            return finalize((x_k * v1 + x_k1 * v2) / denominator);
         } else {
               // Forward: y = (y_k*v1 + y_{k+1}*v2) / D
            return finalize((y_k * v1 + y_k1 * v2) / denominator);
         }
      }
   };

   //#########################################################################################################################################################################################

   template<class T>
   using T_LRSplines = T_UnifiedMonotonicSpline<T, smInternal>;

   template<class T>
   using T_LRSplinesInput = T_UnifiedMonotonicSpline<T, smExternal>;
}

#endif // LR_SPLINES_H