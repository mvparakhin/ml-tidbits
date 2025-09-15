# ml-tidbits

Small, focused ML utilities in **Python/PyTorch** and **C++** that emphasize **clarity**, **latency**, and **practicality**.
Each component is self-contained with docs, tests, and (when relevant) a C++ counterpart.
C++ components are all tested on Windows and Mac, gcc, clang and Visual Studio 2022. Current requirements are C++17, I will later switch to C++20.

> First addition: **UnifiedMonotonicSpline** — a unified, batched **monotone rational linear spline** layer with forward/inverse modes and increasing/decreasing directions.

**Why it’s cool:** `UnifiedMonotonicSpline` is an **invertible, batched monotonic rational linear spline** module available in both **PyTorch** and **modern C++**, using one **unconstrained parameterization** (`8N+1` or `8N+3` if uncentered) shared across languages. It guarantees monotonicity by construction (exponential spacing/derivative params), provides stable **forward and inverse** transforms with linear tails, and exposes **analytic derivatives** (and inverse‑gradients via the implicit function theorem) for seamless training. You can run it with **internal weights** (single spline applied to many values) or **external weights** (per‑example splines in a batch), making it ideal for **normalizing flows**, **calibration layers**, **monotone neural networks**, tabular feature transforms, and differentiable bijective scalers. The C++17 implementation mirrors the PyTorch API, is tested on Windows/macOS with MSVC/Clang/GCC, and the on‑the‑fly knot computation uses vectorized barycentric rational interpolation with binary search over the `4N+1` knots for fast, low‑latency inference.
