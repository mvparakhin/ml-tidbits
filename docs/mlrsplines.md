This is a very common frustration when writing detailed technical documentation on GitHub\! You have an excellent and comprehensive `help.md` file.

The rendering issues you are seeing, particularly incorrect spacing in complex formulas, are usually caused by conflicts between GitHub Flavored Markdown (GFM) and LaTeX syntax.

### The Problem: Markdown Interference

While GitHub supports LaTeX rendering (using MathJax), its Markdown processor can sometimes interfere with the math content.

The most common conflict involves the underscore `_`. In LaTeX, `_` is used for subscripts (e.g., $X\_s$). In Markdown, `_text_` means *italics*.

When you use the `$$ ... $$` delimiters for block equations, GitHub sometimes attempts to apply Markdown formatting *before* rendering the LaTeX. If it sees a complex equation with many subscripts, it might misinterpret the underscores, corrupting the LaTeX code and leading to broken rendering or incorrect spacing.

### The Solution: Use Fenced Math Blocks

The most robust way to ensure your formulas render correctly on GitHub is to use a **fenced code block with the language identifier `math`**.

````markdown
```math
[Your LaTeX equation here]
```
````

This tells the GitHub Markdown processor: "Do not apply standard Markdown formatting inside this block; treat it purely as math code."

### Additional Fixes Applied

I have gone through your `help.md` file and applied the following fixes:

1.  **Replaced all `$$` blocks with ` ```math ` blocks.**
2.  **Optimized Alignment:** For the multi-column equations in Equation (5), I used the `alignat` environment instead of `aligned`. This provides better control over the spacing between columns on GitHub.
3.  **Inline Math Consistency:** I ensured all mathematical variables and symbols within your text paragraphs (e.g., $W$, $8N+1$, $\\log(2)$) are wrapped in single dollar signs (`$ ... $`) for consistent formatting. I also changed `eps` to the symbol $\\varepsilon$.

### The Corrected Markdown File

You can copy the content below and paste it into your `help.md` file on GitHub.

```markdown
# Monotonic Linear Rational Splines (MLRS)

*A lightweight Python + C++ library for fast, invertible, monotone **linear rational splines** with closed‑form inverse, linear tails, centered or non‑centered parameterizations, and full analytical derivatives.*

> **Credits.** This work is informed by the linear rational spline (LRS) flow introduced by **Dolatabadi, Erfani, and Leckie (AISTATS 2020)**. If you use this library, please cite their paper: *Invertible Generative Modeling using Linear Rational Splines* (PMLR v108). ([Proceedings of Machine Learning Research][1])
> The construction of monotone LRS interpolants with an interior point per bin follows classic ideas from **Fuhr & Kallay (1992)**. ([ScienceDirect][2], [ACM Digital Library][3])

---

## Table of contents

1. [What is in the box](#what-is-in-the-box)

2. [Installation & repository layout](#installation--repository-layout)

3. [Quick start: Python](#quick-start-python)

4. [Quick start: C++](#quick-start-c)

5. [Mathematical overview](#mathematical-overview)

   * [5.1 Parameterization](#51-parameterization)
   * [5.2 Knots and midpoints](#52-knots-and-midpoints)
   * [5.3 Forward / inverse evaluation](#53-forward--inverse-evaluation)
   * [5.4 Linear tails](#54-linear-tails)

6. \[Analytical derivatives (full)]

   * [6.1 Local adjoints](#61-local-adjoints)
   * [6.2 Node Jacobians](#62-node-jacobians)
   * [6.3 Assembled derivatives (interior)](#63-assembled-derivatives-interior)
   * [6.4 Assembled derivatives (tails)](#64-assembled-derivatives-tails)
   * [6.5 Inverse‑mode parameter derivatives](#65-inversemode-parameter-derivatives)

7. [Python API](#python-api)

8. [C++ API](#c-api)

9. [File format & cross‑language interoperability](#file-format--crosslanguage-interoperability)

10. [Numerical stability & notes](#numerical-stability--notes)

11. [Examples & tests](#examples--tests)

12. [FAQ](#faq)

13. [References](#references)

---

## What is in the box

* **Elementwise monotone transforms** on $\mathbb{R}$ using *linear rational splines* (homographic segments) with a **single interior point per bin**.
* **Closed‑form inverse** with the same homographic form: forward and inverse cost are symmetric.
* **Two operation modes**

  * **Mode 1 (internal parameters):** one spline object with trainable parameters.
  * **Mode 2 (external weights):** pass a different parameter vector per sample (batched).
* **Centered vs non‑centered:** either force the spline to pass through $(0,0)$ or learn $(x_0,y_0)$.
* **Increasing / decreasing** directions with a single flag (decreasing is just an odd reflection wrapper).
* **Linear tails** outside the interior, set by the end‑knot derivatives.
* **Full analytical derivatives** of the forward mapping w\.r.t. unconstrained parameters (C++), plus **inverse‑mode** gradients via the implicit function theorem.
* **Reference implementations**

  * `python/mlrsplines/LRSplines.py` — PyTorch module (`UnifiedMonotonicSpline`)
  * `cpp/mlrsplines/LRSplines.h` — single‑header C++ (header‑only)
* **Test suites** for Python and C++ that check value/derivative consistency and inverse identities.

---

## Installation & repository layout

```

cpp/                    \# C++ header and tests
mlrsplines/
LRSplines.h         \# header-only implementation
tests/
TestLRSplines.cpp   \# C++ tests (expect data/\*.txt)
data/               \# example parameter files
docs/
mlrsplines.md         \# THIS document
python/
mlrsplines/
LRSplines.py        \# PyTorch module
tests/
TestLRSplines.py    \# Python tests

````

**Build C++ tests (example):**

```bash
cd cpp/tests
c++ -O3 -std=c++17 -I../mlrsplines TestLRSplines.cpp -o TestLRSplines
./TestLRSplines
````

**Python tests (example):**

```bash
cd python/tests
python TestLRSplines.py
```

-----

## Quick start (Python)

```python
import torch
from mlrsplines.LRSplines import UnifiedMonotonicSpline, ExtractParamsForExternal

device = torch.device("cpu")

# Mode 1: single trainable spline (increasing, centered at (0,0))
spline = UnifiedMonotonicSpline(n_of_nodes=5, inverse=False,
                                direction='increasing', centered=True).to(device)

# Fit to a monotone target, e.g., tanh(2x)
x = torch.linspace(-3, 3, 256).unsqueeze(-1)
y = torch.tanh(2*x)

opt = torch.optim.Adam(spline.parameters(), lr=1e-2)
for _ in range(1000):
    opt.zero_grad()
    pred = spline(x)
    loss = torch.nn.functional.mse_loss(pred, y)
    loss.backward()
    opt.step()

# Save parameters (Mode 1 format)
spline.SaveSplineWeights("spline_tanh.txt")

# Switch to Mode 2: same spline as external weights
weights = ExtractParamsForExternal(spline)          # shape = 8N+1 (centered) or 8N+3 (non-centered)
x_batch = torch.linspace(-2, 2, 10).unsqueeze(-1)
spline2 = UnifiedMonotonicSpline(n_of_nodes=None, inverse=False,
                                 direction=1, centered=True)
y2 = spline2(x_batch, weights.expand(x_batch.shape[0], -1))
```

-----

## Quick start (C++)

```cpp
#include "mlrsplines/LRSplines.h"
using t_spline = ns_base::T_LRSplines<double>;       // Mode 1: internal
using t_input  = ns_base::T_LRSplinesInput<double>;  // Mode 2: external

// Mode 1: load parameters saved by Python (text format)
t_spline s(true, +1);       // centered=true, increasing
s.TextLoad("cpp/tests/data/spline_tanh.txt");

double y  = s.Calc(0.5);
double dy = s.CalcDeriv(0.5);
double x  = s.CalcInv(y);

// Mode 2: evaluate from a raw parameter buffer (8N+1 or 8N+3)
std::vector<double> params = /* ... load ... */;
t_input fly(true, +1);
double y2  = fly.Calc(params.data(), params.size(), 0.5);
double dy2 = fly.CalcDeriv(params.data(), params.size(), 0.5);
```

-----

## Mathematical overview

### 5.1 Parameterization

We construct a symmetric spline around a center (by default the origin) using **$n$ bins per side**.

  * **Unconstrained parameters per side**

      * $2n$ *x‑spacings*: $\\boldsymbol{\\ell}^{\\pm}\\in\\mathbb{R}^{2n}$
        (two positive spacings per bin; exponentiated to guarantee positivity).
      * $n$ *y‑heights*: $\\mathbf{m}^{\\pm}\\in\\mathbb{R}^{n}$
        (one positive increment per bin; exponentiated).

  * **Derivatives at base knots**: $\\mathbf{r}\\in\\mathbb{R}^{2n+1}$
    with $D\_t=\\exp r\_t\>0$.

  * **Optional center offsets** (non‑centered only): $(x\_0,y\_0)\\in\\mathbb{R}^2$.

We define

```math
p_k^{\pm}=\exp\!\left(\ell_k^{\pm}-\log 2\right),\qquad
h_i^{\pm}=\exp(m_i^{\pm}),\qquad
D_t=\exp(r_t),\qquad
W_{2t}=\frac{1}{\sqrt{D_t}}.
```

> **Why $-\\log 2$?** It aligns Python Mode‑1 and Mode‑2 parameter initializations so that “internal” and “external” paths produce identical knots and numerics (see §9). The goal is to have natural scale when input is a vector of elements close to zero - regular situation with Neural Network initialization.

### 5.2 Knots and midpoints

There are **$2n+1$ base knots** (even indices $s=0,2,\\dots,4n$), plus **one interior point per bin** (odd indices $s=1,3,\\dots,4n-1$), for a total of $4n+1$ nodes.

  * **X grid.** Cumulative sums of $p\_k^\\pm$ produce strictly increasing $X\_s$.
    Negative side grows to the left, positive to the right; the center is $s=2n$.
  * **Y grid.** Cumulative sums of $h\_i^\\pm$ create monotone $Y\_s$ with $Y\_{2n}=0$.
  * **Midpoints in each bin** (odd $s$) use a *location* $\\lambda\\in(0,1)$ **derived from the two x‑spacings of that bin**:

<!-- end list -->

```math
\lambda=\frac{p_0}{p_0+p_1}.
```

  * **Midpoint weights** $W\_s$ enforce $C^1$ matching of the homographic pieces (see Eq. below).

Let the even indices correspond to base knots; then (schematically)

```math
\underbrace{\cdots\,Y_0}_{s=0}
\;-\;
\underbrace{Y_1}_{\text{mid}}
\;-\;
\underbrace{Y_2}_{\text{base}}
\;-\; \cdots \;-\;
\underbrace{Y_{2n}}_{\text{center}}
\;-\; \cdots \;-\;
\underbrace{Y_{4n}}_{\text{right end}}.
```

### 5.3 Forward / inverse evaluation

On a single interior segment $[X\_{j-1},X\_j]$ we use a **linear rational** (homographic) form. Writing

```math
a=W_{j-1}(X_j-v),\quad b=W_j(v-X_{j-1}),\quad S=a+b,
```

the output is the **barycentric** interpolation

```math
g(v)=\frac{Y_{j-1}a+Y_j b}{S}.
\tag{F}
```

This is precisely the **two‑weight** form used in LRS flows and ensures a **closed‑form inverse** by swapping the roles of $(X,W)$ and $(Y,W)$ (see §A.2 below and Dolatabadi et al. for flow context). ([Proceedings of Machine Learning Research][1])

### 5.4 Linear tails

Outside $[X\_0,X\_{4n}]$ we attach linear tails with slopes $D\_0$ and $D\_{2n}$. This covers $\\mathbb{R}$ without remapping. (An alternative in the literature is to map $\\mathbb{R}$ to $[0,1]$ first; we adopt linear tails as common in normalizing‑flow practice.)

-----

## Analytical derivatives (full)

This section reproduces the complete, verified analytical derivatives of the **forward** map $g(v)$ with respect to all unconstrained parameters, matching the implementation in `cpp/mlrsplines/LRSplines.h`.

We first restate the **two‑piece form** within a bin with an interior point (following Fuhr & Kallay; see also Dolatabadi et al. for flows). Let $\\phi=\\frac{v-X^{(k)}}{X^{(k+1)}-X^{(k)}}\\in[0,1]$, mid‑location $\\lambda\\in(0,1)$, and positive weights $w^{(k)},w^{(m)},w^{(k+1)}$. Then

```math
g(\phi)=
\begin{cases}
\dfrac{w^{(k)}Y^{(k)}(\lambda-\phi)+w^{(m)}Y^{(m)}\phi}{w^{(k)}(\lambda-\phi)+w^{(m)}\phi},
& 0\le \phi \le \lambda,\\[8pt]
\dfrac{w^{(m)}Y^{(m)}(1-\phi)+w^{(k+1)}Y^{(k+1)}(\phi-\lambda)}{w^{(m)}(1-\phi)+w^{(k+1)}(\phi-\lambda)},
& \lambda \le \phi \le 1.
\end{cases}
\tag{1}
```

The **derivative in $\\phi$** is

```math
\frac{d g(\phi)}{d\phi}=
\begin{cases}
\dfrac{\lambda\,w^{(k)}w^{(m)}\bigl(Y^{(m)}-Y^{(k)}\bigr)}{\bigl(w^{(k)}(\lambda-\phi)+w^{(m)}\phi\bigr)^2},
& 0\le \phi \le \lambda,\\[10pt]
\dfrac{(1-\lambda)\,w^{(m)}w^{(k+1)}\bigl(Y^{(k+1)}-Y^{(m)}\bigr)}{\bigl(w^{(m)}(1-\phi)+w^{(k+1)}(\phi-\lambda)\bigr)^2},
& \lambda \le \phi \le 1,
\end{cases}
\tag{2}
```

and $\\tfrac{d g}{d v} = \\tfrac{1}{\\delta^{(k)}},\\tfrac{d g}{d\\phi}$ with $\\delta^{(k)}=X^{(k+1)}-X^{(k)}$.
The **inverse** on the same bin has the same form with roles of $(X, W)$ and $(Y, W)$ swapped:

```math
g^{-1}(y)=
\begin{cases}
\dfrac{\lambda\,w^{(k)}\bigl(Y^{(k)}-y\bigr)}{w^{(k)}\bigl(Y^{(k)}-y\bigr)+w^{(m)}\bigl(y-Y^{(m)}\bigr)},
& Y^{(k)}\le y\le Y^{(m)},\\[10pt]
\dfrac{\lambda\,w^{(k+1)}\bigl(Y^{(k+1)}-y\bigr)+w^{(m)}\bigl(y-Y^{(m)}\bigr)}{w^{(k+1)}\bigl(Y^{(k+1)}-y\bigr)+w^{(m)}\bigl(y-Y^{(m)}\bigr)},
& Y^{(m)}\le y\le Y^{(k+1)}.
\end{cases}
\tag{3}
```

The **inverse derivative** is

```math
\frac{d g^{-1}(y)}{dy}=
\begin{cases}
\dfrac{\lambda\,w^{(k)}w^{(m)}\bigl(Y^{(m)}-Y^{(k)}\bigr)}{\Bigl(w^{(k)}(Y^{(k)}-y)+w^{(m)}(y-Y^{(m)})\Bigr)^2},
& Y^{(k)}\le y\le Y^{(m)},\\[10pt]
\dfrac{(1-\lambda)\,w^{(m)}w^{(k+1)}\bigl(Y^{(k+1)}-Y^{(m)}\bigr)}{\Bigl(w^{(k+1)}(Y^{(k+1)}-y)+w^{(m)}(y-Y^{(m)})\Bigr)^2},
& Y^{(m)}\le y\le Y^{(k+1)}.
\end{cases}
\tag{4}
```

The library implements these ideas with a **symmetric grid** around the center and $\\lambda$ **derived** from two x‑spacings per bin:

```math
\lambda=\frac{p_0}{p_0+p_1},\qquad p_0=\exp(\ell_{2i}^\sigma-\log 2),\quad p_1=\exp(\ell_{2i+1}^\sigma-\log 2).
```

### 6.1 Local adjoints

On a segment $(X\_{j-1},X\_j]$, write

```math
a=W_{j-1}(X_j-v),\quad b=W_j(v-X_{j-1}),\quad S=a+b,\quad g=\frac{Y_{j-1}a+Y_j b}{S}.
```

Local derivatives of $g$ with respect to the *node variables* $(X\_s,Y\_s,W\_s)\\in{(X\_{j-1},Y\_{j-1},W\_{j-1}), (X\_j,Y\_j,W\_j)}$ are:

```math
\begin{alignat}{2}
\hat{Y}_{j-1}&=\frac{a}{S}, &\qquad \hat{Y}_{j}&=\frac{b}{S},\\
\hat{X}_{j-1}&=-\frac{W_j(Y_j-g)}{S}, &\qquad \hat{X}_{j}&=\frac{W_{j-1}(Y_{j-1}-g)}{S},\\
\hat{W}_{j-1}&=\frac{(Y_{j-1}-g)(X_j-v)}{S}, &\qquad \hat{W}_{j}&=\frac{(Y_j-g)(v-X_{j-1})}{S}.
\end{alignat}
\tag{5}
```

For any unconstrained parameter $\\theta$,

```math
\frac{\partial g}{\partial \theta}=\sum_{s\in\{j-1,j\}}\left(\hat{X}_s\frac{\partial X_s}{\partial\theta}+\hat{Y}_s\frac{\partial Y_s}{\partial\theta}+\hat{W}_s\frac{\partial W_s}{\partial\theta}\right).
\tag{6}
```

### 6.2 Node Jacobians

Below, $C\_i^+=2n+2(i+1)$ and $C\_i^-=2i$ denote the **even** indices of base knots on the positive/negative side. Odd indices are midpoints. We denote side $\\sigma\\in{+,-}$, bin index $i$, and derivative index $t$ (left endpoint of that bin): $t=i$ on the negative side, $t=n+i$ on the positive side. Let

```math
P_0=p_{2i}^\sigma,\ \ P_1=p_{2i+1}^\sigma,\ \ \Delta x=P_0+P_1,\ \ \lambda=\frac{P_0}{\Delta x},\ \ \Delta y=h_i^\sigma,
```

```math
a=W_{2t},\ \ b=W_{2t+2},\ \ \alpha=\sqrt{D_t},\ \ \beta=\sqrt{D_{t+1}},\ \ A=(1-\lambda)a+\lambda b.
```

**Shifts (non‑centered only).** For all nodes $s$:

```math
\frac{\partial X_s}{\partial x_0}=1,\qquad \frac{\partial Y_s}{\partial y_0}=1.
\tag{7}
```

**X grid (even and odd $s$).** For any $k$:

```math
\frac{\partial X_s}{\partial \ell_k^+}=p_k^{+}\,\mathbf{1}\{s\ge 2n+1+k\},\qquad
\frac{\partial X_s}{\partial \ell_k^-}=-p_k^{-}\,\mathbf{1}\{s\le k\}.
\tag{8}
```

**W grid (even $s=2t$).**

```math
\frac{\partial W_{2t}}{\partial r_{t'}}=-\frac{1}{2}\,W_{2t}\,\delta_{t t'}.
\tag{9}
```

**Midpoint weight $W\_s$ (odd $s$).** The implementation uses the classic Fuhr–Kallay construction:

```math
W_s
= \bigl(\lambda\,W_{2t}\,D_t + (1-\lambda)\,W_{2t+2}\,D_{t+1}\bigr)\,
\frac{\Delta x}{\Delta y}.
\tag{10}
```

Hence,

```math
\frac{\partial W_s}{\partial m_i^\sigma}=-W_s,\qquad
\frac{\partial W_s}{\partial \ell_{2i}^\sigma}= \frac{\alpha\,P_0}{\Delta y},\qquad
\frac{\partial W_s}{\partial \ell_{2i+1}^\sigma}= \frac{\beta\,P_1}{\Delta y},
\tag{11}
```

```math
\frac{\partial W_s}{\partial r_t}= \frac{1}{2}\frac{\Delta x}{\Delta y}\,\lambda\,\alpha,\qquad
\frac{\partial W_s}{\partial r_{t+1}}= \frac{1}{2}\frac{\Delta x}{\Delta y}\,(1-\lambda)\,\beta.
\tag{12}
```

**Midpoint value $Y\_s$ (odd $s$).**
Let the interpolation weights

```math
W_s^L=\frac{(1-\lambda)a}{A},\qquad W_s^R=\frac{\lambda b}{A},\qquad Y_s=W_s^L\,Y_{s-1}+W_s^R\,Y_{s+1}.
\tag{13}
```

Then

```math
\frac{\partial Y_s}{\partial \lambda}=\frac{a\,b\,(Y_{s+1}-Y_{s-1})}{A^2},
\quad
\frac{\partial Y_s}{\partial \ell_{2i}^\sigma}=\frac{\partial Y_s}{\partial \lambda}\,\lambda(1-\lambda),
\quad
\frac{\partial Y_s}{\partial \ell_{2i+1}^\sigma}=-\frac{\partial Y_s}{\partial \lambda}\,\lambda(1-\lambda),
\tag{14}
```

```math
\frac{\partial Y_s}{\partial r_t}
=-\frac{1}{2}\,a\,\frac{(1-\lambda)(Y_{s-1}-Y_s)}{A},\qquad
\frac{\partial Y_s}{\partial r_{t+1}}
=-\frac{1}{2}\,b\,\frac{\lambda\,(Y_{s+1}-Y_s)}{A}.
\tag{15}
```

**Y grid (even vs. odd $s$).** Prefix‑sum contributions:

```math
\frac{\partial Y_s}{\partial m_i^+} = h_i^+ \times
\begin{cases}
\mathbf{1}\{s \ge C_i^+\}, & s\ \text{even},\\
W_s^L\,\mathbf{1}\{s-1 \ge C_i^+\} + W_s^R\,\mathbf{1}\{s+1 \ge C_i^+\}, & s\ \text{odd},
\end{cases}
\tag{16a}
```

```math
\frac{\partial Y_s}{\partial m_i^-} = -h_i^- \times
\begin{cases}
\mathbf{1}\{s \le C_i^-\}, & s\ \text{even},\\
W_s^L\,\mathbf{1}\{s-1 \le C_i^-\} + W_s^R\,\mathbf{1}\{s+1 \le C_i^-\}, & s\ \text{odd}.
\end{cases}
\tag{16b}
```

### 6.3 Assembled derivatives (interior)

From (6) and the Jacobians above:

**Center coordinates.**

```math
\frac{\partial g}{\partial x_0}=\hat{X}_{j-1}+\hat{X}_j,\qquad
\frac{\partial g}{\partial y_0}=\hat{Y}_{j-1}+\hat{Y}_j \ (=1).
\tag{17}
```

**X‑spacings $\\ell\_k^\\sigma$.** Two types of contributions:

```math
\frac{\partial g}{\partial \ell_k^\sigma} = T_X + T_{\mathrm{Mid}},
```

```math
T_X(\ell_k^+) = p_k^+\bigl(\hat{X}_{j-1}\mathbf{1}\{j-1\ge 2n+1+k\}+\hat{X}_j\mathbf{1}\{j\ge 2n+1+k\}\bigr),
```

```math
T_X(\ell_k^-) = -p_k^-\bigl(\hat{X}_{j-1}\mathbf{1}\{j-1\le k\}+\hat{X}_j\mathbf{1}\{j\le k\}\bigr),
\tag{18}
```

```math
T_{\mathrm{Mid}}=\sum_{s\in\{j-1,j\}}\Bigl(\hat{W}_s\frac{\partial W_s}{\partial \ell_k^\sigma} + \hat{Y}_s\frac{\partial Y_s}{\partial \ell_k^\sigma}\Bigr),
```

with $\\partial W\_s/\\partial\\ell,\\ \\partial Y\_s/\\partial\\ell$ taken from (11) and (14).

**Y‑heights $m\_i^\\sigma$.**

```math
\frac{\partial g}{\partial m_i^\sigma}= T_Y + T_W,
```

```math
T_W(m_i^\sigma)=\sum_{s\in\{j-1,j\}}\hat{W}_s(-W_s)\cdot \mathbf{1}\{ s=\text{odd midpoint index of bin } i \text{ on side } \sigma\},
\tag{19}
```

```math
T_Y(m_i^\sigma)=\hat{Y}_{j-1}\frac{\partial Y_{j-1}}{\partial m_i^\sigma}+\hat{Y}_j\frac{\partial Y_j}{\partial m_i^\sigma},
```

with $\\partial Y/\\partial m$ from (16).

**Log‑derivatives $r\_t$.**

```math
\frac{\partial g}{\partial r_t}= T_{W,\mathrm{even}} + T_{\mathrm{Mid}},
```

```math
T_{W,\mathrm{even}}=\sum_{s\in\{j-1,j\}}\hat{W}_s\Bigl(-\tfrac{1}{2}W_s\Bigr)\mathbf{1}\{s=2t\},\qquad
T_{\mathrm{Mid}}=\sum_{s\in\{j-1,j\}}\Bigl(\hat{W}_s\frac{\partial W_s}{\partial r_t}+\hat{Y}_s\frac{\partial Y_s}{\partial r_t}\Bigr),
\tag{20}
```

with $\\partial W\_s/\\partial r\_t$ from (12) and $\\partial Y\_s/\\partial r\_t$ from (15).

### 6.4 Assembled derivatives (tails)

Let $M=4n$, $D\_L=D\_0$, $D\_R=D\_{2n}$.

**Left tail** $(v\\le X\_0)$: $g(v)=Y\_0+(v-X\_0)D\_L$.

```math
\frac{\partial g}{\partial y_0}=1,\quad
\frac{\partial g}{\partial x_0}=-D_L,\quad
\frac{\partial g}{\partial r_0}=D_L\,(v-X_0),
```

```math
\frac{\partial g}{\partial \ell_k^-}=D_L\,p_k^-,\qquad
\frac{\partial g}{\partial m_i^-}=-h_i^-,
```

all other partials are zero.

**Right tail** $(v\\ge X\_M)$: $g(v)=Y\_M+(v-X\_M)D\_R$.

```math
\frac{\partial g}{\partial y_0}=1,\quad
\frac{\partial g}{\partial x_0}=-D_R,\quad
\frac{\partial g}{\partial r_{2n}}=D_R\,(v-X_M),
```

```math
\frac{\partial g}{\partial \ell_k^+}=-D_R\,p_k^+,\qquad
\frac{\partial g}{\partial m_i^+}=h_i^+,
```

all other partials are zero.

### 6.5 Inverse‑mode parameter derivatives

For the **inverse map** $x=g^{-1}(y)$, we use the **Implicit Function Theorem**. Let $y=g(x; \\Theta)$, with $x=g^{-1}(y;\\Theta)$. Then

```math
\frac{\partial x}{\partial \Theta}
= -\,\frac{\partial g/\partial \Theta}{\partial g/\partial x}
\quad \text{evaluated at } x=g^{-1}(y).
\tag{21}
```

In code, we compute forward‑mode parameter derivatives $\\partial g/\\partial\\Theta$ at the recovered $x$, divide by the forward derivative $\\partial g/\\partial x$, and apply a robust $\\varepsilon$ clamp. In *saturation* regions where $\\partial g/\\partial x \\approx 0$, we zero the inverse gradients for numerical stability.

-----

## Python API

### `UnifiedMonotonicSpline`

```python
UnifiedMonotonicSpline(
    n_of_nodes: Optional[int],       # N bins per side if not None (Mode 1). If None, Mode 2.
    inverse: bool = False,           # Apply inverse mapping?
    direction: Union[str,int] = 'increasing',   # 'increasing'/1 or 'decreasing'/-1
    centered: bool = True            # Fix (0,0) or learn (x_0,y_0)
)
```

  * **Modes**

      * **Mode 1 (internal params):** `n_of_nodes` is an `int`. The layer holds trainable parameters.
      * **Mode 2 (external params):** `n_of_nodes=None`. You must pass `spline_weights` to `forward`.

#### `forward(input_data, spline_weights=None)`

  * **Mode 1:** `spline_weights` must be `None`. A **single** spline is applied elementwise to **all** values in `input_data` (any shape).

  * **Mode 2:** `spline_weights` is required.
    Shape rules:

      * Let `input_data.shape = (*B, L)`. We flatten leading dims: `B_flat = prod(*B)`.
      * `spline_weights.shape` must be either `(*B, W)` or `(B_flat, W)`.
      * $W$ must be $8N+1$ (centered) or $8N+3$ (non‑centered), $N\\ge 1$.

**Return:** same shape as `input_data`.

**Parameter layout (Mode 2)** for a single vector of length $W$:

```
[ x_pos(2N), x_neg(2N), y_pos(N), y_neg(N), ln_d(2N+1),  [x_0, y_0] ]
```

where each block contains unconstrained entries (internally exponentiated); $x\_0,y\_0$ are present iff `centered=False`.

#### `SaveSplineWeights(path, append=False)`  *(Mode 1 only)*

Writes a simple text format (see §9) with a version header `#VER = 1001`.

#### `ExtractParamsForExternal(model) -> torch.Tensor`

Utility returning a single Mode‑2 weight vector that reproduces the given Mode‑1 model (handles the $\\log 2$ offset required by Mode‑2 parsing). Length is $8N+1$ or $8N+3$.

-----

## C++ API

All in `cpp/mlrsplines/LRSplines.h` under namespace `ns_base`.

```cpp
// Mode 1 (internal parameters live inside the object)
template<class T> using T_LRSplines      = T_UnifiedMonotonicSpline<T, smInternal>;

// Mode 2 (external parameters supplied at call time)
template<class T> using T_LRSplinesInput = T_UnifiedMonotonicSpline<T, smExternal>;
```

Common constructor:

```cpp
T_LRSplines<double> s(centered /*bool*/, direction /*+1 or -1*/);
```

**Mode 1 methods**

  * `void TextLoad(const std::string& file)` — load text weights (from Python `SaveSplineWeights`).
  * `T Calc(T x) const` — forward value.
  * `T CalcInv(T y) const` — inverse value.
  * `T CalcDeriv(T x) const` — forward derivative $\\partial g/\\partial x$.
  * `T CalcInvDeriv(T y) const` — inverse derivative $\\partial g^{-1}/\\partial y$.
  * `t_params CalculateGradients(T v) const` — **parameter** derivatives $\\partial g/\\partial\\Theta$ at input $v$.
  * `t_params CalculateInverseGradients(T y) const` — **inverse‑mode** parameter derivatives using (21).

**Mode 2 methods**

  * `T Calc(const T* p, size_t n, T x) const`
  * `T CalcInv(const T* p, size_t n, T y) const`
  * `T CalcDeriv(const T* p, size_t n, T x) const`
  * `T CalcInvDeriv(const T* p, size_t n, T y) const`
  * `t_params CalculateGradients(const T* p, size_t n, T v) const`
  * `t_params CalculateInverseGradients(const T* p, size_t n, T y) const`
  * `void UpdateCache(const T* p, size_t n)` — precompute knots if repeatedly applying the same params.

**Exceptions.** Errors throw `ns_base::LRSplinesException` with a small error code taxonomy (uninitialized, bad sizes, bad file format, invalid config).

-----

## File format & cross‑language interoperability

Python `SaveSplineWeights` (Mode 1) writes:

```
#VER = 1001
x_pos (2N numbers, tab-separated)
x_neg (2N)
y_pos (N)
y_neg (N)
ln_d  (2N+1)
[x_0  y_0]  # only if non-centered
```

**Important offsets.**

  * **C++ Mode 1 `TextLoad`**: adds $\\log(2)$ to `x_pos` and `x_neg` on load so that the *internal* cache and the *external* path use the same exponential $\\exp(P - \\log 2)$ convention.
  * **Python `ExtractParamsForExternal`**: adds $+\\log(2)$ to `x_pos/x_neg` so the resulting vector can be passed to **Mode 2** on both Python and C++ sides and produce identical knots.

-----

## Numerical stability & notes

  * We clamp denominators by a small $\\varepsilon$ depending on dtype (FP16/BF16: $10^{-4}$, else $10^{-6}$).
  * The “decreasing” direction is implemented as a **wrapper**: forward uses $g(-x)$, inverse returns $-g^{-1}(y)$. Derivatives respect the chain rule.
  * Inverse‑mode parameter gradients divide by $\\partial g/\\partial x$; when the forward slope saturates to \~0, the library **zeros** inverse gradients to avoid blow‑ups.
  * Binary search (`std::upper_bound` / `torch.searchsorted`) picks the right interval; indices are clamped to $[1,4n]$.

-----

## Examples & tests

### Training in Python (Mode 1)

See `python/tests/TestLRSplines.py` for a complete harness that trains several configurations (increasing/decreasing, centered/non‑centered, inverse) and plots predicted vs target curves. It also:

  * Exports to text and reloads in C++.
  * Shows **Mode‑2 equivalence**: the external‑weights path matches Mode 1 within $10^{-6}$.
  * Checks **round‑trip** $x \\stackrel{g}{\\mapsto} y \\stackrel{g^{-1}}{\\mapsto} x$ to $10^{-4}$.
  * Verifies **gradient flow** to both inputs and external weights in autograd.

### C++ tests

`cpp/tests/TestLRSplines.cpp` runs a concise battery:

  * Value/derivative consistency across **Internal / External (on‑the‑fly / cached / ctor)** paths.
  * Inverse identities $g^{-1}(g(x))=x$, and $(\\partial g/\\partial x)\\cdot(\\partial g^{-1}/\\partial y)\\approx 1$ away from saturation.
  * **Analytical vs. numerical** derivatives via Richardson extrapolation.
  * API behavior (uninitialized errors, move semantics).

-----

## FAQ

**Q: Why use linear rational (homographic) splines?**
They admit a **closed‑form inverse** with the **same algebraic form** as the forward, so forward and inverse costs are symmetric. In contrast, quadratic/cubic rational splines often require solving a polynomial for inversion. See Dolatabadi et al. for flow motivation. ([Proceedings of Machine Learning Research][1]). For using in flows/copulas/probability density estimation, you often want to use splines to model the shape of the distribution, not location. So, the default mode forces (0,0) to be the central point: it is always transformed into itself, everything below it stay below, above stay above.

**Q: How is monotonicity guaranteed?**
All increments $p^\\pm, h^\\pm, D$ are exponentials of unconstrained parameters, hence positive. Midpoint weights $W\_s$ are constructed to maintain positive slope in each piece (Fuhr–Kallay), and linear tails inherit positive slopes from $D\_0,D\_{2n}$. ([ScienceDirect][2])

**Q: Can I condition the spline on context (flows)?**
Yes—use **Mode 2** and *predict the external weight vector* from your context network, then call the layer with those weights.

**Q: What about log‑det‑Jacobian for flows?**
The C++ API exposes $\\partial g/\\partial x$ and $\\partial g^{-1}/\\partial y$. In elementwise flows, `logabsdet` is the sum of $\\log\\bigl|\\partial g/\\partial x\\bigr|$ across dimensions. In Python, you can compute it by porting the derivative formula or wrapping the C++.

-----

## References

  * **H. M. Dolatabadi, S. Erfani, C. Leckie.** *Invertible Generative Modeling using Linear Rational Splines.* AISTATS 2020, PMLR v108. ([Proceedings of Machine Learning Research][1])
  * **R. D. Fuhr, M. Kallay.** *Monotone linear rational spline interpolation.* Computer Aided Geometric Design, 1992. ([ScienceDirect][2], [ACM Digital Library][3])

-----

### Appendix: Implementation correspondence (code ↔ math)

  * **Knots & weights.** `CalculateKnots` (C++) and `_CalculateKnots` (Py) implement the constructions of §5.2–§5.4:

      * base weights $W\_{2t}=1/\\sqrt{D\_t}$,
      * mid‑location $\\lambda=\\tfrac{p\_0}{p\_0+p\_1}$,
      * midpoint weight $W\_s$ from (10),
      * midpoint value $Y\_s$ from (13),
      * cumulative $X,Y$ with optional center offsets $x\_0,y\_0$,
      * linear tails with slopes $D\_0,D\_{2n}$.

  * **Forward evaluation.** `ApplySplineUnified` (C++) / `_ApplySpline` (Py) are the barycentric forms (F) with piecewise search and tail handling.

  * **Gradients.** `CalculateGradientsUnified` (C++) implements (5)–(20).
    `CalculateInverseGradients` applies (21).

<!-- end list -->

```
```

[1]: https://www.google.com/search?q=%5Bhttps://proceedings.mlr.press/v108/dolatabadi20a.html%3Futm_source%3Dchatgpt.com%5D\(https://proceedings.mlr.press/v108/dolatabadi20a.html%3Futm_source%3Dchatgpt.com\) "Invertible Generative Modeling using Linear Rational Splines"
[2]: https://www.google.com/search?q=%5Bhttps://www.sciencedirect.com/science/article/pii/016783969290038Q%3Futm_source%3Dchatgpt.com%5D\(https://www.sciencedirect.com/science/article/pii/016783969290038Q%3Futm_source%3Dchatgpt.com\) "Monotone linear rational spline interpolation"
[3]: https://www.google.com/search?q=%5Bhttps://dl.acm.org/doi/10.1016/0167-8396%252892%252990038-Q%3Futm_source%3Dchatgpt.com%5D\(https://dl.acm.org/doi/10.1016/0167-8396%252892%252990038-Q%3Futm_source%3Dchatgpt.com\) "Monotone linear rational spline interpolation"