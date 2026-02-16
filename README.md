# ml-tidbits

Small, focused ML utilities in **Python/PyTorch** and **C++** that emphasize **clarity**, **latency**, and **practicality**.
Each component is self-contained with docs, tests, and (when relevant) a C++ counterpart.
C++ components are all tested on Windows and Mac, gcc, clang and Visual Studio 2022. Current requirements are C++17, I will later switch to C++20.

---

## Components

| File | What it provides |
|---|---|
| `LRSplines.py` | `UnifiedMonotonicSpline` &mdash; invertible monotone rational linear spline layer |
| `EmbedModels.py` | Learnable Euclidean attention, auto-compressing networks, invertible flows, and the **Wristband Gaussian Loss** for deterministic Gaussian autoencoders |
| `TestLRSplines.py` | Tests and training examples for `LRSplines.py` |
| `DeterministicGAE.py` | End-to-end example: **Deterministic Gaussian Autoencoder** built from the modules in `EmbedModels.py` |

---

## UnifiedMonotonicSpline (`LRSplines.py`)

> A unified, batched, **TorchScript-compatible monotone rational linear spline** layer with forward/inverse modes and increasing/decreasing directions.

**Why it's cool:** `UnifiedMonotonicSpline` is an **invertible, batched monotonic rational linear spline** module available in both **PyTorch** and **modern C++**, using one **unconstrained parameterization** (`8N+1` or `8N+3` if uncentered) shared across languages. It guarantees monotonicity by construction (exponential spacing/derivative params), provides stable **forward and inverse** transforms with linear tails, and exposes **analytic derivatives** (and inverse-gradients via the implicit function theorem) for seamless training &mdash; including an optimized `forward(return_deriv=True)` path that computes both value and derivative in a single pass sharing knot calculation and bin search. The PyTorch module is fully **TorchScript-compatible** (`torch.jit.script`) for deployment without Python overhead. You can run it with **internal weights** (single spline applied to many values) or **external weights** (per-example splines in a batch), making it ideal for **normalizing flows**, **calibration layers**, **monotone neural networks**, tabular feature transforms, and differentiable bijective scalers. The C++17 implementation mirrors the PyTorch API, is tested on Windows/macOS with MSVC/Clang/GCC, and the on-the-fly knot computation uses vectorized barycentric rational interpolation with binary search over the `4N+1` knots for fast, low-latency inference.

---

## Embedding Models and Wristband Gaussian Loss (`EmbedModels.py`)

`EmbedModels.py` is a self-contained PyTorch module that provides everything needed to build a **Deterministic Gaussian Autoencoder** &mdash; an autoencoder whose latent space is pushed toward N(0,I) *without* the reparameterization trick, KL divergence, or any stochastic sampling. The file contains five components that compose together into a clean training pipeline: a learnable attention layer for encoding and decoding, a residual network backbone, an invertible normalizing flow for latent-space shaping, and a novel distribution-matching loss. See `DeterministicGAE.py` for a complete working example.

### C_EmbedAttentionModule

A **multi-head softmax attention layer with fully learnable keys and values**. Unlike standard transformer attention where keys and values are projections of input tokens, here both the keys `k` and values `v` are free parameters of shape `(n_heads, n_basis, dim)` stored directly in the module. The input serves only as the query.

**What makes it interesting:**

- **Euclidean attention mode.** Instead of the standard scaled dot-product `<q,k>/sqrt(d)`, the default mode computes `<q,k> - 0.5*||k||^2`, which is the log-kernel of a Gaussian centered at `k`. This makes each basis point act as an RBF prototype &mdash; attention weight decays with squared Euclidean distance from the query to the key, giving the layer a natural "nearest-prototype" inductive bias. The module can also be switched to standard dot-product attention when Euclidean geometry is not appropriate.

- **Learnable layer-norm temperature.** Logits are passed through a per-basis learned layer-norm scale (and optionally a per-head temperature), giving the network fine-grained control over attention sharpness during training without manual tuning.

- **Rank-1 affine experts.** An optional mode where the value returned for each basis point is not a fixed vector but a query-dependent affine function: `v(q) = v_bias + v_out * (q . v_in)`. This is a rank-1 perturbation that lets each basis point specialize its output based on the query, adding expressiveness at minimal parameter cost. The affine contribution is initialized near zero for training stability.

- **Flexible composition.** The module accepts an optional `q_transform` (applied to queries before attention) and an optional `head_combine` (applied to concatenated head outputs). This makes it easy to drop in any backbone as either the query preprocessor or the output projector &mdash; the example uses `C_ACN` for the head combiner.

### C_ACN (Auto-Compressing Network)

A residual MLP based on the **Auto-Compressing Network** architecture from [Dorovatas et al., 2025](https://arxiv.org/abs/2506.09714).

The key difference from a standard residual network is the connectivity pattern: instead of short skip connections between adjacent layers (`x = f(x) + x`), ACN uses **additive long connections from every hidden layer directly to the output**. Concretely, each block's output is *accumulated* into a running sum, and only the final sum is projected to the output:

```
a = in_proj(x)
res = a                 # contribution from layer 1
for block in blocks:
    a = block(elu(a))
    res = res + a       # layer 2, 3, ... all add directly to res
output = out_proj(elu(res))
```

**Why this matters:** The paper shows that this wiring pattern causes the network to automatically "push" information into earlier layers during training, making later layers progressively more compressible. In practice this means: networks can be pruned aggressively after training (30-80% compression with no accuracy loss in their experiments), they exhibit better noise robustness, and they mitigate catastrophic forgetting in continual learning. In `EmbedModels.py`, ACN is used as the conditioner inside affine coupling layers and as the head combiner for the attention module.

### C_InvertibleFlow (with C_AffineCouplingLayer and C_PermutationLayer)

A composable **RealNVP-style normalizing flow** with exact forward and inverse passes.

The flow is a stack of affine coupling layers interleaved with deterministic permutations. Each coupling layer splits the input into a "pass-through" half and a "transformed" half; the pass-through half conditions an ACN that predicts scale and shift parameters for the transformed half. The log-scale is bounded via `tanh` for numerical stability.

**Design choices:**

- **Exact inverse by construction.** Both `forward()` and `inverse()` are available and exact (no approximation, no fixed-point iteration). This is essential for the Deterministic Gaussian Autoencoder: the encoder maps data through the flow to produce Gaussian latents, and the decoder needs the exact inverse to map latents back before reconstruction.

- **Deterministic permutations.** Between coupling layers, a fixed random permutation shuffles dimensions so that different subsets of features interact in each layer. The permutation is seeded deterministically (`permute_seed`) for reproducibility. Three modes are supported: `per_pair` (permute every two layers), `per_layer` (every layer), or `none` (use alternating/half masks instead).

- **Fast-path detection.** The coupling layer automatically detects whether the binary mask produces a contiguous split (`[pass | trans]` or `[trans | pass]`) and uses simple slicing instead of `index_select`/`index_copy_` for better performance.

- **Identity initialization.** All coupling layers start near the identity transform (conditioner output weights and biases initialized to zero), so the flow begins as a pass-through and gradually learns to reshape the distribution. This makes training stable from the start.

### C_WristbandGaussianLoss

A batch loss that encourages a set of samples to follow N(0,I), designed for training deterministic autoencoders. This is the core novel contribution of the module.

**The problem it solves:** In a VAE, the KL divergence provides a per-sample signal pushing the approximate posterior toward the prior. In a deterministic autoencoder there is no per-sample posterior &mdash; you have a batch of point embeddings and you need a loss that says "this batch looks Gaussian." This is surprisingly hard to do well.

**How it works:** The loss decomposes each sample `x` into a direction `u = x/||x||` (a point on the unit sphere) and a radius CDF `t = gammainc(d/2, ||x||^2/2)` (the chi-squared CDF, which is uniform under the null). This `(u, t)` representation is called the "wristband" because it maps the Gaussian cloud onto the product of a sphere and a unit interval. The loss then applies three complementary forces:

**Conceptual origin:** The wristband loss can be understood as an extension of [Wang & Isola's Uniform loss](https://arxiv.org/abs/2005.10242) from the hypersphere to the Gaussian distribution. Wang & Isola showed that a pairwise repulsive kernel on the unit sphere encourages uniform angular distribution of embeddings, and that this is equivalent to optimizing alignment and uniformity in contrastive learning. The wristband loss takes the same core idea &mdash; pairwise soft repulsion via a log-mean-exp kernel &mdash; but applies it to the full `(u, t)` wristband space so that both the angular *and* radial components are pushed toward the distributions implied by N(0,I). Generally, repulsive component is all you need - uniform on a Wristband implies Gaussian in the original space. However we also add a radial W2 term and moment penalty - they speed up convergence and improve robustness (can be turned off for a pure Wang&Isola-style loss function).

1. **Joint repulsion** (the main term). A soft Gaussian kernel measures the pairwise "closeness" of samples in the wristband space. To handle the bounded radial coordinate `t in [0,1]`, the kernel uses a **3-image reflection** method: each sample is reflected at both boundaries `t=0` and `t=1`, producing three "copies" whose kernel contributions are summed. This eliminates boundary artifacts that would otherwise push mass toward the edges. The loss is the log-mean of pairwise kernel values &mdash; minimizing it spreads samples apart uniformly.

2. **Radial uniformity.** A 1D squared Wasserstein distance between the sorted `t` values and the quantiles of Unif(0,1). This directly enforces that the radial CDF is uniform, which is equivalent to the radial distribution matching the chi distribution with `d` degrees of freedom. Cheap (just a sort) and provides a strong global signal.

3. **Moment matching.** A configurable penalty on the first and second moments. The default (`"w2"`) is the squared 2-Wasserstein distance between the Gaussian fit to the batch and N(0,I), computed via eigenvalues of the sample covariance. Other options include diagonal/full-covariance KL, half-Jeffreys divergence, or simple mean penalty. This term catches any remaining global drift or variance mismatch.

**Automatic calibration.** At construction time, the loss runs a Monte-Carlo calibration: it draws many batches from N(0,I) and records the mean and standard deviation of each component. During training, each component is z-scored against these null statistics and the weighted sum is divided by its own null standard deviation. This means the total loss is a z-score &mdash; a value near zero means the batch is indistinguishable from Gaussian, and positive values indicate how many standard deviations away from Gaussian it is. This eliminates all scale-dependent hyperparameter tuning: the default weights just work across different batch sizes and dimensions.

### Why not other distribution-matching losses?

The specific form of the Wristband Gaussian Loss has been refined over several years of practical use in production systems. Here is why it outperforms the common alternatives:

**Hungarian-algorithm / optimal-transport matching.** Matching each sample in the batch to a corresponding sample from N(0,I) via the Hungarian algorithm gives an excellent gradient signal, but the cubic time complexity O(N^3) makes it impractical for batches larger than a few hundred. It is also inherently sequential and does not parallelize well on GPUs.

**Moment matching (mean + covariance).** Penalizing `||mu||^2 + ||Sigma - I||_F^2` is fast and easy, but it only constrains the first two moments. The resulting latent distributions are often visibly non-Gaussian: skewed, heavy-tailed, or multimodal. Two very different distributions can have identical first two moments.

**Kernel methods and MMD.** Maximum Mean Discrepancy with a Gaussian or inverse-multiquadric kernel is a popular choice (used in WAE, InfoVAE, and others). However, MMD suffers from the curse of dimensionality: as the latent dimension grows, the kernel values between all pairs converge to a constant, and the signal-to-noise ratio of the MMD estimator collapses. In practice, MMD requires increasingly large batches to work in dimensions above 10-15, and kernel bandwidth tuning becomes fragile.

**Distance correlation (dCor).** dCor can detect arbitrary dependencies between dimensions but provides a weak signal for *marginal* Gaussianity. It tells you that dimensions are uncorrelated but not that each margin follows a Gaussian. Empirically, training with dCor produces latent spaces with correct correlation structure but non-Gaussian marginals, and convergence is slow and noisy.

**Sliced Wasserstein distance.** Sliced Wasserstein projects samples onto random 1D directions and compares the resulting 1D distributions. It scales well and avoids the curse of dimensionality that plagues MMD. However, it struggles with two specific issues: angular uniformity (lack thereof), and local dependency (nearby points in the latent space tend to form clumps and correlated patterns that are invisible to random 1D projections but clearly non-Gaussian in the joint distribution). In practice, training with Sliced Wasserstein produces latents whose marginals look reasonable but whose joint structure shows visible local clustering.

**KL divergence (VAE-style).** The standard VAE approach requires the encoder to output mean and log-variance per sample, adding parameters and the reparameterization trick. It also notoriously suffers from posterior collapse (the encoder learns to output the prior, ignoring the input) and requires careful beta-annealing. Crucially, VAEs are **non-deterministic by design**: the same input produces different latent samples on every forward pass due to the injected noise, which is unacceptable in applications that require reproducibility (e.g., financial modeling, scientific simulation, or any system where you need the same input to always produce the same output). The wristband loss operates on point embeddings directly &mdash; the encoder is a plain deterministic function, the mapping is one-to-one, and there is no sampling anywhere in the pipeline.

**Flow matching and diffusion models.** Modern diffusion-based approaches (score matching, flow matching, rectified flows) can learn very accurate transport maps from data to Gaussian. However, they are **multi-step iterative processes**: generation requires running an ODE/SDE solver for many steps (typically 20-1000), each step is an approximation, and the transport is learned via denoising objectives that require noise-level scheduling. Some variants are also non-deterministic (SDE-based samplers). In contrast, the Deterministic Gaussian Autoencoder produces the embedding in a **single forward pass** through the encoder and flow &mdash; no iteration, no approximation, no scheduling. The invertible flow is exact, not learned via a simulation-based objective. This makes it orders of magnitude faster at inference and trivially reproducible.

**Density-ratio matching (the Sugiyama trick / GAN-like discriminator).** Train a discriminator network to estimate the density ratio between your batch and samples from the true Gaussian. The generator (encoder) pushes the ratio toward 1. This is theoretically sound — it's effectively a GAN where the "real" distribution is N(0, I). In practice it inherits GAN training pathologies: the discriminator and encoder play a min-max game that is unstable, mode-seeking, and slow to converge. The density-ratio estimator is noisy in moderate dimensions, and the adversarial dynamics introduce a whole separate tuning nightmare (discriminator architecture, learning rate ratio, update schedule).

The wristband loss avoids all of these issues. The `(u, t)` decomposition factors the problem into angular and radial components that can each be tested efficiently and, importantly, their *independence* is naturally enforced. The reflected-boundary kernel handles the bounded radial coordinate correctly. The calibration makes component weights self-tuning. The whole computation is O(N^2) in batch size (dominated by the pairwise kernel), fully GPU-parallelizable via standard `einsum` and `softmax` operations, and numerically stable under AMP/mixed-precision. And unlike VAEs or diffusion models, the resulting system is fully deterministic and single-pass: the same input always produces the same embedding, in one forward call.

---

## Putting it all together: Deterministic Gaussian Autoencoder

The four components above compose into a simple and effective architecture:

```
Input x
  |
  v
[Encoder: C_EmbedAttentionModule]  -- learnable-key attention maps x to a raw embedding
  |
  v
[Flow: C_InvertibleFlow.forward()]  -- invertible flow shapes the embedding toward N(0,I)
  |
  v
Latent z  <-- C_WristbandGaussianLoss is applied here to push z toward N(0,I)
  |
  v
[Flow: C_InvertibleFlow.inverse()]  -- exact inverse undoes the flow
  |
  v
[Decoder: C_EmbedAttentionModule]  -- learnable-key attention maps embedding back to data space
  |
  v
Reconstruction x_hat  <-- MSE loss against x
```

The total loss is simply `lambda_rec * MSE(x, x_hat) + lambda_wb * wristband(z)`. No reparameterization trick, no sampling, no KL balancing, no beta-annealing, no multi-step ODE solving. The flow gives the encoder freedom to produce whatever intermediate representation is best for reconstruction, while the wristband loss ensures that *after* the flow, the latent distribution is Gaussian. Because the flow is exactly invertible, no information is lost in the forward-inverse round trip. The entire pipeline is **deterministic** &mdash; the same input always maps to the same latent code and the same reconstruction, in a **single forward pass**.

See `DeterministicGAE.py` for a complete, runnable training script that generates non-Gaussian synthetic data, trains this architecture, and verifies that the latent space converges to N(0,I).

---

## Quick start

```python
import torch
from EmbedModels import (
    C_EmbedAttentionModule,
    C_ACN,
    C_InvertibleFlow,
    C_WristbandGaussianLoss,
)

# Encoder: 15-D input -> 8-D latent
encoder = C_EmbedAttentionModule(
    15, 128, 8, n_of_basis=256, n_of_heads=64,
    q_transform=torch.nn.Linear(15, 15),
    head_combine=C_ACN(64 * 128, 8, 128, 2),
)

# Invertible flow on the 8-D latent space
flow = C_InvertibleFlow(8, n_layers=4, hidden_dim=32, permute_mode="per_pair")

# Wristband loss, calibrated for batch=1024, dim=8
wristband = C_WristbandGaussianLoss(calibration_shape=(1024, 8))

# Forward pass
x = torch.randn(1024, 15)
z = flow(encoder(x))
loss_components = wristband(z)        # .total is the scalar to backpropagate
loss_components.total.backward()
```
