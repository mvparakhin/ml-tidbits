# Deterministic counterfactuals when factors aren't independent

**Sampling MNIST top halves from the right conditional distribution, given the bottom half (no stochastic encoder)**

If you haven't seen my first writeup: I spent years chasing a very specific goal — a *deterministic* encoder that maps data into a latent space that actually behaves like $\mathcal{N}(0, I)$, so you can do "swap a factor / resample a factor" counterfactuals by literally appending fresh Gaussian noise. The first post was mostly about the *loss* that makes this practical (Wristband). This one is about the *architecture* you need when the world isn't neatly factorized.

Because the "just sample the missing block from a Gaussian" trick has a catch:

It only works cleanly when the two pieces you're swapping are (approximately) independent.

---

## Where the naive idea breaks

In my earlier example (text + weather), those two things were effectively independent, so sampling "random weather" independently was legitimate.

Images are not like that.

The top of a digit and the bottom of a digit are strongly dependent. If you show someone only the bottom half of an MNIST "3", they won't imagine a random top — the bottom already constrains what the top *should probably be*.

So if we want conditional generation (inpainting), we need to respect that dependency.

---

## The MNIST inpainting task

Take MNIST ($28 \times 28$), split each image into:

- **bottom** = rows 14–27
- **top** = rows 0–13

**Training:** we see both top and bottom.
**Inference:** we pretend we only see bottom.

What we want after training:

- given a fixed bottom, we can generate multiple sharp, plausible full digits
- the bottom stays consistent (it's the evidence)
- the top varies in the right ways (the uncertainty)

That's a conditional distribution: "top given bottom". I'm also going to call it a *posterior* sometimes, but I'll clarify what I mean by that below.

---

## The core idea: context vs residual

The mistake is trying to make "top pixels" and "bottom pixels" independent. They aren't.

What you can do instead is split responsibility:

- **context** (deterministic): everything the bottom lets you infer about the whole digit
- **residual** (random): the genuinely missing information about the top after you've used the bottom

The dependence between top and bottom is handled through the deterministic context. The random part is only the leftover ambiguity.

That's the "dependency-aware" version of factorization.

---

## The asymmetric architecture (and why it's asymmetric)

We build:

**Bottom encoder** (large latent)

$$
z_{\text{bottom}} = E_{\text{bottom}}(\text{bottom})
$$

**Top encoder** (small latent)

$$
z_{\text{top}} = E_{\text{top}}(\text{top})
$$

**Decoder**

$$
z = [z_{\text{bottom}},\; z_{\text{top}}]
$$

$$
\hat{x} = \text{Decoder}(z) \quad \text{(full } 28 \times 28\text{)}
$$

And then we enforce a key constraint:

**Gaussian interface** — batches of $z$ should look like $\mathcal{N}(0, I)$.

(This is where Wristband comes in for me: it makes "looks like $\mathcal{N}(0, I)$" achievable and stable in a deterministic model.)

The asymmetry matters: $z_{\text{top}}$ is smaller on purpose.

If $z_{\text{top}}$ is large, the model can "cheat" by routing too much information through the top encoder during training (when the top is available). Then at inference time, sampling random $z_{\text{top}}$ won't land on the right manifold for the given bottom.

Making $z_{\text{top}}$ small forces a clean division of labor:

- $z_{\text{bottom}}$ must carry everything the bottom implies
- $z_{\text{top}}$ is reserved for what the bottom cannot determine

---

## Training: two modes, two losses

We train the same network in two modes.

### Loss \#1 — Full reconstruction (the "sharpness" anchor)

We have both halves, so we do the obvious thing:

- encode bottom and top
- decode the full image
- penalize reconstruction error ($L_1$ or $L_2$)
- plus Gaussianity loss on the concatenated latent

This is the constraint that forces the model to actually learn to reconstruct real, sharp digits.

### Loss \#2 — Bottom-only + sampling (marginalize the unknown top)

Now we train the model the way we'll use it:

- encode only bottom: $z_{\text{bottom}} = E_{\text{bottom}}(\text{bottom})$
- do **not** encode top at all
- sample $K$ times: $\varepsilon_1 \ldots \varepsilon_K \sim \mathcal{N}(0, I)$ (these stand in for unknown $z_{\text{top}}$)
- decode $K$ reconstructions: $\hat{x}(\varepsilon_k) = \text{Decoder}([z_{\text{bottom}},\; \varepsilon_k])$
- and score them against the same ground-truth image

Here's the important disambiguation:

We average the *loss*, not the output.

$$
\mathcal{L}_2 = \frac{1}{K} \sum_{k=1}^{K} L\bigl(\hat{x}(\varepsilon_k),\; x_{\text{true}}\bigr)
$$

Conceptually, this is integrating out the missing information (the unknown "top latent") inside the objective:

> "I don't know $z_{\text{top}}$. Treat it as standard normal noise, and minimize the expected reconstruction error under that uncertainty."

A useful note: if your loss is squared $L_2$, there's a special relationship where people often talk as if this corresponds to matching an "expected image". That intuition is fine for $L_2$, but it is not the general definition. For $L_1$ and most other losses, you really should think "expected loss", not "loss of the expected output".

**Connection to variational inference** (in plain terms): this is the same general move as VI/VAEs in one specific sense — you can't (or don't want to) integrate out a latent exactly, so you approximate the expectation by sampling and averaging. The difference is that here the encoder stays deterministic; the only sampling is the block that represents genuinely missing information.

---

## A real side effect: marginalizing reduces variance

Loss \#2 has an unfortunate tendency: it pushes the model toward "safe" outputs.

If you minimize an expected reconstruction loss too aggressively, the model is incentivized to reduce variance:

- under $L_2$, the safest way to do well on average is to drift toward the mean (average image)
- under $L_1$, the analogous effect is drifting toward a median-type solution

That's exactly the opposite of what we want from samples: each particular realization should be sharp and digit-like.

So, strictly speaking, the objective is more like a cascade / constrained optimization:

> Minimize $\mathcal{L}_2$ (push information into $z_{\text{bottom}}$)
> …subject to $\mathcal{L}_1$ already being minimized (keep reconstructions sharp and faithful).

That's hard to do perfectly in practice, so the usual approximation is simple and effective:

- treat $\mathcal{L}_2$ as a weak regularizer
- set its weight much smaller than $\mathcal{L}_1$
- (often after a short warmup where you learn good reconstructions first)

This keeps $\mathcal{L}_2$ doing its real job — reallocating information into the deterministic context embedding — without letting it wash out the diversity that makes sampling meaningful.

---

## What "posterior" means here

I'm using "posterior" in a very practical, model-based sense:

After training on the dataset, the model defines a distribution over missing tops given the observed bottom. It's conditional on:

- the particular bottom half you feed in (the evidence)
- the learned parameters (which reflect the dataset)

So it's both:

- a conditional distribution $p(\text{top} \mid \text{bottom})$
- and a "posterior over the missing part" in the sense of: what the trained model believes is plausible, given the evidence and everything it learned from data

No explicit $p(x \mid z)$ is required for this framing. We're using a reconstruction loss as the training signal, and a Gaussian latent as the sampling interface.

---

## Inference: how you sample

Given a new bottom half:

1. $z_{\text{bottom}} = E_{\text{bottom}}(\text{bottom})$
2. Sample $\varepsilon \sim \mathcal{N}(0, I)$
3. $\hat{x} = \text{Decoder}([z_{\text{bottom}},\; \varepsilon])$

Repeat steps 2–3 to get multiple sharp completions.

Each sample should be sharp. The "averaging" only lives inside the training objective ($\mathcal{L}_2$), where it's used to integrate out unknown information and force the right separation between context and residual.

---

That's the dependent-factors upgrade to the original "Gaussian latents as a plug-compatible interface" idea.

If the factors are independent, you can often get away with just concatenation + sampling.

If they're dependent, you can still get the same usability — but you have to split the problem into "what's inferable from what you observed" (deterministic, large) and "what remains uncertain" (Gaussian, small), and train with an objective that explicitly marginalizes the missing block while keeping sharp reconstruction as the primary constraint.