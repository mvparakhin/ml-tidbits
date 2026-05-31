# Configurable Pairwise Ranking Loss (`C_RankingLoss`)

*A single-expression, fully-batched learning-to-rank loss: a RankNet sigmoid surrogate with an optional detached LambdaRank/DCG pair weight. Toggle gain and discount independently, and take the gain from graded labels (`2 ** relevance`) or from rank position. The returned value is exactly `1 - weighted Kendall tau`, so 0 is a perfect ranking, 1 is random, and 2 is fully reversed.*

> Source: [`python/ranking/RankingLoss.py`](../python/ranking/RankingLoss.py) · Tests & training example: [`python/tests/TestRankingLoss.py`](../python/tests/TestRankingLoss.py)

---

## Table of contents

1. [What it is](#what-it-is)
2. [The objective: 1 - weighted Kendall tau](#the-objective)
3. [Gain and discount](#gain-and-discount)
4. [API](#api)
5. [Usage](#usage)
6. [Design notes](#design-notes)

---

## What it is

`C_RankingLoss` is a **learning-to-rank** objective: given model `scores` and target `labels`, it pushes the model to order the items by their labels. Both tensors are `[..., N]` -- the **last axis is one group to rank**, and any **leading axes are independent groups** (e.g. `[n_queries, n_docs]`). The result is the mean over groups.

It composes three standard ideas into one vectorized expression:

- **RankNet** -- a pairwise logistic surrogate `sigmoid(-rank_scale * (scores_i - scores_j))` applied to every pair that *should* be ordered (`labels_i > labels_j`).
- **DCG** -- the relevance gain `2 ** relevance` and the positional discount `1 / log2(rank + 2)`.
- **LambdaRank** -- each pair is weighted (detached, treated as a constant) by the change in DCG that swapping it would cause: `|gain_i - gain_j| * |discount_i - discount_j|`.

Everything is built from `[..., N, N]` broadcasts: the pairs come from a `labels_i > labels_j` comparison, and the discount/position-gain **ranks** are obtained by *counting* comparisons -- an item's rank is `(x_j > x_i).sum(...)`, the number of items that beat it. So there is **no sorting** (no `torch.sort` / `argsort`), no precomputed pair list, no buffers, and no state; the object holds only its four scalar hyperparameters, at a cost of `O(N^2)` per group (see [Design notes](#design-notes)).

## The objective

For one group, take the unordered pairs `{i, j}` with `labels_i != labels_j`, oriented so that `labels_i > labels_j`, each carrying a weight `w_ij >= 0`. Write `D` for the weighted fraction of **discordant** (inverted) pairs and `tau` for the weighted Kendall rank correlation between `scores` and `labels`. Then

```
tau = (W_concordant - W_discordant) / W_total = 1 - 2*D    =>    2*D = 1 - tau
```

The loss returns `2*D` per group, i.e. **`1 - (weighted Kendall tau)`**, averaged over the groups that have at least one orderable pair:

| ordering vs labels | tau | loss |
|---|---|---|
| perfect | +1 | **0** |
| random | 0 | **~1** |
| reversed | -1 | **2** |

For training, the hard 0/1 "is this pair inverted?" indicator is replaced by the smooth `sigmoid(-rank_scale * margin)` surrogate. Pass `hard=True` to recover the exact 0/1 count as an evaluation metric.

> **"Random ~ 1" is exact only without the discount.** When `discount=True`, the discount weight is computed from the model's *own* current ranking, so a random permutation has slightly negative weighted tau and the loss sits a little above 1. The `1 - weighted tau` identity itself always holds.

## Gain and discount

The per-pair weight is `w_ij = [gain] * [discount]`, both optional:

- **`gain="none"`** -- weight 1 on every orderable pair (plain RankNet / unweighted Kendall tau).
- **`gain="label"`** -- `gain_i = gain_base ** labels_i`. With `gain_base=2` this is the textbook DCG gain `2 ** relevance`; the label *magnitudes* set the weight.
- **`gain="position"`** -- `gain_i = gain_base ** ((N-1 - rank_i) / (N-1))`, where `rank_i` is the (tie-averaged) position of item `i` in the ideal label order. Only the label *order* matters, which is robust when label magnitudes are arbitrary.
- **`discount`** -- when True, multiply by `|1/log2(rank_i + 2) - 1/log2(rank_j + 2)|`, where `rank` is the item's position in the model's current score order (0 = top). This is the standard DCG `log2` discount; it makes the weight a LambdaRank swap importance, so top-of-list mistakes count more.

`rank_scale` controls how sharply the surrogate approximates the hard 0/1 inversion (larger = sharper).

## API

```python
C_RankingLoss(*, gain="label", discount=True, gain_base=2., rank_scale=2.)
loss(scores, labels, hard=False) -> scalar tensor
```

- `scores`, `labels`: same shape `[..., N]`; a higher label means the item should rank higher.
- `hard=False`: differentiable sigmoid surrogate (training). `hard=True`: exact `1 - weighted tau` in `[0, 2]` (evaluation; no useful gradient).
- The object is **not** an `nn.Module` (it has no parameters) -- construct it once and call it.

## Usage

```python
import torch
from RankingLoss import C_RankingLoss

# A. Textbook graded-relevance LambdaRank/DCG: 2**label gain + log2 discount
loss = C_RankingLoss(gain="label", discount=True)
scores = torch.randn(32, 10, requires_grad=True)     # 32 groups, 10 items each
labels = torch.randint(0, 5, (32, 10)).float()       # relevance grades 0..4
loss(scores, labels).backward()

# B. Position-as-gain: only the order of `labels` matters, not its magnitude
rank_only = C_RankingLoss(gain="position", gain_base=32.)
rank_only(scores, labels)

# C. Plain RankNet (uniform pair weights)
plain = C_RankingLoss(gain="none", discount=False)

# D. Honest hard metric for evaluation: 1 - weighted Kendall tau
with torch.inference_mode():
    metric = loss(scores, labels, hard=True)          # in [0, 2]; lower is better

# E. A single group is just a 1-D pair of vectors
loss(torch.tensor([1.0, 0.2, 0.5]), torch.tensor([2., 0., 1.]))
```

## Design notes

- **Tie handling (and why there is still no sort).** A rank is `(x_j > x_i).sum()` -- the count of strictly-greater items, a pure value comparison, so the inputs need not be ordered. Only *ties* need a convention. The score discount uses *ordinal* ranks: it adds a count of equal-valued *earlier* items (`eq.tril(-1).sum`, where `eq` is the equality matrix), so tied scores still get distinct ranks. Here `tril`'s `i > j` is the items' **tensor position**, not their value -- an index-based tie-break identical to a stable `argsort`, applied only to ties, which is what keeps a gradient even when all scores tie. Label-position gain instead uses *average* ranks (`0.5 * (eq.sum - 1)`), so equally-relevant items share one gain with no positional bias. For distinct values the two modes coincide.
- **Degenerate groups.** A group with all-equal labels has no orderable pair; it is excluded from the mean (not counted as a perfect 0). If no group has an orderable pair the loss is 0.
- **Dtype.** The detached weight math runs in `scores.dtype`, so passing `float64` labels does not silently promote the loss (or its gradient) to double precision.
- **Cost.** `O(N^2)` per group via dense `N x N` broadcasts -- well suited to the small-to-moderate group sizes typical of learning-to-rank. The object is stateless and `torch.compile`-friendly (the `gain`/`discount` branches are static Python).
