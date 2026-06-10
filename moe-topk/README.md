# MoE Top-K

In **Mixture of Experts (MoE)** layers, a router produces a score (logit) for each token-expert pair.
Typically, `softmax` is applied, the **Top-K** experts are selected (commonly `K=8`), and weights are renormalized — as in [Mixtral](https://huggingface.co/docs/transformers/model_doc/mixtral#transformers.MixtralSparseMoeBlock).
**This task covers only the expert selection step:** given a matrix of **fp16** scores, find the `K` largest elements per row (ties broken by smaller index), and output their **indices** and **values**.

## Specification

For each token `t` (row of the logit matrix):

1. Among all experts `e ∈ [0, numExperts)`, select `topK` distinct indices with the **largest** `logits[t][e]` values. At each step, pick the current maximum among unselected experts.
2. On **ties**, prefer the expert with the **smaller** index.
3. Write indices to `outIdxs[t, :]` and corresponding logit values to `topkWeights[t, :]`.

Data format is **row-major**, strides are given **in elements** (as in the quantization task).
Implement the function (declaration in `moe_topk.cuh`):

```cpp
void MoeTopK(size_t batchSize, size_t numExperts, size_t topK, const __half* logits,
             size_t inputStride, int32_t* outIdxs, size_t idxsStride,
             __half* topkWeights, size_t outStride);
```

Only kernel launches (and `__device__` helper functions) are allowed; no separate allocations or explicit `cudaDeviceSynchronize` are required.
Guaranteed: logits contain no inf, -inf, or NaN; `topK ≤ numExperts`; `numExperts > 0 && numExperts <= 256`.
