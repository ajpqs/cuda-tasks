# MoE Top-K + Expert Histogram

Same setup as **MoE Top-K** (Top-K over `__half` logits per token), plus an **expert load histogram**.

`expertHistogram[e]` stores how many times expert `e` was selected across all tokens (each token contributes `topK` selections; the same expert may appear for different tokens).

`expertHistogram` is zeroed before `MoeTopKHist` is called.

```cpp
void MoeTopKHist(size_t batchSize, size_t numExperts, size_t topK, const __half* logits,
                 size_t inputStride, int32_t* outIdxs, size_t idxsStride, __half* topkWeights,
                 size_t outStride, unsigned int* expertHistogram);
```

Logits may contain **−∞**.
