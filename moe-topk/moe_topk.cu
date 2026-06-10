#include "moe_topk.cuh"
#define INF __float2half(65504.0f)
// MoE router: for each token take Top-K experts by probability.
//
// logits:        [batchSize, numExperts], row-major, stride in elements (inputStride)
// outIdxs:  [batchSize, topK], int32 expert ids, stride idxsStride
// topkWeights:  [batchSize, topK], __half, stride outStride
//
// Tie-breaking when probabilities are equal: smaller expert index is preferred.
constexpr int maxThreadsPerBlock = 32;
__global__ void __launch_bounds__(maxThreadsPerBlock)
    topk_kernel(size_t numExperts, size_t topK, const __half* logits, size_t inputStride,
                int32_t* outIdxs, size_t idxsStride, __half* topkWeights, size_t outStride) {
    const auto inpt_logits = (&logits[inputStride * blockIdx.x]);
    __half x[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        x[i] = -INF;
    }
    for (int i = 0; i < (numExperts + 31) / 32; i += 1) {
        if (i * 32 + threadIdx.x < numExperts)
            x[i] = inpt_logits[i * 32 + threadIdx.x];
    }
    __half cur_max = x[0];
    int cur_idx = threadIdx.x;
#pragma unroll
    for (int i = 1; i < 8; ++i) {
        if (x[i] > cur_max) {
            cur_max = x[i];
            cur_idx = threadIdx.x + i * 32;
        }
    }
    for (int i = 0; i < topK; ++i) {
        __half next_max = cur_max;
        int next_idx = cur_idx;
        for (int j = 1; j <= 16; j *= 2) {
            __half _next_max = __shfl_down_sync(0xffffffff, next_max, j);
            int _next_idx = __shfl_down_sync(0xffffffff, next_idx, j);
            if (_next_max > next_max || (_next_max == next_max && _next_idx < next_idx)) {
                next_max = _next_max;
                next_idx = _next_idx;
            }
        }
        next_max = __shfl_sync(0xffffffff, next_max, 0);
        next_idx = __shfl_sync(0xffffffff, next_idx, 0);
        if (threadIdx.x == 0) {
            outIdxs[idxsStride * blockIdx.x + i] = next_idx;
            topkWeights[outStride * blockIdx.x + i] = next_max;
        }
        if (next_idx == cur_idx) {
            x[cur_idx / 32] = -INF;
            cur_max = x[0];
            cur_idx = threadIdx.x;
#pragma unroll
            for (int i = 1; i < 8; ++i) {
                if (x[i] > cur_max) {
                    cur_max = x[i];
                    cur_idx = threadIdx.x + i * 32;
                }
            }
        }
    }
}
void MoeTopK(size_t batchSize, size_t numExperts, size_t topK, const __half* logits,
             size_t inputStride, int32_t* outIdxs, size_t idxsStride, __half* topkWeights,
             size_t outStride) {
    topk_kernel<<<batchSize, 32>>>(numExperts, topK, logits, inputStride, outIdxs, idxsStride,
                                   topkWeights, outStride);
}
