#include "moe_topk_hist.cuh"
#include <math_constants.h>
// Same as MoeTopK plus expertHistogram[e] = count of assignments to expert e.
// Zero expertHistogram (length numExperts) before calling.
#define MINUS_INF __float2half(-CUDART_INF_F)
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
                int32_t* outIdxs, size_t idxsStride, __half* topkWeights, size_t outStride,
                unsigned int* expertHistogram) {
    const auto inpt_logits = (&logits[inputStride * blockIdx.x]);
    const int tid = threadIdx.x;
    __half x[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        x[i] = MINUS_INF;
    }
#pragma unroll
    for (int i = 0; i < 8; i += 1) {
        if (i * 32 + tid < numExperts)
            x[i] = inpt_logits[i * 32 + tid];
    }
    __half cur_max = x[0];
    int cur_idx = tid;
#pragma unroll
    for (int i = 1; i < 8; ++i) {
        if (x[i] > cur_max) {
            cur_max = x[i];
            cur_idx = tid + i * 32;
        }
    }
    for (int i = 0; i < topK; ++i) {
        __half next_max = cur_max;
        int next_idx = cur_idx;
#pragma unroll
        for (int j = 1; j <= 16; j *= 2) {
            __half _next_max = __shfl_down_sync(0xffffffff, next_max, j);
            int _next_idx = __shfl_down_sync(0xffffffff, next_idx, j);
            if (_next_max > next_max || (_next_max == next_max && _next_idx < next_idx)) {
                next_max = _next_max;
                next_idx = _next_idx;
            }
        }
        next_idx = __shfl_sync(0xffffffff, next_idx, 0);
        if (tid == 0) {
            outIdxs[idxsStride * blockIdx.x + i] = next_idx;
            topkWeights[outStride * blockIdx.x + i] = next_max;
            atomicAdd(&expertHistogram[next_idx], 1);
        }
        if (next_idx == cur_idx) {
            x[cur_idx >> 5] = MINUS_INF;
            cur_max = x[0];
            cur_idx = tid;
#pragma unroll
            for (int i = 1; i < 8; ++i) {
                if (x[i] > cur_max) {
                    cur_max = x[i];
                    cur_idx = tid + i * 32;
                }
            }
        }
    }
}

void MoeTopKHist(size_t batchSize, size_t numExperts, size_t topK, const __half* logits,
                 size_t inputStride, int32_t* outIdxs, size_t idxsStride, __half* topkWeights,
                 size_t outStride, unsigned int* expertHistogram) {
    topk_kernel<<<batchSize, 32>>>(numExperts, topK, logits, inputStride, outIdxs, idxsStride,
                                   topkWeights, outStride, expertHistogram);
}
