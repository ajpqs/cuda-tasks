#include "norm.cuh"
#include <assert.h>
// Apply per-head rms_norm and add gated activation with activ_func=silu.
// Reference code:
// https://github.com/huggingface/transformers/blob/96c41be562d43785d07744924c2f1e96bc7d6884/src/transformers/models/qwen3_next/modeling_qwen3_next.py#L69
// inOut: [NUM_TOKENS, NUM_HEADS * HEAD_SIZE], dtype=fp16
// rmsNormWeight: [HEAD_SIZE], dtype=fp16
// gate: [NUM_TOKENS, NUM_HEADS * HEAD_SIZE], dtype=fp16

__device__ __forceinline__ __half2 siluf(const __half2 gate_val, const int thread_idx) {
    __half2 x_gate = gate_val;
    float2 f_x_gate = __half22float2(x_gate);
    f_x_gate.x = __fdividef(f_x_gate.x, (1.0f + __expf(-f_x_gate.x)));
    f_x_gate.y = __fdividef(f_x_gate.y, (1.0f + __expf(-f_x_gate.y)));
    x_gate = __float22half2_rn(f_x_gate);
    return x_gate;
}

__global__ void __launch_bounds__(128)
    sumKernel(const size_t numTokens, const size_t numHeads, const size_t headSize,
              __half* __restrict__ inOut, const size_t inOutStride, const __half* __restrict__ gate,
              const size_t gateStride, const __half* __restrict__ gamma, const float epsilon,
              const int active_threads) {
    const int thread_idx = threadIdx.x;
    const int token_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const size_t common_offset = headSize * head_idx;
    __shared__ float smem[32];
    __half2* inpt = reinterpret_cast<__half2*>(&inOut[token_idx * inOutStride + common_offset]);
    const __half2* gate_inpt =
        reinterpret_cast<const __half2*>(&gate[token_idx * gateStride + common_offset]);
    const __half2* gamma_inpt = reinterpret_cast<const __half2*>(gamma);
    __half2 x;
    __half2 gate_val;
    __half2 gamma_val;
    float cur_sum = 0;
    if (thread_idx < active_threads) {
        x = __ldlu(&inpt[thread_idx]);
        gate_val = __ldlu(&gate_inpt[thread_idx]);
        gamma_val = gamma_inpt[thread_idx];
        float2 f_x = __half22float2(x * x);
        cur_sum = f_x.x + f_x.y;
    }
#pragma unroll
    for (int i = 1; i <= 16; i *= 2) {
        float nex_sum = __shfl_down_sync(0xffffffff, cur_sum, i);
        cur_sum += nex_sum;
    }
    if ((thread_idx & 31) == 0) {
        smem[thread_idx >> 5] = cur_sum;
    }
    __syncthreads();
    if (thread_idx < 32) {
        int num_active_warps = (blockDim.x + 31) / 32;
        cur_sum = (thread_idx < num_active_warps) ? smem[thread_idx] : 0.0f;
#pragma unroll
        for (int i = 1; i <= 16; i *= 2) {
            float nex_sum = __shfl_down_sync(0xffffffff, cur_sum, i);
            cur_sum += nex_sum;
        }
    }
    if (thread_idx == 0) {
        smem[0] = rsqrtf(cur_sum * (1.0f / static_cast<float>(headSize)) + epsilon);
    }
    __syncthreads();
    if (thread_idx < active_threads) {
        cur_sum = smem[0];
        __half2 h2_cur_sum = __float2half2_rn(cur_sum);
        x *= h2_cur_sum;
        x *= gamma_val;
        half2 silu_outpt = siluf(gate_val, thread_idx);
        x *= silu_outpt;
        __stwt(&inpt[thread_idx], x);
    }
};

void RmsNormGated(const size_t numTokens, const size_t numHeads, const size_t headSize,
                  __half* inOut, const size_t inOutStride, const __half* gate,
                  const size_t gateStride, const __half* gamma, const float epsilon) {
    const int active_threads = headSize / 2;
    const int threads_num = ((active_threads + 31) / 32) * 32;
    assert(threads_num <= 1024);
    dim3 num_blocks = {uint32_t(numTokens), uint32_t(numHeads), 1u};
    sumKernel<<<num_blocks, threads_num>>>(numTokens, numHeads, headSize, inOut, inOutStride, gate,
                                           gateStride, gamma, epsilon, active_threads);
}
