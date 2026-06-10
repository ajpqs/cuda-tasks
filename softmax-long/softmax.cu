#include "softmax.cuh"

#include <cuda_runtime.h>
#include <float.h>

constexpr int maxThreadsPerBlock = 1024;

__device__ __forceinline__ void upd(float& cur_sum, float& cur_max, float& new_max,
                                    const float4& x) {
    new_max = fmaxf(fmaxf(fmaxf(x.x, x.y), fmaxf(x.z, x.w)), cur_max);
    cur_sum = fmaf(cur_sum, __expf(cur_max - new_max),
                   __expf(x.x - new_max) + __expf(x.y - new_max) + __expf(x.z - new_max) +
                       __expf(x.w - new_max));
    cur_max = new_max;
}
__global__ void __launch_bounds__(maxThreadsPerBlock)
    softmax(size_t rows, size_t cols, const float* d_input_matrix, size_t input_stride,
            float* d_out, size_t out_stride) {
    size_t thread_idx = threadIdx.x;
    const float4* inpt =
        reinterpret_cast<const float4*>(&d_input_matrix[blockIdx.x * input_stride]);
    float cur_sum = 0, cur_max = __int_as_float(0xFF7FFFFF), new_max = __int_as_float(0xFF7FFFFF);
    for (int i = thread_idx; i < cols / 4; i += blockDim.x) {
        float4 x = inpt[i];
        upd(cur_sum, cur_max, new_max, x);
    }
#pragma unroll
    for (int i = 1; i <= 16; i *= 2) {
        float next_max, next_sum;
        next_max = __shfl_down_sync(0xffffffff, cur_max, i);
        next_sum = __shfl_down_sync(0xffffffff, cur_sum, i);
        new_max = fmaxf(cur_max, next_max);
        cur_sum = fmaf(cur_sum, __expf(cur_max - new_max), next_sum * __expf(next_max - new_max));
        cur_max = new_max;
    }
    extern __shared__ float smem[];
    if (thread_idx % 32 == 0) {
        smem[thread_idx / 32] = cur_max;
        smem[32 + thread_idx / 32] = cur_sum;
    }
    __syncthreads();
    if (thread_idx < 32) {
        cur_sum = 0.0f;
        cur_max = __int_as_float(0xFF7FFFFF);
        if (thread_idx < blockDim.x / 32) {
            cur_max = smem[thread_idx];
            cur_sum = smem[32 + thread_idx];
        }
#pragma unroll
        for (int i = 1; i <= 16; i *= 2) {
            float next_max = __shfl_down_sync(0xffffffff, cur_max, i);
            float next_sum = __shfl_down_sync(0xffffffff, cur_sum, i);

            float new_max = fmaxf(cur_max, next_max);
            cur_sum =
                fmaf(cur_sum, __expf(cur_max - new_max), next_sum * __expf(next_max - new_max));
            cur_max = new_max;
        }
        if (thread_idx == 0) {
            smem[0] = cur_max;
            smem[1] = cur_sum;
        }
    }
    __syncthreads();
    cur_max = smem[0];
    cur_sum = smem[1];
    float4* outpt = reinterpret_cast<float4*>(&d_out[blockIdx.x * out_stride]);
    const float inv_sum = 1.0f / cur_sum;
    for (int i = thread_idx; i < cols / 4; i += blockDim.x) {
        float4 x = inpt[i];
        x.x = __expf(x.x - cur_max) * inv_sum;
        x.y = __expf(x.y - cur_max) * inv_sum;
        x.z = __expf(x.z - cur_max) * inv_sum;
        x.w = __expf(x.w - cur_max) * inv_sum;
        outpt[i] = x;
    }
}

void Softmax(size_t rows, size_t cols, const float* d_input_matrix, size_t input_stride,
             float* d_out, size_t out_stride, cudaStream_t stream) {
    // YOUR CODE HERE
    // NB: no need to do any allocations here
    // NB: no explicit cudaDeviceSynchronize is required here
    size_t smem_sz = sizeof(float) * 32 * 2;
    softmax<<<rows, maxThreadsPerBlock, smem_sz, stream>>>(rows, cols, d_input_matrix, input_stride,
                                                           d_out, out_stride);
}
