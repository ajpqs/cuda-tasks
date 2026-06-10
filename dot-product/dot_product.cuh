#pragma once

#include <cuda_helpers.h>
#include <cuda_fp16.h>

#include <cuda_helpers.h>

#include <assert.h>

#include <vector>
#include <string>
#include <stdexcept>
#include <cuda_helpers.h>

constexpr int threadsPerBlock = 256;

#define CUDA_CHECK(call)                                                                     \
    do {                                                                                     \
        cudaError_t err = call;                                                              \
        if (err != cudaSuccess) {                                                            \
            throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(err) + \
                                     " at " + __FILE__ + ":" + std::to_string(__LINE__));    \
        }                                                                                    \
    } while (0)

size_t EstimateDotProductWorkspaceSizeBytes(size_t num_elements) {
    int blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
    return blocksPerGrid * sizeof(float);
}

__global__ void dot_kernel(const float* lhs_device, const float* rhs_device, size_t num_elements,
                           float* workspace_device) {
    __shared__ float s_sum[threadsPerBlock];
    size_t idx = threadIdx.x;
    float acc = 0;
    if (blockIdx.x * blockDim.x + idx < num_elements)
        acc = lhs_device[blockIdx.x * blockDim.x + idx] * rhs_device[blockIdx.x * blockDim.x + idx];
    int margin = threadsPerBlock / 2;
    while (margin > 0) {
        if (idx >= margin) {
            s_sum[idx] = acc;
        }
        __syncthreads();
        if (idx < margin) {
            acc += s_sum[idx + margin];
        }
        margin /= 2;
    }
    if (idx == 0)
        workspace_device[blockIdx.x] = acc;
};

__global__ void final_red(float* workspace_device, size_t num_elements, float* out_device) {
    __shared__ float s_sum[threadsPerBlock];
    size_t idx = threadIdx.x;  // one block
    s_sum[idx] = 0;
    float acc = 0;
    for (int i = 0; i < (num_elements + threadsPerBlock - 1) / threadsPerBlock; ++i) {
        if (i * threadsPerBlock + idx < num_elements)
            acc += workspace_device[i * threadsPerBlock + idx];
    }
    int margin = threadsPerBlock / 2;
    while (margin > 0) {
        if (idx >= margin) {
            s_sum[idx] = acc;
        }
        __syncthreads();
        if (idx < margin) {
            acc += s_sum[idx + margin];
        }
        margin /= 2;
    }
    if (threadIdx.x == 0)
        out_device[0] = acc;
};

void DotProduct(const float* lhs_device, const float* rhs_device, size_t num_elements,
                float* workspace_device, float* out_device) {
    int blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
    dot_kernel<<<blocksPerGrid, threadsPerBlock>>>(lhs_device, rhs_device, num_elements,
                                                   workspace_device);
    final_red<<<1, threadsPerBlock>>>(workspace_device, blocksPerGrid, out_device);
}
