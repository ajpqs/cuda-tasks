#pragma once

#include <cstddef>
#include <cuda/std/utility>
#include <cuda_helpers.h>

__global__ void rev_str(char* str, size_t length) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= length / 2)
        return;
    cuda::std::swap(str[idx], str[length - idx - 1]);
}

void ReverseDeviceStringInplace(char* str, size_t length) {
    int threadsPerBlock = 256;
    int blocksPerGrid = max((length / 2 + threadsPerBlock - 1) / threadsPerBlock, 1l);
    rev_str<<<blocksPerGrid, threadsPerBlock>>>(str, length);
    cudaDeviceSynchronize();
}
