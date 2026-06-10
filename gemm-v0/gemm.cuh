#pragma once

#include <cuda_fp16.h>

#include <cuda_helpers.h>

#include <assert.h>

#include <vector>
#include <string>
#include <stdexcept>
#include <cuda_helpers.h>

#define CUDA_CHECK(call)                                                                     \
    do {                                                                                     \
        cudaError_t err = call;                                                              \
        if (err != cudaSuccess) {                                                            \
            throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(err) + \
                                     " at " + __FILE__ + ":" + std::to_string(__LINE__));    \
        }                                                                                    \
    } while (0)

enum class MatrixLayout { RowMajor, ColMajor };

struct DeviceMatrix {
    __half* data;
    size_t rows;
    size_t cols;
    size_t stride;  // Distance in elements between first values of consecutive rows/columns
    MatrixLayout layout;
};

__device__ __inline__ __half& get_val(const DeviceMatrix& a, size_t row, size_t col) {
    assert(row < a.rows);
    assert(col < a.cols);
    if (a.layout == MatrixLayout::RowMajor) {
        return a.data[row * a.stride + col];
    }
    return a.data[col * a.stride + row];
}

__device__ __inline__ float mult_num(const DeviceMatrix& a, const DeviceMatrix& b, size_t m,
                                     size_t k, size_t n) {
    __half a_num = get_val(a, m, k);
    __half b_num = get_val(b, k, n);
    return a_num * b_num;
}

__global__ void mat_mull(const DeviceMatrix a, const DeviceMatrix b, const DeviceMatrix c,
                         DeviceMatrix d, float alpha, float beta) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx >= a.rows || idy >= b.cols)
        return;
    float res = 0;
    for (int i = 0; i < a.cols; ++i) {
        res += mult_num(a, b, idx, i, idy);
    }
    float c_val = get_val(c, idx, idy);
    res = alpha * res + beta * c_val;
    get_val(d, idx, idy) = res;
}

void GEMM(const DeviceMatrix& a, const DeviceMatrix& b, const DeviceMatrix& c, DeviceMatrix& d,
          float alpha, float beta) {
    int threadsPerBlockx = 16;
    int blocksPerGridx = (a.rows + threadsPerBlockx - 1) / threadsPerBlockx;
    int threadsPerBlocky = 16;
    int blocksPerGridy = (b.cols + threadsPerBlocky - 1) / threadsPerBlocky;
    dim3 griddim(blocksPerGridx, blocksPerGridy);
    dim3 threaddim(threadsPerBlockx, threadsPerBlocky);
    mat_mull<<<griddim, threaddim>>>(a, b, c, d, alpha, beta);
    CUDA_CHECK(cudaDeviceSynchronize());
}
