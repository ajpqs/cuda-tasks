#pragma once

#include <iostream>
#include <cassert>
#include <cuda_fp16.h>
#include <cuda_helpers.h>

enum class MatrixLayout { RowMajor, ColMajor };

struct DeviceMatrix {
    __half* data;
    size_t rows;
    size_t cols;
    size_t stride;
    MatrixLayout layout;
};

#define CUDA_CHECK(call)                                                                     \
    do {                                                                                     \
        cudaError_t err = call;                                                              \
        if (err != cudaSuccess) {                                                            \
            throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(err) + \
                                     " at " + __FILE__ + ":" + std::to_string(__LINE__));    \
        }                                                                                    \
    } while (0)

template <int BM, int BN, int BP>
__global__ void GemmDevice(size_t m, size_t n, size_t p, const __half* a, size_t aStride,
                           const __half* b, size_t bStride, const __half* c, size_t cStride,
                           __half* d, size_t dStride, const float alpha, const float beta) {
    static_assert(BM == BN);
    static_assert(BM % BP == 0);
    static constexpr int ThreadTile = BN / BP;
    assert(blockDim.x == BM * BP);

    // BP + 2 provides mathematically perfect misalignment for 16-bit datatypes
    __shared__ __half aSmem[BM][BP + 2];
    __shared__ __half bSmem[BN][BP];

    float accum[ThreadTile] = {};

    const size_t nIters = (p + BP - 1) / BP;
    for (size_t iter = 0; iter < nIters; ++iter) {

        size_t aGlobalI = blockIdx.x * BM + (threadIdx.x / BP);
        size_t aP = iter * BP + (threadIdx.x % BP);
        if (aGlobalI < m && aP < p) {
            aSmem[threadIdx.x / BP][threadIdx.x % BP] = a[aGlobalI * aStride + aP];
        } else {
            aSmem[threadIdx.x / BP][threadIdx.x % BP] = __float2half(0.0f);
        }

        size_t bP = iter * BP + (threadIdx.x % BP);
        size_t bGlobalJ = blockIdx.y * BN + (threadIdx.x / BP);
        if (bP < p && bGlobalJ < n) {
            bSmem[threadIdx.x / BP][threadIdx.x % BP] = b[bGlobalJ * bStride + bP];
        } else {
            bSmem[threadIdx.x / BP][threadIdx.x % BP] = __float2half(0.0f);
        }

        __syncthreads();

// --- Phase 1: Math Loop ---
#pragma unroll
        for (int localP = 0; localP < BP; ++localP) {
            // 1. Keep as __half
            __half aVal = aSmem[threadIdx.x % BM][localP];

#pragma unroll
            for (int accumIdx = 0; accumIdx < ThreadTile; ++accumIdx) {
                // 1. Keep as __half
                __half bVal = bSmem[(threadIdx.x / BN) * ThreadTile + accumIdx][localP];

                // 2. Multiply in __half (matching V0 truncation), THEN cast to float for
                // accumulation
                accum[accumIdx] += __half2float(bVal * aVal);
            }
        }
        __syncthreads();
    }

    size_t j = blockIdx.x * BM + (threadIdx.x % BM);

#pragma unroll
    for (int accumIdx = 0; accumIdx < ThreadTile; ++accumIdx) {
        size_t i = blockIdx.y * BN + (threadIdx.x / BN) * ThreadTile + accumIdx;

        if (j < m && i < n) {
            float cVal = __half2float(c[i * cStride + j]);
            float finalAns = (alpha * accum[accumIdx]) + (beta * cVal);
            d[i * dStride + j] = __float2half(finalAns);
        }
    }
}

template <int BM, int BN, int BP>
void DoRun(const DeviceMatrix a, const DeviceMatrix b, const DeviceMatrix c, DeviceMatrix d,
           float alpha, float beta) {
    size_t M = d.rows;
    size_t N = d.cols;
    size_t P = a.cols;
    static_assert(BM == BN);
    static_assert(BN % BP == 0);
    static_assert(BM * BP <= 1024);
    dim3 block(BM * BP);
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
    // std::cout << "Testing [" << BM << ", " << BN << "] blocks with BP = " << BP << std::endl;
    // std::cout << "Launch " << grid.x * grid.y << " blocks" << std::endl;

    // {
    //     int numBlocksPerSm = 0;
    //     CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    //         &numBlocksPerSm,
    //         GemmDevice<BM, BN, BP>,
    //         static_cast<int>(block.x * block.y * block.z),
    //         /* dynamicSmemSizeInBytes = */ 0));

    //     size_t threadsPerBlock = block.x * block.y * block.z;
    //     size_t numThreadsPerSm = numBlocksPerSm * threadsPerBlock;
    //     float occupancy = 100.f * numThreadsPerSm / deviceProp.maxThreadsPerMultiProcessor;

    //     std::cout << "We have " << numBlocksPerSm << " blocks per SM" << std::endl;
    //     std::cout << "Occupancy = " << occupancy << "%" << std::endl;

    //     size_t blocksInGrid = 1ll * grid.x * grid.y * grid.z;
    //     size_t numBlocksPerWave = numBlocksPerSm * deviceProp.multiProcessorCount;
    //     size_t numWaves = (blocksInGrid + numBlocksPerWave - 1) / numBlocksPerWave;

    //     std::cout << "We have " << numWaves << " waves" << std::endl;
    // }

    GemmDevice<BM, BN, BP><<<grid, block>>>(M, N, P, a.data, a.stride, b.data, b.stride, c.data,
                                            c.stride, d.data, d.stride, alpha, beta);
}

void GEMM(const DeviceMatrix& a, const DeviceMatrix& b, const DeviceMatrix& c, DeviceMatrix& d,
          float alpha, float beta) {
    assert(a.layout == MatrixLayout::RowMajor);
    assert(b.layout == MatrixLayout::ColMajor);
    assert(c.layout == MatrixLayout::ColMajor);
    assert(d.layout == MatrixLayout::ColMajor);

    DoRun<64, 64, 16>(a, b, c, d, alpha, beta);
}
