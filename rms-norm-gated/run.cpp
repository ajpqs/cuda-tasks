
#include "norm.cuh"
#include <cstddef>
#include <vector>

#include <cuda_helpers.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

namespace {

template <typename T>
struct Matrix {
    size_t stride;
    T* pointer;
};

template <typename Value>
Value* AllocDeviceVector(size_t count) {
    Value* ptr_device = nullptr;

    CheckStatus(cudaMalloc(reinterpret_cast<void**>(&ptr_device), count * sizeof(Value)));

    return ptr_device;
}

template <typename T>
Matrix<T> AllocDeviceMatrix(size_t lines, size_t lineSize) {
    uint8_t* device_ptr = nullptr;
    size_t stride = 0;

    CheckStatus(cudaMallocPitch(reinterpret_cast<void**>(&device_ptr), &stride,
                                lineSize * sizeof(T), lines));

    return {.stride = stride, .pointer = reinterpret_cast<T*>(device_ptr)};
}

struct BenchmarkData {
    BenchmarkData(size_t batchSize, size_t numHeads, size_t headSize)
        : BatchSize_(batchSize), NumHeads_(numHeads), HeadSize_(headSize) {
        const size_t rowWidth = numHeads * headSize;

        std::vector<__half> input(batchSize * rowWidth);
        std::vector<__half> gates(batchSize * rowWidth);
        std::vector<__half> rmsGamma(headSize);
        for (size_t i = 0; i < input.size(); ++i) {
            input[i] = __float2half(static_cast<float>(i % 512) * 0.01f);
            gates[i] = __float2half(static_cast<float>((i + 7) % 512) * 0.01f);
        }
        for (size_t j = 0; j < headSize; ++j) {
            rmsGamma[j] = __float2half(1.0f + static_cast<float>(j % 64) * 0.01f);
        }

        RmsGammaDevice_ = AllocDeviceVector<__half>(headSize);
        CheckStatus(cudaMemcpy(RmsGammaDevice_, rmsGamma.data(), headSize * sizeof(__half),
                               cudaMemcpyHostToDevice));

        GatesDevice_ = AllocDeviceMatrix<__half>(batchSize, rowWidth);
        CheckStatus(cudaMemcpy2D(reinterpret_cast<void*>(GatesDevice_.pointer), GatesDevice_.stride,
                                 gates.data(), rowWidth * sizeof(__half), rowWidth * sizeof(__half),
                                 batchSize, cudaMemcpyHostToDevice));

        InOutDevice_ = AllocDeviceMatrix<__half>(batchSize, rowWidth);
        CheckStatus(cudaMemcpy2D(reinterpret_cast<void*>(InOutDevice_.pointer), InOutDevice_.stride,
                                 input.data(), rowWidth * sizeof(__half), rowWidth * sizeof(__half),
                                 batchSize, cudaMemcpyHostToDevice));
    }
    void DoBenchmark() {
        RmsNormGated(BatchSize_, NumHeads_, HeadSize_, InOutDevice_.pointer,
                     InOutDevice_.stride / sizeof(__half), GatesDevice_.pointer,
                     GatesDevice_.stride / sizeof(__half), RmsGammaDevice_, 1e-6f);
        CheckStatus(cudaGetLastError());
        CheckStatus(cudaDeviceSynchronize());
    }
    ~BenchmarkData() {
        CheckStatus(cudaFree(RmsGammaDevice_));
        CheckStatus(cudaFree(GatesDevice_.pointer));
        CheckStatus(cudaFree(InOutDevice_.pointer));
    }

private:
    size_t BatchSize_;
    size_t NumHeads_;
    size_t HeadSize_;
    __half* RmsGammaDevice_;
    Matrix<__half> GatesDevice_;
    Matrix<__half> InOutDevice_;
};

}  // namespace

TEST_CASE("BenchmarkRmsNormGated") {
    BenchmarkData large(8192, 32, 128);
    BenchmarkData small(100, 12, 256);

    BENCHMARK("RmsNormGatedBigKernel") {
        large.DoBenchmark();
    };
    BENCHMARK("RmsNormGatedSmallKernel") {
        small.DoBenchmark();
    };
}
