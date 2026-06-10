#include "norm.cuh"

#include <cuda_helpers.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>
#include <random>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

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

void CosineSimilarityCheck(const std::vector<__half>& vecA, const std::vector<__half>& vecB) {
    if (vecA.size() != vecB.size()) {
        throw std::invalid_argument("Vectors must be of the same dimension.");
    }

    float dotProduct = 0.0;
    float magnitudeA = 0.0;
    float magnitudeB = 0.0;

    for (size_t i = 0; i < vecA.size(); ++i) {
        dotProduct += float(vecA[i]) * float(vecB[i]);
        magnitudeA += float(vecA[i]) * float(vecA[i]);
        magnitudeB += float(vecB[i]) * float(vecB[i]);
    }

    if (magnitudeA == 0.0 || magnitudeB == 0.0) {
        throw std::invalid_argument("One or both vectors have zero magnitude.");
    }
    magnitudeA = std::sqrt(magnitudeA);
    magnitudeB = std::sqrt(magnitudeB);

    const float normError = std::abs(magnitudeA - magnitudeB) / magnitudeA;
    REQUIRE_THAT(normError, Catch::Matchers::WithinAbs(0.001, 0.001));

    const float similarity = dotProduct / (magnitudeA * magnitudeB);
    REQUIRE_THAT(similarity, Catch::Matchers::WithinAbs(0.999, 0.002));
}

std::vector<__half> NormGateSpec(size_t totalHeads, size_t headSize,
                                 const std::vector<__half>& hiddens,
                                 const std::vector<__half>& gate, const std::vector<__half>& weight,
                                 float epsilon) {
    // Prepare output array
    std::vector<__half> result(totalHeads * headSize);
    const auto siluFn = [](auto a) {
        return static_cast<float>(a) / (1.0f + expf(static_cast<float>(-a)));
    };
    for (size_t i = 0; i < totalHeads; ++i) {
        // 1. Compute mean of squares (variance, per row)
        float sq_sum = 0.0f;
        for (size_t j = 0; j < headSize; ++j) {
            sq_sum += static_cast<float>(hiddens[i * headSize + j]) *
                      static_cast<float>(hiddens[i * headSize + j]);
        }
        float variance = sq_sum / float(headSize);

        // 2. Compute scaling factor
        float scale = 1.0f / std::sqrt(variance + epsilon);

        // 3. For each feature, apply normalization, scale and gating
        for (size_t j = 0; j < headSize; ++j) {
            float normed = static_cast<float>(hiddens[i * headSize + j]) * scale;
            float scaled = static_cast<float>(weight[j]) * normed;
            float g = siluFn(gate[i * headSize + j]);
            result[i * headSize + j] = scaled * g;
        }
    }
    return result;
}

void DoTest(size_t batchSize, size_t numHeads, size_t headSize) {
    std::mt19937_64 rng(481516);
    std::uniform_real_distribution<float> dist(0.0f, 15.0f);
    const auto generateRandomData = [&]() -> __half { return __float2half(dist(rng) / 10.0f); };

    std::vector<__half> input(batchSize * numHeads * headSize);
    std::vector<__half> gates(batchSize * numHeads * headSize);
    std::vector<__half> rmsGamma(headSize);

    std::generate(input.begin(), input.end(), generateRandomData);
    std::generate(gates.begin(), gates.end(), generateRandomData);
    std::generate(rmsGamma.begin(), rmsGamma.end(), generateRandomData);
    const float epsilon = 1e-6f;

    const auto expectedResult =
        NormGateSpec(batchSize * numHeads, headSize, input, gates, rmsGamma, epsilon);

    __half* rmsGammaDevice = AllocDeviceVector<__half>(rmsGamma.size());
    CheckStatus(cudaMemcpy(rmsGammaDevice, rmsGamma.data(), rmsGamma.size() * sizeof(__half),
                           cudaMemcpyHostToDevice));

    Matrix<__half> gatesDevice = AllocDeviceMatrix<__half>(batchSize, numHeads * headSize);
    CheckStatus(cudaMemcpy2D(reinterpret_cast<void*>(gatesDevice.pointer), gatesDevice.stride,
                             gates.data(), numHeads * headSize * sizeof(__half),
                             numHeads * headSize * sizeof(__half), batchSize,
                             cudaMemcpyHostToDevice));

    Matrix<__half> inOutDevice = AllocDeviceMatrix<__half>(batchSize, numHeads * headSize);
    CheckStatus(cudaMemcpy2D(reinterpret_cast<void*>(inOutDevice.pointer), inOutDevice.stride,
                             input.data(), numHeads * headSize * sizeof(__half),
                             numHeads * headSize * sizeof(__half), batchSize,
                             cudaMemcpyHostToDevice));

    INFO("BATCH = " << batchSize << " HEADSIZE = " << headSize << " NUMHEADS " << numHeads);

    RmsNormGated(batchSize, numHeads, headSize, inOutDevice.pointer,
                 inOutDevice.stride / sizeof(__half), gatesDevice.pointer,
                 gatesDevice.stride / sizeof(__half), rmsGammaDevice, 1e-6f);
    CheckStatus(cudaGetLastError());

    std::vector<__half> actualOut(input.size());
    CheckStatus(cudaMemcpy2D(reinterpret_cast<void*>(actualOut.data()),
                             numHeads * headSize * sizeof(__half), inOutDevice.pointer,
                             inOutDevice.stride, numHeads * headSize * sizeof(__half), batchSize,
                             cudaMemcpyDeviceToHost));

    CosineSimilarityCheck(actualOut, expectedResult);
    CheckStatus(cudaFree(rmsGammaDevice));
    CheckStatus(cudaFree(gatesDevice.pointer));
    CheckStatus(cudaFree(inOutDevice.pointer));
}

}  // namespace

TEST_CASE("RmsNormGated") {
    SECTION("Basic") {
        const auto batches = GENERATE(1, 2, 7, 16);
        const auto headsizes = GENERATE(8, 16);
        const auto heads = GENERATE(2, 13, 11, 13);

        DoTest(batches, heads, headsizes);
    }

    SECTION("Large") {
        const auto batches = GENERATE(8192, 4000);
        const auto headsizes = GENERATE(128, 256);
        const auto heads = GENERATE(32, 16);

        DoTest(batches, heads, headsizes);
    }
}
