#include "quaternions.cuh"

#include <cassert>

#include <cuda_runtime.h>

__global__ void quant_red(size_t rows, size_t cols, const Quaternion* inp, size_t inp_stride,
                          Quaternion* out) {
    extern __shared__ Quaternion smem[];
    Quaternion row_acc;
    bool is_first_tile = true;
#pragma unroll
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        Quaternion x = inp[blockIdx.x * inp_stride + i];
        for (int j = 1; j <= 16; j *= 2) {
            Quaternion next_x;
            next_x.a = __shfl_down_sync(0xffffffff, x.a, j);
            next_x.b = __shfl_down_sync(0xffffffff, x.b, j);
            next_x.c = __shfl_down_sync(0xffffffff, x.c, j);
            next_x.d = __shfl_down_sync(0xffffffff, x.d, j);
            x = QuaternionMultiplier()(x, next_x);
        }
        if (threadIdx.x % 32 == 0) {
            smem[threadIdx.x / 32] = x;
        }
        __syncthreads();
        if (threadIdx.x < 32) {
            x = smem[threadIdx.x];
            for (int j = 1; j <= 16; j *= 2) {
                Quaternion next_x;
                next_x.a = __shfl_down_sync(0xffffffff, x.a, j);
                next_x.b = __shfl_down_sync(0xffffffff, x.b, j);
                next_x.c = __shfl_down_sync(0xffffffff, x.c, j);
                next_x.d = __shfl_down_sync(0xffffffff, x.d, j);
                x = QuaternionMultiplier()(x, next_x);
            }
            if (threadIdx.x == 0) {
                if (is_first_tile) {
                    row_acc = x;
                    is_first_tile = false;
                } else {
                    row_acc = QuaternionMultiplier()(row_acc, x);
                }
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[blockIdx.x] = row_acc;
    }
}

void QuaternionsReduce(size_t rows, size_t cols, const Quaternion* inp, size_t inp_stride,
                       Quaternion* out, cudaStream_t stream) {
    int num_shared = 32 * sizeof(Quaternion);
    quant_red<<<rows, 1024, num_shared, stream>>>(rows, cols, inp, inp_stride, out);
}
