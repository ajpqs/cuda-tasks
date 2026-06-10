#include <cuda_fp16.h>

#include <cuda_helpers.h>

constexpr int threadsPerBlockx = 16;
constexpr int threadsPerBlocky = 16;

__global__ void transpose_kernel(const __half* input_device, size_t input_stride,
                                 __half* output_device, size_t output_stride, size_t num_rows,
                                 size_t num_cols) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t idy = blockIdx.y * blockDim.y + threadIdx.y;
    __shared__ __half Csmem[threadsPerBlocky][threadsPerBlockx + 1];
    if (idy < num_rows && idx < num_cols)
        Csmem[threadIdx.y][threadIdx.x] = input_device[input_stride * idy + idx];
    __syncthreads();
    if (blockIdx.x * blockDim.x + threadIdx.y < num_cols &&
        blockIdx.y * blockDim.y + threadIdx.x < num_rows)
        output_device[output_stride * (blockIdx.x * blockDim.x + threadIdx.y) +
                      (blockIdx.y * blockDim.y + threadIdx.x)] = Csmem[threadIdx.x][threadIdx.y];
}

void TransposeDevice(const __half* input_device, size_t input_stride, __half* output_device,
                     size_t output_stride, size_t num_rows, size_t num_cols) {
    int blocksPerGridx = (num_cols + threadsPerBlockx - 1) / threadsPerBlockx;
    int blocksPerGridy = (num_rows + threadsPerBlocky - 1) / threadsPerBlocky;
    dim3 griddim(blocksPerGridx, blocksPerGridy);
    dim3 threaddim(threadsPerBlockx, threadsPerBlocky);
    transpose_kernel<<<griddim, threaddim>>>(input_device, input_stride, output_device,
                                             output_stride, num_rows, num_cols);
}
