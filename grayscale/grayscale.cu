#include "grayscale.cuh"

Image AllocHostImage(size_t width, size_t height, size_t channels) {
    uint8_t* h_data;
    cudaMallocHost((void**)&h_data, width * height * channels * sizeof(uint8_t));
    return Image{h_data, width, height, width * channels * sizeof(uint8_t), channels};
}

Image AllocDeviceImage(size_t width, size_t height, size_t channels) {
    uint8_t* h_data;
    size_t stride;
    cudaMallocPitch(&h_data, &stride, width * channels * sizeof(uint8_t), height);
    return Image{h_data, width, height, stride, channels};
}

void CopyImageHostToDevice(const Image& src_host, Image& dst_device) {
    cudaMemcpy2D(dst_device.pixels, dst_device.stride, src_host.pixels, src_host.stride,
                 src_host.width * src_host.channels * sizeof(uint8_t), src_host.height,
                 cudaMemcpyHostToDevice);
}

void CopyImageDeviceToHost(const Image& src_device, Image& dst_host) {
    cudaMemcpy2D(dst_host.pixels, dst_host.stride, src_device.pixels, src_device.stride,
                 src_device.width * src_device.channels * sizeof(uint8_t), src_device.height,
                 cudaMemcpyDeviceToHost);
}

__global__ void kr(const Image rgb_image, Image gray_image) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx >= gray_image.height || idy >= gray_image.width)
        return;
    float coeffs[] = {0.299f, 0.587f, 0.114f};
    float res = 0;
    for (int i = 0; i < 3; ++i)
        res +=
            rgb_image.pixels[idx * (rgb_image.stride / sizeof(uint8_t)) + idy * 3 + i] * coeffs[i];
    gray_image.pixels[idx * (gray_image.stride / sizeof(uint8_t)) + idy] = res;
}

void ConvertToGrayscaleDevice(const Image& rgb_device_image, Image& gray_device_image) {
    int threadsPerBlockx = 16;
    int blocksPerGridx = (gray_device_image.height + threadsPerBlockx - 1) / threadsPerBlockx;
    int threadsPerBlocky = 16;
    int blocksPerGridy = (gray_device_image.width + threadsPerBlocky - 1) / threadsPerBlocky;
    dim3 griddim(blocksPerGridx, blocksPerGridy);
    dim3 threaddim(threadsPerBlockx, threadsPerBlocky);
    kr<<<griddim, threaddim>>>(rgb_device_image, gray_device_image);
}

void FreeDeviceImage(const Image& image) {
    cudaFree(image.pixels);
}

void FreeHostImage(const Image& image) {
    cudaFreeHost(image.pixels);
}
