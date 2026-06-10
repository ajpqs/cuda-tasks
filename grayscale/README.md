# Grayscale

Implement a GPU function that converts a 3-channel RGB color image to a single-channel grayscale image.
The image is stored as a 2D array in HWC format, where red $R$, green $G$, and blue $B$ channel values are laid out consecutively for each pixel:

$$
\begin{matrix}
    R_{00} & G_{00} & B_{00} & R_{01} & G_{01} & B_{01} \\
    R_{10} & G_{10} & B_{10} & R_{11} & G_{11} & B_{11}
\end{matrix}
$$

The conversion formula:
$$
Y = 0.299 \times R + 0.587 \times G + 0.114 \times B
$$

Implement the following helper functions for memory management and data transfer:

```cpp
Image AllocHostImage(size_t width, size_t height, size_t channels);
Image AllocDeviceImage(size_t width, size_t height, size_t channels);
void CopyImageHostToDevice(const Image& src_host, Image& dst_device);
void CopyImageDeviceToHost(const Image& src_device, Image& dst_host);
void ConvertToGrayscaleDevice(const Image& rgb_device_image, Image& gray_device_image);
void FreeDeviceImage(const Image& image);
void FreeHostImage(const Image& image);
```

The `ConvertToGrayscaleDevice` function receives a pre-allocated output image — no allocations inside.
The function is expected to be asynchronous, so no explicit synchronization at the end.

Note that image rows may be separated by padding for alignment purposes.
The parameter `stride` refers to the byte distance between consecutive row starts.
In CUDA API this is often called `pitch` — see `cudaMallocPitch` and `cudaMemcpy2D`.

## Useful links

- [OpenCV: Color conversions](https://docs.opencv.org/4.10.0/de/d25/imgproc_color_conversions.html)
- [CUDA Toolkit Documentation: Memory management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY)
- [How to Optimize Data Transfers in CUDA C/C++](https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/)
