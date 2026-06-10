# Reverse String

Implement an in-place GPU kernel that reverses a string of length $N$ ($N \leq 10^{10}$) without allocating additional memory for the result:

$$
roza \rightarrow azor
$$

```cpp
void ReverseDeviceStringInplace(char* str, size_t length);
```

**Important:** some tests use very large data volumes (exceeding 8 GB).
On RTX 4000 cards with limited VRAM this may not fit, so RTX A4000 with 16 GB VRAM is required.
List available GPUs and free memory with `nvidia-smi`, then select a GPU using `CUDA_VISIBLE_DEVICES`:

```bash
cd build
ninja test_reverse_string
CUDA_VISIBLE_DEVICES=2 ./test_reverse_string
```

GPU numbering in `nvidia-smi` may differ from `CUDA_VISIBLE_DEVICES` due to driver settings.
This can be resolved by passing the GPU UUID (from `nvidia-smi -L`) to `CUDA_VISIBLE_DEVICES`:

```bash
CUDA_VISIBLE_DEVICES=GPU-... ./test_reverse_string
```

GPU order inside `CUDA_VISIBLE_DEVICES` can be controlled via [`CUDA_DEVICE_ORDER`](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html#cuda-device-order).

Tests are split into two groups: small data and large data for easier debugging.
