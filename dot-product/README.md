# Dot Product

Write a kernel that computes the dot product of two vectors:

$$
A \cdot B = \sum_{i}^{N}{A_iB_i}, \text{ where } N \leq 10^6
$$

Only shared memory should be used. Partial sums from individual blocks are written to a `workspace` array, then reduced by a separate kernel.
The testing code allocates `workspace` memory; its size is determined by your implementation of `EstimateDotProductWorkspaceSizeBytes`.

Input pointers are guaranteed to be 256-byte aligned.

**NB:** this task introduces benchmarks in the testing system.
Run them locally by building and executing the `bench_dot_product` binary.
To pass benchmarks, your solution must run within 110% of the reference solution's time.

## Useful links
- [Using shared memory in CUDA C/C++](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)
- [CUDA Pro Tip: Increase Performance with Vectorized Memory Access](https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/)
