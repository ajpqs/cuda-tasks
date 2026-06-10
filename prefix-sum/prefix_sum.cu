#include "prefix_sum.cuh"

#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_IDX(n) ((n) + ((n) >> LOG_NUM_BANKS))

constexpr int num_threads_pref_sum = 512;
constexpr int num_elements_per_thread = 16;

size_t EstimatePrefixSumWorkspaceSizeBytes(size_t num_elements) {
    if (num_elements <= num_threads_pref_sum * num_elements_per_thread)
        return 0;
    int num_processed = num_elements_per_thread * num_threads_pref_sum;
    int num_blocks = (num_elements + num_processed - 1) / num_processed;
    int num_second_blocks = (num_blocks + num_processed - 1) / num_processed;
    return (num_blocks + num_second_blocks + 2) * sizeof(int);
}

__device__ __forceinline__ int warp_inclusive_scan(int x) {
    unsigned mask = 0xffffffff;
    int y = __shfl_up_sync(mask, x, 1);
    if ((threadIdx.x & 31) >= 1)
        x += y;
    y = __shfl_up_sync(mask, x, 2);
    if ((threadIdx.x & 31) >= 2)
        x += y;
    y = __shfl_up_sync(mask, x, 4);
    if ((threadIdx.x & 31) >= 4)
        x += y;
    y = __shfl_up_sync(mask, x, 8);
    if ((threadIdx.x & 31) >= 8)
        x += y;
    y = __shfl_up_sync(mask, x, 16);
    if ((threadIdx.x & 31) >= 16)
        x += y;
    return x;
}

__global__ void __launch_bounds__(512)
    prefixsum_small_1024_kernel(const int* __restrict__ input, int* __restrict__ output, size_t n) {
    __shared__ int warp_sums[16];
    int t = threadIdx.x;
    int lane = t & 31;
    int warp = t >> 5;
    int i0 = t * 2;
    int i1 = i0 + 1;
    int x0 = (i0 < n) ? input[i0] : 0;
    int x1 = (i1 < n) ? input[i1] : 0;
    int thread_sum = x0 + x1;
    int warp_scan = warp_inclusive_scan(thread_sum);
    if (lane == 31) {
        warp_sums[warp] = warp_scan;
    }
    __syncthreads();
    if (warp == 0) {
        int y = (lane < 16) ? warp_sums[lane] : 0;
        int scanned = warp_inclusive_scan(y);
        if (lane < 16) {
            warp_sums[lane] = scanned - y;
        }
    }
    __syncthreads();
    int thread_offset = warp_sums[warp] + warp_scan - thread_sum;
    if (t == 0) {
        output[0] = 0;
    }
    if (i0 < n) {
        output[i0 + 1] = thread_offset + x0;
    }
    if (i1 < n) {
        output[i1 + 1] = thread_offset + x0 + x1;
    }
}

template <bool STORE_SUM>
__global__ void __launch_bounds__(num_threads_pref_sum)
    prefixsum_kernel(const int* _input, int* output, size_t num_elements,
                     int* __restrict__ workspace) {
    const int thread_idx = threadIdx.x;
    const int block_idx = blockIdx.x;
    const int bdim = blockDim.x;
    const int cell = block_idx * bdim * num_elements_per_thread;
    const int* __restrict__ input = &_input[cell];
    const int base_smem_idx = thread_idx * num_elements_per_thread;
    extern __shared__ int smem[];

    int in_idx = thread_idx;
#pragma unroll
    for (int i = 0; i < num_elements_per_thread; ++i) {
        int idx_smem = CONFLICT_FREE_IDX(in_idx);
        if (cell + in_idx < num_elements) {
            smem[idx_smem] = input[in_idx];
        } else {
            smem[idx_smem] = 0;
        }
        in_idx += bdim;
    }
    __syncthreads();

    int local_vals[num_elements_per_thread];
    int sum_A = 0;
    int sum_B = 0;
    const int half_el = (num_elements_per_thread >> 1);
#pragma unroll
    for (int i = 0; i < half_el; ++i) {
        int smem_idx = CONFLICT_FREE_IDX(base_smem_idx + i);
        int val = smem[smem_idx];
        local_vals[i] = sum_A;
        sum_A += val;
    }
#pragma unroll
    for (int i = half_el; i < num_elements_per_thread; ++i) {
        int smem_idx = CONFLICT_FREE_IDX(base_smem_idx + i);
        int val = smem[smem_idx];
        local_vals[i] = sum_B;
        sum_B += val;
    }
    __syncthreads();
    smem[CONFLICT_FREE_IDX(thread_idx << 1)] = sum_A;
    smem[CONFLICT_FREE_IDX((thread_idx << 1) + 1)] = sum_B;
    __syncthreads();

    int pw = 2;
    for (int i = bdim; i >= 1; i >>= 1) {
        if (thread_idx < i) {
            int l = thread_idx * pw + (pw >> 1) - 1, r = thread_idx * pw + pw - 1;
            int idx_l = CONFLICT_FREE_IDX(l), idx_r = CONFLICT_FREE_IDX(r);
            int new_sum = smem[idx_l] + smem[idx_r];
            smem[idx_r] = new_sum;
            if (i == 1) {
                if (STORE_SUM) {
                    workspace[block_idx] = new_sum;
                }
                smem[idx_r] = 0;
                if (block_idx == gridDim.x - 1) {
                    output[num_elements] = new_sum;
                }
            }
        }
        pw <<= 1;
        __syncthreads();
    }
    for (int i = 1; i <= bdim; i <<= 1) {
        pw >>= 1;
        if (thread_idx < i) {
            int l = thread_idx * pw + (pw >> 1) - 1, r = thread_idx * pw + pw - 1;
            int idx_l = CONFLICT_FREE_IDX(l), idx_r = CONFLICT_FREE_IDX(r);
            int left_el = smem[idx_l];
            smem[idx_l] = smem[idx_r];
            smem[idx_r] += left_el;
        }
        __syncthreads();
    }

    int offset_A = smem[CONFLICT_FREE_IDX(thread_idx << 1)];
    int offset_B = smem[CONFLICT_FREE_IDX((thread_idx << 1) + 1)];
    __syncthreads();
    for (int i = 0; i < half_el; ++i) {
        smem[CONFLICT_FREE_IDX(base_smem_idx + i)] = local_vals[i] + offset_A;
    }
    for (int i = half_el; i < num_elements_per_thread; ++i) {
        smem[CONFLICT_FREE_IDX(base_smem_idx + i)] = local_vals[i] + offset_B;
    }
    __syncthreads();
    int global_out_idx = cell + thread_idx;
    int smem_read_idx = thread_idx;
#pragma unroll
    for (int i = 0; i < num_elements_per_thread; ++i) {
        if (global_out_idx < num_elements) {
            output[global_out_idx] = smem[CONFLICT_FREE_IDX(smem_read_idx)];
        }
        global_out_idx += bdim;
        smem_read_idx += bdim;
    }
}

__global__ void finalsum_kernel_with_agg(int* __restrict__ input, size_t num_elements,
                                         const int* __restrict__ workspace) {
    const int thread_idx = threadIdx.x;
    const int block_idx = blockIdx.x;
    const int bdim = blockDim.x;
    __shared__ int smem[1];
    if (thread_idx == 0) {
        int prev_sum = 0;
        for (int i = 0; i < block_idx; ++i) {
            prev_sum += workspace[i];
        }
        smem[0] = prev_sum;
    }
    __syncthreads();
    int prev_sum = smem[0];
    int cur_idx = (block_idx * bdim * num_elements_per_thread) + thread_idx;
#pragma unroll
    for (int i = 0; i < num_elements_per_thread; ++i) {
        if (cur_idx < num_elements) {
            input[cur_idx] += prev_sum;
        }
        cur_idx += bdim;
    }
    if (block_idx == gridDim.x - 1 && thread_idx == 0) {
        input[num_elements] += prev_sum;
    }
}

__global__ void finalsum_kernel(int* __restrict__ input, size_t num_elements,
                                const int* __restrict__ workspace) {
    const int thread_idx = threadIdx.x;
    const int block_idx = blockIdx.x;
    const int bdim = blockDim.x;
    int prev_sum = workspace[block_idx];
    int cur_idx = (block_idx * bdim * num_elements_per_thread) + thread_idx;
#pragma unroll
    for (int i = 0; i < num_elements_per_thread; ++i) {
        if (cur_idx < num_elements) {
            input[cur_idx] += prev_sum;
        }
        cur_idx += bdim;
    }
    if (block_idx == gridDim.x - 1 && thread_idx == 0) {
        input[num_elements] += prev_sum;
    }
}

void PrefixSumDevice(const int* __restrict__ input, int* __restrict__ output,
                     int* __restrict__ workspace, size_t num_elements) {
    size_t num_processed = num_elements_per_thread * num_threads_pref_sum;
    int num_blocks = (num_elements + num_processed - 1) / num_processed;
    if (num_elements <= 1024) {
        prefixsum_small_1024_kernel<<<1, 512>>>(input, output, num_elements);
        return;
    }
    if (num_elements <= num_processed) {
        int threads_min = (num_elements + num_elements_per_thread - 1) / num_elements_per_thread;
        int threads = 32;
        while (threads < threads_min)
            threads *= 2;
        num_processed = threads * num_elements_per_thread;
        int smem_size = (num_processed + num_processed / 32) * sizeof(int);
        prefixsum_kernel<false><<<1, threads, smem_size>>>(input, output, num_elements, nullptr);
        return;
    }
    int smem_size = (num_processed + num_processed / 32) * sizeof(int);
    prefixsum_kernel<true>
        <<<num_blocks, num_threads_pref_sum, smem_size>>>(input, output, num_elements, workspace);
    const int second_sum_blocks = (num_blocks + num_processed - 1) / num_processed;
    int* new_workspace = &workspace[num_blocks + 1];

    prefixsum_kernel<true><<<second_sum_blocks, num_threads_pref_sum, smem_size>>>(
        workspace, workspace, num_blocks, new_workspace);
    finalsum_kernel_with_agg<<<second_sum_blocks, num_threads_pref_sum>>>(workspace, num_blocks,
                                                                          new_workspace);
    finalsum_kernel<<<num_blocks, num_threads_pref_sum>>>(output, num_elements, workspace);
}
