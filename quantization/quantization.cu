#include "quantization.cuh"

inline __device__ float4 operator*(const float4& a, const float4& b) {
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

inline __device__ float4 operator*(const float4& a, const float& b) {
    return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
}

inline __device__ float4 operator+(const float4& a, const float4& b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

inline __device__ float4 take_abs(const float4& a) {
    return make_float4(fabsf(a.x), fabsf(a.y), fabsf(a.z), fabsf(a.w));
}

inline __device__ char4 quant(const float4& a) {
    return make_char4(
        static_cast<int8_t>(__float2int_rn(a.x)), static_cast<int8_t>(__float2int_rn(a.y)),
        static_cast<int8_t>(__float2int_rn(a.z)), static_cast<int8_t>(__float2int_rn(a.w)));
}

inline __device__ float max_element(const float4& a) {
    return fmaxf(fmaxf(a.x, a.y), fmaxf(a.z, a.w));
}

__global__ void quantization(size_t rows, size_t cols, const float* d_input_matrix,
                             const float* d_balance_factors, size_t input_stride, size_t out_stride,
                             int8_t* d_out, float* d_out_scales) {
    extern __shared__ float smem[];
    float4* smem_float4 = reinterpret_cast<float4*>(smem);
    int row = blockIdx.x;
    float cur_max = 1e-5f;
    for (int i = threadIdx.x; i < cols / 4; i += blockDim.x) {

        float4 inpt_mat = reinterpret_cast<const float4*>(d_input_matrix + row * input_stride)[i];
        float4 inpt_bal = reinterpret_cast<const float4*>(d_balance_factors)[i];

        float4 x = inpt_mat + inpt_bal;
        smem_float4[i] = x;
        float4 y = take_abs(x);
        cur_max = fmaxf(cur_max, max_element(y));
    }

    float* smem_max = &smem[cols];
    for (int i = 1; i <= 16; i *= 2) {
        cur_max = fmaxf(cur_max, __shfl_down_sync(0xffffffff, cur_max, i));
    }
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    if (lane_id == 0) {
        smem_max[warp_id] = cur_max;
    }
    __syncthreads();
    int warps_count = blockDim.x / 32;
    if (threadIdx.x < 32) {
        cur_max = (threadIdx.x < warps_count) ? smem_max[threadIdx.x] : 0.f;

        for (int i = 1; i <= 16; i *= 2) {
            cur_max = fmaxf(cur_max, __shfl_down_sync(0xffffffff, cur_max, i));
        }

        if (threadIdx.x == 0) {
            constexpr float mScale = 1e-5f;
            const float cur_scale = 127.f / fmaxf(mScale, cur_max);  // fmaxf для float
            smem_max[0] = cur_scale;
            d_out_scales[row] = cur_scale;
        }
    }
    __syncthreads();
    float final_scale = smem_max[0];
    for (int i = threadIdx.x; i < cols / 4; i += blockDim.x) {
        char4 res = quant(smem_float4[i] * final_scale);
        reinterpret_cast<char4*>(d_out + row * out_stride)[i] = res;
    }
}

void Quantization(size_t rows, size_t cols, const float* d_input_matrix,
                  const float* d_balance_factors, size_t input_stride, size_t out_stride,
                  int8_t* d_out, float* d_out_scales) {
    int threads_num = min(1024l, (((cols / 4) + 31) / 32) * 32);
    int warps_num = threads_num / 32;
    int smem_size = ((warps_num > 0 ? warps_num : 1) + cols) * sizeof(float);

    quantization<<<rows, threads_num, smem_size>>>(rows, cols, d_input_matrix, d_balance_factors,
                                                   input_stride, out_stride, d_out, d_out_scales);
}
