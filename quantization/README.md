# Quantization

Quantization reduces the bit-width of model parameters to decrease memory footprint and computation costs without significant quality loss.

This task implements one sub-problem of model quantization: quantizing a linear layer's weight matrix.

Quantize the weight matrix `W` with balance factors `S` from `float` to `int8_t` using the formula:

$$
W'_{r'c'} = \operatorname{RoundNearest}\left(
\frac{(W_{r'c'} + S_{c'}) \cdot 127}
{\max\left(mScale, \max_{c}\left|W_{r'c} + S_c\right|\right)}
\right)
$$

where `W'` is the quantized `int8_t` matrix, `W` is the original `float` matrix, `S` is a `float` balance array, and `mScale = 1e-5`.

Example:
```
            W                    S                        W'                     Scales
       1.0 0.1 -0.08          0.2 0 1      =>            127 11 97                105.83333
      -3.3 -0.3 3                                      -98 -10 127               31.75
```

## Required functions

```cpp
void Quantization(size_t rows, size_t cols, const float* d_inputMatrix, const float* d_balanceFactors, size_t inputStride, size_t outStride, int8_t* d_out, float* d_outScales);
```

- `rows, cols` — dimensions of the row-major matrix
- `inputStride, outStride` — distance (in elements) between rows; may exceed `cols` due to alignment
- `d_outScales` — array of length `rows` storing the scale for each row:

$$
scale_r = \frac{\max(mScale, \max_c |W_{rc} + S_c|)}{127}
$$

Used for dequantization:

$$
W_{rc} + S_c \approx W'_{rc} \cdot scale_r
$$

- `d_balanceFactors` — array of length `cols`

All allocations and host-device copies are already done before `Quantization` is called.
The function only needs to compute launch parameters and invoke `QuantizationKernel`.

Guaranteed: `cols % 4 == 0`.
