# Softmax

Implement softmax — a generalization of the sigmoid function to multi-dimensional inputs.
This task focuses on efficient softmax for **large dimensions** (tens of thousands or more).

### Sigmoid

Sigmoid is a monotonic non-linear function mapping real numbers to the interval `(0; 1)`:

$$
\sigma(x) = \cfrac{1}{1 + e^{-x}}
$$

### Softmax

Softmax generalizes sigmoid to vectors.

Given a vector $x = {x_1, x_2, \ldots, x_N} \in \mathbb{R}^N$, softmax is defined as:

$$
Softmax(x) = \left\{\cfrac{e^{x_1}}{C}, \cfrac{e^{x_2}}{C}, \ldots \cfrac{e^{x_N}}{C}\right\} \in [0; 1]^N,
$$

where the normalization constant $C = \sum_{i=1}^N e^{x_i}$.

Thus, if $y = Softmax(x)$:
$$
y_i \ge 0, \quad \sum_{i=1}^N y_i = 1
$$

**Example:**
```cpp
x = {     100.0f,      101.0f,      102.0f,      103.0f,      104.0f};
y = {0.01165623f, 0.03168492f, 0.08612854f, 0.23412165f, 0.63640864f};
```

See CPU softmax implementation in the tests for details.

### Task

Implement one function:

```cpp
void Softmax(size_t rows, size_t cols,
             const float* d_input_matrix, size_t input_stride,
             float* d_out_matrix, size_t out_stride,
             cudaStream_t stream);
```

- `rows` — number of input vectors
- `cols` — dimension of each vector
- Input/output are `rows x cols` row-major matrices
- `stream` — non-default CUDA stream

**Guarantees:**
- `cols % 4 == 0` (enables 128-bit loads)

**Notes:**
- No allocations or copies inside `Softmax`
- At most one kernel launch is expected; multiple kernels may not meet benchmark constraints
