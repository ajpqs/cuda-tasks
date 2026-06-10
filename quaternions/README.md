# Quaternions

Implement multiplication of sets of quaternions on GPU.

### Quaternion

A quaternion is a generalization of complex numbers. While a complex number $a + b \textbf{i}$ is encoded as a pair $(a, b)$,
a quaternion $a + b \textbf{i} + c \textbf{j} + d \textbf{k}$ is represented as a 4-tuple $(a, b, c, d)$.

The quaternion multiplication formula:

$$
(a_1 + b_1 \textbf{i} + c_1 \textbf{j} + d_1 \textbf{k}) (a_2 + b_2 \textbf{i} + c_2 \textbf{j} + d_2 \textbf{k}) =
$$

$$
(a_1 a_2 - b_1 b_2 - c_1 c_2 - d_1 d_2) +
$$

$$
(a_1 b_2 + b_1 a_2 + c_1 d_2 - d_1 c_2) \textbf{i} +
$$

$$
(a_1 c_2 - b_1 d_2 + c_1 a_2 + d_1 b_2) \textbf{j} +
$$

$$
(a_1 d_2 + b_1 c_2 - c_1 b_2 + d_1 a_2) \textbf{k}
$$

For details:
* https://en.wikipedia.org/wiki/Quaternion#Multiplication_of_basis_elements
* https://en.wikipedia.org/wiki/Quaternion#Hamilton_product

**Note #1:** Quaternion multiplication is non-commutative.

**Note #2:** A `Quaternion` class and a `QuaternionMultiplier` helper are provided.

### Task

Implement multiplication of sets of quaternions with the following signature:

```cpp
void QuaternionsReduce(size_t rows, size_t cols,
                       const Quaternion* inp, size_t inp_stride,
                       Quaternion* out, cudaStream_t stream);
```

- `rows` — number of input quaternion sets to multiply
- `cols` — number of quaternions in each set
- Input is stored as a `rows x cols` row-major matrix
- Output is a vector of length `rows`
- `inp`/`inp_stride` define input memory layout
- `out` defines output vector location
- `stream` — non-default CUDA stream for kernel launches

**Note #1:** `cols` is guaranteed to be one of 1024, 2048, or 4096, so load loops can be unrolled.

**Note #2:** `Quaternion` is 16 bytes, 16-byte aligned — loading the whole struct maps to a single 128-bit load instruction.

**Note #3:** Since quaternion multiplication is **associative but not commutative**, ensure correct multiplication order at least in the thread writing the result (e.g., thread #0).

**Note #4:** See the CPU reference implementation in the tests for more context.

**Note #5:** No allocations or copies should be performed inside `QuaternionsReduce`.
