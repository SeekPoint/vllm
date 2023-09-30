#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

namespace vllm {

// silu_mult
// silu_mult是融合算子，将silu与下一步的乘加运算融合到一起进行计算。
// silu函数很容易理解，就是按照silu公式写的函数。 怀疑silu_mult中input的2d是相同的值，
// 那么对应的公式便是 y = x * silu(x)
// ldg的作用：ldg会将数据从全局内存中搬运到blcok内的纹理缓存中。
template<typename T>
__device__ __forceinline__ T silu(const T& x) {
  // x * sigmoid(x)
  return (T) (((float) x) / (1.0f + expf((float) -x)));
}

template<typename scalar_t>
__global__ void silu_and_mul_kernel(
  scalar_t* __restrict__ out,               // [num_tokens, d]
  const scalar_t* __restrict__ input,       // [num_tokens, 2, d]
  const int d) {
  const int token_idx = blockIdx.x;
  for (int idx = threadIdx.x; idx < d; idx += blockDim.x) {
    const scalar_t x = __ldg(&input[token_idx * 2 * d + idx]);
    const scalar_t y = __ldg(&input[token_idx * 2 * d + d + idx]);
    out[token_idx * d + idx] = silu(x) * y;
  }
}

} // namespace vllm

void silu_and_mul(
  torch::Tensor& out,      // [num_tokens, d]
  torch::Tensor& input)    // [num_tokens, 2 * d]
{
  int num_tokens = input.size(0);
  int d = input.size(1) / 2;

  dim3 grid(num_tokens);
  dim3 block(std::min(d, 1024));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::Half,
    at::ScalarType::BFloat16,
    input.scalar_type(),
    "silu_and_mul_kernel",
    [&] {
      vllm::silu_and_mul_kernel<scalar_t><<<grid, block, 0, stream>>>(
        out.data_ptr<scalar_t>(),
        input.data_ptr<scalar_t>(),
        d);
    });
}
