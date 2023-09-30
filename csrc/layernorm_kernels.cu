#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "reduction_utils.cuh"

namespace vllm {

/*
rms_norm核函数
rms核函数按照不同的token数划分，所以一共有num_tokens个block，每个block内开多个线程去执行rms算子。

平方和也在block内部计算，每个线程累加了部分和，使用FT中的函数BlockReduceSum计算所有结果的平方和，
规约到线程0处，然后更新不同token的output得到结果。

*/

// TODO(woosuk): Further optimize this kernel.
template<typename scalar_t>
__global__ void rms_norm_kernel(
  scalar_t* __restrict__ out,             // [num_tokens, hidden_size]
  const scalar_t* __restrict__ input,     // [num_tokens, hidden_size]
  const scalar_t* __restrict__ weight,    // [hidden_size]
  const float epsilon,
  const int num_tokens,
  const int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float x = (float) input[blockIdx.x * hidden_size + idx];
    variance += x * x;
  }
  variance = blockReduceSum<float>(variance);
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = (float) input[blockIdx.x * hidden_size + idx];
    out[blockIdx.x * hidden_size + idx] = ((scalar_t) (x * s_variance)) * weight[idx];
  }
}

} // namespace vllm

void rms_norm(
  torch::Tensor& out,      // [num_tokens, hidden_size]
  torch::Tensor& input,    // [num_tokens, hidden_size]
  torch::Tensor& weight,   // [hidden_size]
  float epsilon) {
  int num_tokens = input.size(0);
  int hidden_size = input.size(1);

  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::Half,
    at::ScalarType::BFloat16,
    input.scalar_type(),
    "rms_norm_kernel",
    [&] {
      vllm::rms_norm_kernel<scalar_t><<<grid, block, 0, stream>>>(
        out.data_ptr<scalar_t>(),
        input.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        epsilon,
        num_tokens,
        hidden_size);
    });
}
