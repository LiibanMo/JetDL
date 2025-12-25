#include <cuda_runtime.h>
#include <curand.h>

#include "jetdl/random/distributions.h"

namespace jetdl {

// Kernel to transform uniform [0,1) to uniform [low, high)
__global__ void transform_uniform_kernel(float* data, const float low,
                                         const float range, const size_t N) {
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N) {
    data[i] = low + data[i] * range;
  }
}

void c_random_uniform_cuda(float* d_dest, const float low, const float high,
                           const size_t N, const size_t seed) {
  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, seed);

  // Generate uniform in [0, 1)
  curandGenerateUniform(gen, d_dest, N);

  curandDestroyGenerator(gen);

  // Transform to [low, high) range
  if (low != 0.0f || high != 1.0f) {
    const float range = high - low;
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    transform_uniform_kernel<<<blocks, threads>>>(d_dest, low, range, N);
  }
}

void c_random_normal_cuda(float* d_dest, const float mean, const float std,
                          const size_t N, const size_t seed) {
  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, seed);

  // curandGenerateNormal uses Box-Muller transform which produces pairs
  // of values, so N must be even. For odd N, we generate into a temp buffer.
  if (N % 2 == 0) {
    curandGenerateNormal(gen, d_dest, N, mean, std);
  } else {
    // Allocate temp buffer for N+1 values (even count)
    float* d_temp;
    cudaMalloc(&d_temp, (N + 1) * sizeof(float));
    curandGenerateNormal(gen, d_temp, N + 1, mean, std);
    // Copy first N values to destination
    cudaMemcpy(d_dest, d_temp, N * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaFree(d_temp);
  }

  curandDestroyGenerator(gen);
}

}  // namespace jetdl
