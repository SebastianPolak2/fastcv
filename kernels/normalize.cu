#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

__global__ void normalizeKernelLinear(unsigned char* data, int total_elements, float newMin, float newMax, float oldMin, float oldMax) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) {
        return;
    }
    unsigned char I_in = data[idx];

    double ratio = (double)(I_in - oldMin) / (oldMax - oldMin);
    double normalized = ratio * (newMax - newMin) + newMin;
    unsigned char output_value = (normalized < 0.0) ? 0 : (normalized > 255.0) ? 255 : (unsigned char)(normalized + 0.5);
    data[idx] = output_value;
}

torch::Tensor normalize(torch::Tensor img, float new_min, float new_max, float old_min, float old_max) {
    assert(img.device().type() == torch::kCUDA);
    assert(img.dtype() == torch::kByte);

    int total_elements = img.numel();
    auto result = img.clone(); 

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (!result.is_contiguous()) {
        result = result.contiguous();
    }
    thrust::device_ptr<unsigned char> dev_ptr(result.data_ptr<unsigned char>());
    int threadsPerBlock = 256;
    int blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;

    normalizeKernelLinear<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        thrust::raw_pointer_cast(dev_ptr), 
        total_elements,
        new_min,
        new_max,
        old_min,
        old_max
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return result;
}