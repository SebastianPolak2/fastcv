#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/ThrustAllocator.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>


__global__ void normalizeKernelLinear(const unsigned char* __restrict__ in, unsigned char* __restrict__ out, int n, float scale, float shift) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    int v = __float2int_rn(fmaf(in[idx], scale, shift));    //fmaf(a,b,c) = a*b+c z jednym zaokragleniem na koncu (fused multiply add)
    out[idx] = (unsigned char)min(max(v, 0), 255);
}


torch::Tensor normalize(torch::Tensor img, double alpha, double beta) {
    TORCH_CHECK(img.device().type() == torch::kCUDA, "normalize: img must be a CUDA tensor");
    TORCH_CHECK(img.dtype() == torch::kByte, "normalize: img must be uint8");

    img = img.contiguous(); // if bajty tensora nie leza jedne za druigm, tworzy nowy tenor z przepisanymi bajtami po kolei
    auto result = torch::empty_like(img); //nowy tensor na wynik typ sam - torch::Tensor
    int n = img.numel();    //H*W*3
    if (n == 0) {
        return result;
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    at::cuda::ThrustAllocator allocator;  // Thrust bierze pamiec pomocnicza z puli PyTotcha
    auto policy = thrust::cuda::par(allocator).on(stream); // polityka wyonania dla Thrusta

   
    thrust::device_ptr<const unsigned char> p(img.data_ptr<unsigned char>());
    auto mm = thrust::minmax_element(policy, p, p + n);
    double smin = *mm.first;    //cpu czeka bo wynik musi trafic do cpu, kopiowanie 1 bajtu z karty na ram
    double smax = *mm.second;

    // wspolczynniki scale i shitf takie same dla kazdego bajtu dst = src * scale + shift.
    double dmin = std::min(alpha, beta);
    double dmax = std::max(alpha, beta);
    double scale = smax > smin ? (dmax - dmin) / (smax - smin) : 0.0;
    double shift = dmin - smin * scale;


    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    normalizeKernelLinear<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        img.data_ptr<unsigned char>(),  //adres obrazu wejsciowego
        result.data_ptr<unsigned char>(),   //adres bufora na wynik 
        n,
        (float)scale,
        (float)shift
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK(); //RuntimeError w pythonie jesli zglosi wyjatwek uruchomienia (nie liczenia, kernel wciaz liczy)

    return result;
}
