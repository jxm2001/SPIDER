// Copyright (c) 2025 Qiqi Gu, Chenpeng Wu.
// SPDX-License-Identifier: Apache-2.0
#include "gpu_2d_7r_tf32.h"
#include "tf32_host.h"

#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>
#include <string>

namespace {
using namespace spider_tf32;

constexpr int shared_rows = block_rows + 2 * halo;
constexpr int load_cols = block_cols + 2 * halo;
constexpr int shared_cols = load_cols + 8;
constexpr int output_cols = block_cols + 8;
constexpr int threads = (block_rows / matrix_rows) * 32;
constexpr std::size_t shared_bytes = shared_rows * shared_cols * sizeof(float);
static_assert(shared_bytes <= 48 * 1024);

__device__ __forceinline__ unsigned to_tf32(float value) {
    unsigned bits;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(bits) : "f"(value));
    return bits;
}

template<int Selector>
__device__ __forceinline__ void sparse_mma(float (&acc)[4], const unsigned (&a)[4],
                                          const unsigned (&b)[4], unsigned metadata) {
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, "
        "{%0,%1,%2,%3}, %12, %13;\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]),
          "r"(metadata), "n"(Selector));
}

__global__ void kernel_2d_7r_tf32(const float* __restrict__ in,
                                 const float* __restrict__ params,
                                 const unsigned* __restrict__ metadata,
                                 float* __restrict__ out, std::size_t ldm) {
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const std::size_t begin = static_cast<std::size_t>(blockIdx.x) * block_rows * ldm
                           + static_cast<std::size_t>(blockIdx.y) * block_cols;
    __shared__ __align__(16) float shared[shared_rows * shared_cols];
    const unsigned base = static_cast<unsigned>(__cvta_generic_to_shared(shared));

    // A 16-byte transfer now contains four floats, instead of eight halves.
    #pragma unroll
    for (int i = threadIdx.x; i < shared_rows * load_cols / 4; i += threads) {
        const int row = i * 4 / load_cols;
        const int col = i * 4 % load_cols;
        const unsigned dst = base + (row * shared_cols + col) * sizeof(float);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                     : : "r"(dst), "l"(in + begin + row * ldm + col) : "memory");
    }
    asm volatile("cp.async.commit_group;\n" : : : "memory");
    asm volatile("cp.async.wait_group 0;\n" : : : "memory");
    __syncthreads();

    float accumulators[block_cols / 8][4] = {};
    const unsigned meta = metadata[lane];
    #pragma unroll
    for (int tile = 0; tile < block_cols / 8; ++tile) {
        #pragma unroll
        for (int h = 0; h < diameter; ++h) {
            #pragma unroll
            for (int slice = 0; slice < 2; ++slice) {
                const float4 weights = *reinterpret_cast<const float4*>(
                    params + h * 256 + lane * 8 + slice * 4);
                const unsigned a[4] = {to_tf32(weights.x), to_tf32(weights.y),
                                       to_tf32(weights.z), to_tf32(weights.w)};
                unsigned b[4];
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    int row = slice * 16 + lane % 4 + 4 * i;
                    // All four B registers of an even lane swap rows. FP16's
                    // alternating-register rule is not the TF32 fragment map.
                    if (lane % 2 == 0) {
                        row ^= 16;
                    }
                    b[i] = to_tf32(shared[(warp * 16 + row) * shared_cols
                                          + tile * 8 + h + 1 + lane / 4]);
                }
                if (slice == 0) {
                    sparse_mma<0>(accumulators[tile], a, b, meta);
                } else {
                    sparse_mma<1>(accumulators[tile], a, b, meta);
                }
            }
        }
    }

    __syncthreads();
    #pragma unroll
    for (int tile = 0; tile < block_cols / 8; ++tile) {
        const int row = warp * 16 + lane / 4;
        const int col = tile * 8 + (lane % 4) * 2;
        shared[row * output_cols + col] = accumulators[tile][0];
        shared[row * output_cols + col + 1] = accumulators[tile][1];
        shared[(row + 8) * output_cols + col] = accumulators[tile][2];
        shared[(row + 8) * output_cols + col + 1] = accumulators[tile][3];
    }
    __syncthreads();

    #pragma unroll
    for (int i = threadIdx.x; i < block_rows * block_cols / 4; i += threads) {
        const int row = i * 4 / block_cols;
        const int col = i * 4 % block_cols;
        *reinterpret_cast<float4*>(out + begin + (row + halo) * ldm + col + halo) =
            *reinterpret_cast<const float4*>(shared + row * output_cols + col);
    }
}

void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

struct Resources {
    float* arrays[2]{};
    float* params = nullptr;
    unsigned* metadata = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    Resources() = default;
    Resources(const Resources&) = delete;
    Resources& operator=(const Resources&) = delete;
    ~Resources() {
        if (stop) cudaEventDestroy(stop);
        if (start) cudaEventDestroy(start);
        if (metadata) cudaFree(metadata);
        if (params) cudaFree(params);
        for (float* array : arrays) {
            if (array) cudaFree(array);
        }
    }
};

}  // namespace

void gpu_2d_7r_tf32(const float* in, float* out, const float* params,
                    int m, int n, int times, bool check) {
    using namespace spider_tf32;
    const Shape shape = validate_problem(m, n, times);
    if (!in || !out || !params) {
        throw std::invalid_argument("in, out and params must not be null");
    }
    const PackedParams packed = pack_params(params);
    int device;
    cuda_check(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp properties;
    cuda_check(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
    if (properties.major < 8 || properties.sharedMemPerBlock < shared_bytes) {
        throw std::runtime_error("TF32 sparse stencil requires SM80+ and at least 48640 bytes of shared memory per block");
    }
    if (m / block_rows > properties.maxGridSize[0] || n / block_cols > properties.maxGridSize[1]) {
        throw std::invalid_argument("Input dimensions exceed the device grid limits");
    }

    Resources resources;
    cuda_check(cudaMalloc(&resources.params, sizeof(packed.values)), "cudaMalloc params");
    cuda_check(cudaMalloc(&resources.metadata, sizeof(packed.metadata)), "cudaMalloc metadata");
    cuda_check(cudaMemcpy(resources.params, packed.values.data(), sizeof(packed.values), cudaMemcpyHostToDevice), "cudaMemcpy params");
    cuda_check(cudaMemcpy(resources.metadata, packed.metadata.data(), sizeof(packed.metadata), cudaMemcpyHostToDevice), "cudaMemcpy metadata");
    for (float*& array : resources.arrays) {
        cuda_check(cudaMalloc(&array, shape.bytes), "cudaMalloc input/output");
        cuda_check(cudaMemset(array, 0, shape.bytes), "cudaMemset input/output");
    }
    const std::size_t interior = halo * shape.cols + halo;
    const std::size_t pitch = shape.cols * sizeof(float);
    cuda_check(cudaMemcpy2D(resources.arrays[0] + interior, pitch, in + interior, pitch,
                           static_cast<std::size_t>(n) * sizeof(float), m, cudaMemcpyHostToDevice),
               "cudaMemcpy2D input interior");

    const dim3 grid(m / block_rows, n / block_cols);
    const int iterations = check ? 1 : times;
    if (!check) {
        cuda_check(cudaEventCreate(&resources.start), "cudaEventCreate start");
        cuda_check(cudaEventCreate(&resources.stop), "cudaEventCreate stop");
        cuda_check(cudaEventRecord(resources.start), "cudaEventRecord start");
    }
    for (int i = 0; i < iterations; ++i) {
        kernel_2d_7r_tf32<<<grid, threads>>>(resources.arrays[i % 2], resources.params,
                                           resources.metadata, resources.arrays[(i + 1) % 2], shape.cols);
        cuda_check(cudaGetLastError(), "kernel_2d_7r_tf32 launch");
        cuda_check(cudaDeviceSynchronize(), "kernel_2d_7r_tf32 execution");
    }
    if (!check) {
        cuda_check(cudaEventRecord(resources.stop), "cudaEventRecord stop");
        cuda_check(cudaEventSynchronize(resources.stop), "cudaEventSynchronize");
        float elapsed_ms;
        cuda_check(cudaEventElapsedTime(&elapsed_ms, resources.start, resources.stop), "cudaEventElapsedTime");
        const double updates = static_cast<double>(m) * n * iterations;
        std::cout << elapsed_ms << ", " << (elapsed_ms > 0 ? updates / (elapsed_ms * 1.0e6) : 0.0) << '\n';
    }
    cuda_check(cudaMemcpy(out, resources.arrays[iterations % 2], shape.bytes, cudaMemcpyDeviceToHost), "cudaMemcpy output");
}
