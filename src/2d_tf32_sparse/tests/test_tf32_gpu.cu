// SPDX-License-Identifier: Apache-2.0
#include "gpu_2d_7r_tf32.h"
#include "test_data.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iostream>

namespace {
using namespace spider_tf32;
using namespace tf32_test;

void run_case(const std::string& name, const std::vector<float>& input,
              const Weights& weights, int m, int n, int times = 1, bool check = true) {
    const auto shape = validate_problem(m, n, times);
    const auto params = make_params(weights);
    std::vector<float> actual(shape.elements, -12345.0f);
    gpu_2d_7r_tf32(input.data(), actual.data(), params.data(), m, n, times, check);
    std::vector<float> expected = input;
    std::vector<double> sum_abs;
    const int iterations = check ? 1 : times;
    for (int i = 0; i < iterations; ++i) {
        expected = reference_step(expected.data(), params.data(), m, n, &sum_abs);
    }
    for (std::size_t i = 0; i < shape.elements; ++i) {
        if (!within_fp32_error(actual[i], expected[i], sum_abs[i])) {
            throw std::runtime_error(name + " mismatch at (" + std::to_string(i / shape.cols) + ", "
                                     + std::to_string(i % shape.cols) + "): GPU=" + std::to_string(actual[i])
                                     + ", reference=" + std::to_string(expected[i]));
        }
        const std::size_t row = i / shape.cols, col = i % shape.cols;
        if (row < 8 || row >= static_cast<std::size_t>(m) + 8 ||
            col < 8 || col >= static_cast<std::size_t>(n) + 8) {
            require(actual[i] == 0.0f, name + " wrote a nonzero halo");
        }
    }
    std::cout << name << " passed\n";
}

void test_single_steps() {
    auto input = make_input(64, 128);
    Weights integer_weights{};
    for (int i = 0; i < 225; ++i) integer_weights[i] = (i / 15 + 1) % 13 + 1;
    for (float& value : input) value = std::trunc(value * 18);
    run_case("integer single block", input, integer_weights, 64, 128);
    run_case("asymmetric signed multiblock", make_input(128, 256), asymmetric_weights(), 128, 256);
    run_case("all-zero coefficients", make_input(64, 128), Weights{}, 64, 128);

    input = make_input(128, 256);
    std::fill(input.begin(), input.end(), 0.0f);
    const auto shape = validate_problem(128, 256, 1);
    const int impulses[][2] = {{0,0}, {0,255}, {63,127}, {64,128}, {127,0}, {127,255}};
    for (const auto& point : impulses) input[(point[0] + 8) * shape.cols + point[1] + 8] = 1.0f;
    run_case("halo and block-edge impulses", input, asymmetric_weights(), 128, 256);

    // Nonzero caller padding must not leak into the fixed zero boundary.
    for (std::size_t i = 0; i < shape.rows; ++i) {
        for (std::size_t j = 0; j < shape.cols; ++j) {
            if (i < 8 || i >= 136 || j < 8 || j >= 264) input[i * shape.cols + j] = 9999.0f;
        }
    }
    run_case("zero boundary ignores caller padding", input, asymmetric_weights(), 128, 256);
}

void test_precision() {
    Weights weights{};
    weights[7 * 15 + 7] = 0.75f;
    auto input = make_input(64, 128);
    for (float& value : input) value *= 1.0e8f;
    run_case("input exceeds FP16 range", input, weights, 64, 128);
    input = make_input(64, 128);
    for (float& value : input) value *= 1.0e-10f;
    run_case("input below FP16 range", input, weights, 64, 128);
    weights[7 * 15 + 7] = 1.0e8f;
    run_case("coefficient exceeds FP16 range", make_input(64, 128), weights, 64, 128);
    weights[7 * 15 + 7] = 1.0e-10f;
    run_case("coefficient below FP16 range", make_input(64, 128), weights, 64, 128);

    weights[7 * 15 + 7] = 1.00048828125f;
    const auto shape = validate_problem(64, 128, 1);
    input.assign(shape.elements, 0.0f);
    const float values[] = {1.00048828125f, -1.00048828125f, 1.00048816204071044921875f,
                            -1.00048816204071044921875f, 1.00048840045928955078125f, 0.0f};
    for (int i = 0; i < 64; ++i) {
        for (int j = 0; j < 128; ++j) input[(i + 8) * shape.cols + j + 8] = values[(i + j) % 6];
    }
    run_case("RNA conversion of both operands", input, weights, 64, 128);
}

void test_iterations() {
    constexpr int m = 128, n = 256;
    const auto shape = validate_problem(m, n, 1);
    std::vector<float> input(shape.elements, 0.0f);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            input[(i + 8) * shape.cols + j + 8] = ((i * 37 + j * 11) % 97 - 48) / 64.0f + 0.00048828125f;
        }
    }
    // Dyadic weights keep FP32 reduction exact here, while successive inputs
    // still require TF32 rounding. This isolates iteration and buffer bugs.
    Weights weights{};
    weights[0] = 0.25f;
    weights[7 * 15 + 7] = 0.5f;
    weights[224] = 0.25f;
    for (int times : {1, 2, 3}) {
        run_case("profile iterations=" + std::to_string(times), input, weights, m, n, times, false);
    }
    run_case("check mode remains one step", input, weights, m, n, 3, true);
}

}  // namespace

int main() {
    int devices = 0;
    const cudaError_t status = cudaGetDeviceCount(&devices);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
        (status == cudaSuccess && devices == 0)) {
        std::cout << "SKIP: no usable CUDA device/driver\n";
        return 77;
    }
    if (status != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount: " << cudaGetErrorString(status) << '\n';
        return 1;
    }
    cudaDeviceProp properties;
    if (cudaGetDeviceProperties(&properties, 0) != cudaSuccess) return 1;
    if (properties.major < 8) {
        std::cout << "SKIP: sparse TF32 requires SM80+\n";
        return 77;
    }
    try {
        test_single_steps();
        test_precision();
        test_iterations();
        std::cout << "TF32 GPU tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "TF32 GPU test failed: " << error.what() << '\n';
        return 1;
    }
}
