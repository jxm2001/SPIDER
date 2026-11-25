// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace spider_tf32 {

inline constexpr int radius = 7;
inline constexpr int diameter = 2 * radius + 1;
inline constexpr int halo = 8;
inline constexpr int block_rows = 64;
inline constexpr int block_cols = 128;
inline constexpr int matrix_rows = 16;
inline constexpr int matrix_cols = 32;
inline constexpr int param_count = diameter * matrix_rows * matrix_cols;
inline constexpr int compressed_count = diameter * matrix_rows * matrix_cols / 2;

struct Shape {
    std::size_t rows;
    std::size_t cols;
    std::size_t elements;
    std::size_t bytes;
};

struct PackedParams {
    // [stencil column][lane][K slice][A register]
    alignas(16) std::array<float, compressed_count> values{};
    // Selector 0 consumes lanes 4g,4g+1; selector 1 consumes 4g+2,4g+3.
    std::array<std::uint32_t, 32> metadata{};
};

Shape validate_problem(int m, int n, int times);
PackedParams pack_params(const float* params);
float round_tf32(float value);

// One step with TF32-rounded multiplicands, FP32 accumulation and zero halo.
std::vector<float> reference_step(const float* in, const float* params,
                                int m, int n, std::vector<double>* sum_abs = nullptr);
bool within_fp32_error(float actual, float expected, double sum_abs);

}  // namespace spider_tf32
