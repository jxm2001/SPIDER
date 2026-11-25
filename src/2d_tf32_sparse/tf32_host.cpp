// SPDX-License-Identifier: Apache-2.0
#include "tf32_host.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace spider_tf32 {
namespace {

int original_column(int column) {
    return column % 2 == 0 ? column ^ 16 : column;
}

int pair_position(int row, int pair) {
    const int even_column = original_column(2 * pair);
    if (row + 1 <= even_column && even_column <= row + diameter) {
        return 0;
    }
    const int odd_column = 2 * pair + 1;
    // An all-zero pair still needs a legal index (0x4), with a stored zero.
    return row + 1 <= odd_column && odd_column <= row + diameter ? 1 : 0;
}

}  // namespace

Shape validate_problem(int m, int n, int times) {
    if (m <= 0 || n <= 0 || m % block_rows != 0 || n % block_cols != 0) {
        throw std::invalid_argument("M must be a positive multiple of 64 and N a positive multiple of 128");
    }
    if (times <= 0) {
        throw std::invalid_argument("T must be positive");
    }
    const std::size_t rows = static_cast<std::size_t>(m) + 2 * halo;
    const std::size_t cols = static_cast<std::size_t>(n) + 2 * halo;
    if (rows > std::numeric_limits<std::size_t>::max() / sizeof(float) / cols) {
        throw std::overflow_error("Input allocation size overflows size_t");
    }
    return {rows, cols, rows * cols, rows * cols * sizeof(float)};
}

PackedParams pack_params(const float* params) {
    if (!params) {
        throw std::invalid_argument("params must not be null");
    }
    // Metadata describes the stencil's support, independently of coefficient
    // values, so all 15 matrices can share it even when coefficients are zero.
    for (int h = 0; h < diameter; ++h) {
        for (int r = 0; r < matrix_rows; ++r) {
            for (int c = 0; c < matrix_cols; ++c) {
                const float value = params[(h * matrix_rows + r) * matrix_cols + c];
                if (!std::isfinite(value) ||
                    ((c <= r || c > r + diameter) && value != 0.0f)) {
                    throw std::invalid_argument("Coefficients must be finite and use the radius-7 stencil layout");
                }
            }
        }
    }

    PackedParams packed;
    for (int lane = 0; lane < 32; ++lane) {
        const int group = lane / 4;
        const int thread = lane % 4;
        for (int upper = 0; upper < 2; ++upper) {
            for (int p = 0; p < 4; ++p) {
                const int pair = (thread / 2) * 8 + (thread % 2) * 4 + p;
                const std::uint32_t code = pair_position(group + 8 * upper, pair) ? 0xeu : 0x4u;
                packed.metadata[lane] |= code << (upper * 16 + p * 4);
            }
        }
        for (int h = 0; h < diameter; ++h) {
            for (int slice = 0; slice < 2; ++slice) {
                for (int a = 0; a < 4; ++a) {
                    const int row = group + 8 * (a % 2);
                    const int pair = slice * 8 + thread + 4 * (a / 2);
                    const int column = original_column(2 * pair + pair_position(row, pair));
                    packed.values[h * 256 + lane * 8 + slice * 4 + a] =
                        params[(h * matrix_rows + row) * matrix_cols + column];
                }
            }
        }
    }
    return packed;
}

float round_tf32(float value) {
    static_assert(sizeof(float) == sizeof(std::uint32_t) && std::numeric_limits<float>::is_iec559);
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    std::uint32_t magnitude = bits & 0x7fffffffu;
    if (magnitude >= 0x7f800000u) {
        return value;
    }
    // cvt.rna.tf32.f32: nearest, ties away from zero (not ties to even).
    magnitude = (magnitude + 0x1000u) & ~0x1fffu;
    bits = (bits & 0x80000000u) | magnitude;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

std::vector<float> reference_step(const float* in, const float* params,
                                int m, int n, std::vector<double>* sum_abs) {
    const Shape shape = validate_problem(m, n, 1);
    std::vector<float> rounded(shape.elements, 0.0f);
    std::vector<float> out(shape.elements, 0.0f);
    std::array<float, param_count> coefficients{};
    for (int k = 0; k < param_count; ++k) {
        coefficients[k] = round_tf32(params[k]);
    }
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            const std::size_t index = (static_cast<std::size_t>(i) + halo) * shape.cols + j + halo;
            rounded[index] = round_tf32(in[index]);
        }
    }
    if (sum_abs) {
        sum_abs->assign(shape.elements, 0.0);
    }
    for (int i = 0; i < m; ++i) {
        const int row = i % matrix_rows;
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            double magnitude = 0.0;
            for (int v = 0; v < diameter; ++v) {
                for (int h = 0; h < diameter; ++h) {
                    const float x = rounded[(static_cast<std::size_t>(i) + v + 1) * shape.cols + j + h + 1];
                    const float coefficient = coefficients[(h * matrix_rows + row) * matrix_cols + row + v + 1];
                    sum = std::fma(x, coefficient, sum);
                    magnitude += std::abs(static_cast<double>(x) * coefficient);
                }
            }
            const std::size_t index = (static_cast<std::size_t>(i) + halo) * shape.cols + j + halo;
            out[index] = sum;
            if (sum_abs) {
                (*sum_abs)[index] = magnitude;
            }
        }
    }
    return out;
}

bool within_fp32_error(float actual, float expected, double sum_abs) {
    if (!std::isfinite(actual) || !std::isfinite(expected) ||
        !std::isfinite(sum_abs) || sum_abs < 0.0) {
        return false;
    }
    // Bound both FP32 reduction orders, including cancellation. An FP16-sized
    // absolute tolerance would incorrectly accept flushing small TF32 values.
    constexpr double eps = std::numeric_limits<float>::epsilon();
    constexpr double operations = 2 * diameter * diameter + 1;
    constexpr double gamma = operations * eps / (1.0 - operations * eps);
    const double tolerance = gamma * sum_abs + 8.0 * std::numeric_limits<float>::min();
    return std::abs(static_cast<double>(actual) - expected) <= tolerance;
}

}  // namespace spider_tf32
