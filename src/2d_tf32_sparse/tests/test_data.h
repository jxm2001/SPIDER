// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "tf32_host.h"

#include <array>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace tf32_test {

using Weights = std::array<float, 15 * 15>;
using Params = std::array<float, 15 * 16 * 32>;

inline void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

inline Params make_params(const Weights& weights) {
    Params params{};
    for (int v = 0; v < 15; ++v) {
        for (int h = 0; h < 15; ++h) {
            for (int r = 0; r < 16; ++r) {
                params[h * 512 + r * 32 + r + v + 1] = weights[v * 15 + h];
            }
        }
    }
    return params;
}

inline Weights asymmetric_weights() {
    Weights weights{};
    for (int v = 0; v < 15; ++v) {
        for (int h = 0; h < 15; ++h) {
            weights[v * 15 + h] = (v + h) % 5 == 0 ? 0.0f : ((v * 17 + h * 29) % 31 - 15) / 257.0f;
        }
    }
    return weights;
}

inline std::vector<float> make_input(int m, int n) {
    const auto shape = spider_tf32::validate_problem(m, n, 1);
    std::vector<float> in(shape.elements, 0.0f);
    std::mt19937 rng(20261125);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            in[(static_cast<std::size_t>(i) + 8) * shape.cols + j + 8] =
                (static_cast<int>(rng() % 2001) - 1000) / 1009.0f;
        }
    }
    return in;
}

}  // namespace tf32_test
