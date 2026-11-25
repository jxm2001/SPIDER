// Copyright (c) 2025 Qiqi Gu, Chenpeng Wu.
// SPDX-License-Identifier: Apache-2.0
#include "argparse/argparse.hpp"
#include "gpu_2d_7r_tf32.h"
#include "tf32_host.h"

#include <exception>
#include <iostream>
#include <vector>

int main(int argc, const char** argv) {
    using namespace spider_tf32;
    try {
        argparse::ArgumentParser parser("stencil_2d_tf32_sparse");
        parser.add_argument("-M").help("Dimension M (positive multiple of 64)").default_value(16384).scan<'i', int>();
        parser.add_argument("-N").help("Dimension N (positive multiple of 128)").default_value(16384).scan<'i', int>();
        parser.add_argument("-T", "--times").help("Repeat times of the 2D Stencil operation").default_value(100).scan<'i', int>();
        parser.add_argument("--check").help("Check one step with TF32-rounded operands").default_value(false).implicit_value(true);
        parser.add_argument("--profile").help("Profile the kernel").default_value(false).implicit_value(true);
        parser.parse_args(argc, argv);

        const int m = parser.get<int>("-M");
        const int n = parser.get<int>("-N");
        const int times = parser.get<int>("--times");
        const Shape shape = validate_problem(m, n, times);
        std::array<float, param_count> params{};
        for (int h = 0; h < diameter; ++h) {
            for (int v = 0; v < diameter; ++v) {
                for (int row = 0; row < matrix_rows; ++row) {
                    params[(h * matrix_rows + row) * matrix_cols + row + v + 1] = (v + 1) % 13 + 1;
                }
            }
        }
        std::vector<float> in(shape.elements, 0.0f);
        std::vector<float> out(shape.elements, 0.0f);
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                in[(static_cast<std::size_t>(i) + halo) * shape.cols + j + halo] =
                    (static_cast<std::size_t>(i) * shape.cols + j) % 19;
            }
        }
        const auto print_prefix = [&] {
            std::cout << "SPTC_tf32, 2d, 1, " << n << ", " << m << ", " << times << ", ";
        };
        if (parser.get<bool>("--check")) {
            print_prefix();
            std::cout << "Checking result...\n";
            gpu_2d_7r_tf32(in.data(), out.data(), params.data(), m, n, times, true);
            std::vector<double> sum_abs;
            const auto expected = reference_step(in.data(), params.data(), m, n, &sum_abs);
            for (std::size_t i = 0; i < shape.elements; ++i) {
                if (!within_fp32_error(out[i], expected[i], sum_abs[i])) {
                    std::cerr << "Index: [" << i / shape.cols << ", " << i % shape.cols
                              << "], Expected Result: " << expected[i] << ", GPU Result: " << out[i]
                              << "\nResult: Wrong\n";
                    return 1;
                }
            }
            std::cout << "Result: Correct\n";
        }
        if (parser.get<bool>("--profile")) {
            print_prefix();
            gpu_2d_7r_tf32(in.data(), out.data(), params.data(), m, n, times, false);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
