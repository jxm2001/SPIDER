// SPDX-License-Identifier: Apache-2.0
#include "test_data.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>

namespace {
using namespace spider_tf32;
using namespace tf32_test;

float from_bits(std::uint32_t bits) {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

void test_rounding() {
    require(round_tf32(from_bits(0x3f801000)) == from_bits(0x3f802000), "positive RNA tie");
    require(round_tf32(from_bits(0xbf801000)) == from_bits(0xbf802000), "negative RNA tie");
    require(round_tf32(from_bits(0x3f800fff)) == 1.0f, "below RNA midpoint");
    require(round_tf32(from_bits(0x3f801001)) == from_bits(0x3f802000), "above RNA midpoint");
    require(std::signbit(round_tf32(-0.0f)), "negative zero sign");
    require(round_tf32(1.0e-8f) != 0.0f, "TF32 small value survives");
    require(std::isfinite(round_tf32(1.0e8f)), "TF32 large value survives");
    require(std::isinf(round_tf32(std::numeric_limits<float>::infinity())), "infinity conversion");
    require(std::isnan(round_tf32(std::numeric_limits<float>::quiet_NaN())), "NaN conversion");
    require(!within_fp32_error(std::numeric_limits<float>::quiet_NaN(), 0, 0), "NaN must fail validation");
    require(!within_fp32_error(std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), 1),
            "infinity must fail validation");
    require(!within_fp32_error(0.0f, 1.0e-10f, 1.0e-10), "small results cannot use an FP16 absolute tolerance");
}

void test_shapes() {
    const auto shape = validate_problem(64, 128, 1);
    require(shape.rows == 80 && shape.cols == 144 && shape.bytes == 80 * 144 * 4, "padded shape");
    require(validate_problem(131072, 131072, 3).bytes > std::numeric_limits<std::uint32_t>::max(),
            "allocation arithmetic must be wider than 32 bits");
    const int invalid[][3] = {{0,128,1}, {-64,128,1}, {63,128,1}, {65,128,1},
                             {64,0,1}, {64,-128,1}, {64,127,1}, {64,129,1},
                             {64,128,0}, {64,128,-1}};
    for (const auto& args : invalid) {
        bool rejected = false;
        try { validate_problem(args[0], args[1], args[2]); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "invalid dimensions or iteration count accepted");
    }
}

void check_packing(const Params& params) {
    const auto packed = pack_params(params.data());
    // Independently decode the PTX A fragments and the two metadata thread
    // pairs, then undo the column permutation. This must be lossless.
    for (int h = 0; h < 15; ++h) {
        std::array<float, 16 * 32> reconstructed{};
        for (int r = 0; r < 16; ++r) {
            for (int s = 0; s < 2; ++s) {
                for (int p = 0; p < 8; ++p) {
                    const int metadata_lane = 4 * (r % 8) + 2 * s + p / 4;
                    const unsigned nibble = (packed.metadata[metadata_lane] >>
                                              (16 * (r / 8) + 4 * (p % 4))) & 0xfu;
                    require(nibble == 0x4 || nibble == 0xe, "illegal TF32 metadata nibble");
                    const int column = 16 * s + 2 * p + (nibble == 0xe);
                    const int original = column % 2 == 0 ? (column + 16) % 32 : column;
                    const int a_lane = 4 * (r % 8) + p % 4;
                    const int a_register = r / 8 + 2 * (p / 4);
                    reconstructed[r * 32 + original] = packed.values[h * 256 + a_lane * 8 + s * 4 + a_register];
                }
            }
        }
        for (int i = 0; i < 16 * 32; ++i) {
            require(reconstructed[i] == params[h * 512 + i], "packed coefficients do not reconstruct the stencil");
        }
    }
}

void test_packing() {
    auto weights = asymmetric_weights();
    auto params = make_params(weights);
    check_packing(params);
    const auto metadata = pack_params(params.data()).metadata;
    for (int i = 0; i < 225; ++i) weights[i] = static_cast<float>(i + 1);
    check_packing(make_params(weights));
    weights.fill(0.0f);
    params = make_params(weights);
    check_packing(params);
    require(pack_params(params.data()).metadata == metadata, "metadata must not depend on numeric zeros");
    for (int i = 0; i < 225; ++i) {
        weights.fill(0.0f);
        weights[i] = 1.0f;
        check_packing(make_params(weights));
    }
    params[0] = 1.0f;
    bool rejected = false;
    try { pack_params(params.data()); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "out-of-support coefficient was silently dropped");
    params[0] = 0.0f;
    params[1] = std::numeric_limits<float>::quiet_NaN();
    rejected = false;
    try { pack_params(params.data()); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "nonfinite coefficients accepted");
}

void test_reference() {
    constexpr int m = 64, n = 128;
    const auto shape = validate_problem(m, n, 1);
    Weights weights{};
    weights[2 * 15 + 11] = -3.5f;
    const auto params = make_params(weights);
    std::vector<float> in(shape.elements, 12345.0f);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            in[(i + 8) * shape.cols + j + 8] = (i - j) / 4.0f;
        }
    }
    std::vector<double> sum_abs;
    const auto actual = reference_step(in.data(), params.data(), m, n, &sum_abs);
    for (std::size_t i = 0; i < shape.rows; ++i) {
        for (std::size_t j = 0; j < shape.cols; ++j) {
            float expected = 0.0f;
            if (i >= 8 && i < m + 8 && j >= 8 && j < n + 8 &&
                i >= 13 && j + 4 < n + 8) {
                expected = -3.5f * in[(i - 5) * shape.cols + j + 4];
            }
            require(actual[i * shape.cols + j] == expected, "reference orientation or zero boundary incorrect");
            require(sum_abs[i * shape.cols + j] == std::abs(expected), "reference magnitude bound incorrect");
        }
    }
}

}  // namespace

int main() {
    try {
        test_rounding();
        test_shapes();
        test_packing();
        test_reference();
        std::cout << "TF32 host tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "TF32 host test failed: " << error.what() << '\n';
        return 1;
    }
}
