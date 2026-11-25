// SPDX-License-Identifier: Apache-2.0
#pragma once

// in/out contain (m+16)*(n+16) floats, with the interior at offset (8,8).
// The boundary is fixed at zero. params uses the FP16 path's [15][16][32]
// stencil-column/Toeplitz-row/column layout, with float coefficients.
// check runs one step without timing; otherwise times steps are timed.
// Invalid arguments and CUDA failures throw std::exception subclasses.
void gpu_2d_7r_tf32(const float* in, float* out, const float* params,
                    int m, int n, int times, bool check);
