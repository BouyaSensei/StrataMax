// tests/kernels/nvfp4_planes_test.cpp - kernels::nvfp4p::from_blocks: every value's E2M1 code and UE4M3 scale of a random
// ggml NVFP4 matrix land where strata/kernels/nvfp4_planes.hpp says, and dequantize to the same floats as the reference
// blocks.  Exit 77 without a GPU.
//
// Ported from basalt's tests/kernels/nvfp4_planes_test.cpp. basalt read its reference decoder from
// include/basalt/artifact/dequant.hpp; this fork's include/strata/artifact/dequant.hpp has no NVFP4 dequantizer, so the
// two functions it used (ue4m3_to_fp32_half, dequantize_nvfp4) are transcribed here, in the test, from the source.
#include "strata/kernels/nvfp4_planes.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace p = strata::kernels::nvfp4p;

namespace {

// ggml's doubled E2M1 table, transcribed from basalt's artifact/dequant.hpp
const int8_t kvalues_fp4[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

float ue4m3_to_fp32_half(uint8_t x) {
    x &= 0x7F;
    if (x == 0 || x == 0x7F) return 0.0f;
    const int e = (x >> 3) & 0xF, m = x & 0x7;
    const float raw = e == 0 ? std::ldexp((float) m, -9) : std::ldexp(1.0f + (float) m / 8.0f, e - 7);
    return raw * 0.5f;
}

void dequantize_nvfp4(const uint8_t* block, float* out) {
    const uint8_t* d = block;
    const uint8_t* qs = block + 4;
    for (int s = 0; s < 4; ++s) {
        const float sc = ue4m3_to_fp32_half(d[s]);
        for (int j = 0; j < 8; ++j) {
            out[16 * s + j] = (float) kvalues_fp4[qs[8 * s + j] & 0x0F] * sc;
            out[16 * s + j + 8] = (float) kvalues_fp4[qs[8 * s + j] >> 4] * sc;
        }
    }
}

}  // namespace

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) return 77;
    const int64_t rows = 256, K = 640;
    std::mt19937 rng(4);
    std::vector<uint8_t> blk(p::bytes(rows, K));
    for (auto& b : blk) b = (uint8_t) rng();
    for (size_t i = 0; i < blk.size(); i += 36)
        for (int s = 0; s < 4; ++s) blk[i + s] = (uint8_t) (rng() % 0x7F);
    void *d_b, *d_p;
    cudaMalloc(&d_b, blk.size());
    cudaMalloc(&d_p, blk.size());
    cudaMemcpy(d_b, blk.data(), blk.size(), cudaMemcpyHostToDevice);
    p::from_blocks(d_b, rows, K, d_p, nullptr);
    std::vector<uint8_t> pl(blk.size());
    cudaMemcpy(pl.data(), d_p, pl.size(), cudaMemcpyDeviceToHost);
    static const float e2m1[16] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6, -0.0f, -0.5f, -1, -1.5f, -2, -3, -4, -6};
    int bad = 0;
    std::vector<float> ref(64);
    for (int64_t r = 0; r < rows; ++r)
        for (int64_t b = 0; b < K / 64; ++b) {
            dequantize_nvfp4(&blk[(size_t) (r * (K / 64) + b) * 36], ref.data());
            for (int v = 0; v < 64; ++v) {
                const int64_t k = b * 64 + v;
                const uint8_t byte = pl[(size_t) (r * (K / 2) + k / 2)];
                const int code = (k & 1) ? byte >> 4 : byte & 15;
                const uint8_t sc = pl[p::scale_offset(rows, K) + (size_t) p::scale_index(r, k / 16, K)];
                const float got = e2m1[code] * ue4m3_to_fp32_half(sc) * 2.0f;
                bad += got != ref[(size_t) v] && !(got == 0.0f && ref[(size_t) v] == 0.0f);
            }
        }
    // and back: the blocks bit for bit
    void* d_b2;
    cudaMalloc(&d_b2, blk.size());
    p::to_blocks(d_p, rows, K, d_b2, nullptr);
    std::vector<uint8_t> back(blk.size());
    cudaMemcpy(back.data(), d_b2, back.size(), cudaMemcpyDeviceToHost);
    const bool round = back == blk;
    const bool ok = bad == 0 && round && cudaGetLastError() == cudaSuccess;
    std::printf("nvfp4 planes from ggml blocks: %s (%d values differ); back to blocks: %s\n",
                bad == 0 ? "same values" : "WRONG  FAIL", bad, round ? "bitwise" : "DIFFER  FAIL");
    return ok ? 0 : 1;
}
