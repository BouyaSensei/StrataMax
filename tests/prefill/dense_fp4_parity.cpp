// tests/prefill/dense_fp4_parity.cpp - STRATA_DENSE_FP4: the FP4 (W4A4) dense product against a CPU FP32 reference on
// the prompt path's real projection shapes, with the pack model's real weights.
//
// Why the Q8_0 weights are made here and not read from the file: the GSQ-RCO IQ2_XS artifact keeps its dense
// projections as IQ4_XS / IQ3_S / Q2_0 / Q6_K, and the only Q8_0 ones it carries are the shared-expert down
// (K 640 - a shape the FP4 kernel refuses, so it stays on the FP16 path). For each form the test takes the GGUF
// tensor of that shape, dequantizes it with strata::quant (the same arithmetic as ggml's `dequantize_row_*`),
// quantizes the values to Q8_0 with ggml's reference rounding (the form src/core/native_dense.cpp uploads for a BF16
// pack, #4741), and runs `nvfp4p::from_q8` on them. The weight values are the model's; the Q8_0 step is exactly what
// the engine's FP4 path starts from.
//
// The reference is the FP32 product of those Q8_0 weights with the same rows, summed in double. Per form the test
// prints the mean and the maximum relative error over every output, the residual's norm relative to the reference's,
// and the token rows' sums (reference and FP4). The verdict: the mean relative error is under 1 %.
//
// A GGUF is required: `--gguf PATH`. Without it, or without a card, the test exits 77 (CMake's skip code) - it never
// "passes" by finding nothing to compare.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/nvfp4_planes.hpp"
#include "strata/prefill/fp4_dense.hpp"
#include "strata/quant/dequant.hpp"
#include "strata/quant/types.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace np = strata::kernels::nvfp4p;
namespace fd = strata::prefill::fp4_dense;
namespace sq = strata::quant;

namespace {

// ggml's f32 -> f16 rounding (the same code as src/core/native_dense.cpp:121, which builds a pack's Q8_0 copies)
uint16_t half_bits(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t e = (int32_t) ((x >> 23) & 0xff) - 127 + 15;
    uint32_t m = x & 0x7fffffu;
    if (e <= 0) {
        if (e < -10) return (uint16_t) sign;
        m |= 0x800000u;
        const uint32_t shift = (uint32_t) (14 - e);
        uint32_t h = m >> shift;
        const uint32_t rem = m & ((1u << shift) - 1), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (h & 1u))) ++h;
        return (uint16_t) (sign | h);
    }
    if (e >= 31) return (uint16_t) (sign | 0x7c00u);
    uint32_t h = ((uint32_t) e << 10) | (m >> 13);
    const uint32_t rem = m & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;
    return (uint16_t) (sign | h);
}

/// `n` values as ggml's Q8_0 blocks: one half-precision d = amax / 127 per 32 values, then 32 int8 codes.
std::vector<uint8_t> q8_0_of(const float* x, uint64_t n) {
    std::vector<uint8_t> out((size_t) (n / 32 * 34));
    for (uint64_t b = 0; b < n / 32; ++b) {
        float amax = 0.0f;
        for (int j = 0; j < 32; ++j) amax = (std::max)(amax, std::fabs(x[b * 32 + j]));
        const float d = amax / 127.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
        uint8_t* o = out.data() + b * 34;
        const uint16_t hb = half_bits(d);
        std::memcpy(o, &hb, 2);
        for (int j = 0; j < 32; ++j) o[2 + j] = (uint8_t) (int8_t) std::lround(x[b * 32 + j] * id);
    }
    return out;
}

/// The Q8_0 blocks back to floats: the weights the reference multiplies (the values the FP4 planes were built from).
std::vector<float> f32_of_q8(const uint8_t* blocks, uint64_t n) {
    std::vector<float> w((size_t) n);
    for (uint64_t b = 0; b < n / 32; ++b) {
        uint16_t hb;
        std::memcpy(&hb, blocks + b * 34, 2);
        const uint32_t u = (uint32_t) hb << 16;
        float d;
        std::memcpy(&d, &u, 4);
        for (int j = 0; j < 32; ++j) w[b * 32 + j] = d * (float) (int8_t) blocks[b * 34 + 2 + j];   // le stride du bloc est 34 (2 octets d'echelle + 32 codes), pas 32
    }
    return w;
}

template <class T>
T* upload(const std::vector<T>& v) {
    T* d = nullptr;
    if (cudaMalloc(&d, v.size() * sizeof(T)) != cudaSuccess) { cudaGetLastError(); return nullptr; }
    cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}

template <class T>
std::vector<T> download(const T* d, size_t n) {
    std::vector<T> v(n);
    cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
    return v;
}

struct Form {
    const char* suffix;    // a dense projection's GGUF suffix (the list src/core/native_dense.cpp:164 uses)
    int64_t K, N;
    const char* shape;
};

const Form kForms[] = {
    {".attn_qkv.weight", 2560, 10240, "2560->10240"},
    {".attn_gate.weight", 2560, 6144, "2560->6144"},
    {".ssm_out.weight", 6144, 2560, "6144->2560"},
    {".attn_q.weight", 2560, 12288, "2560->12288"},
    {".attn_k.weight", 2560, 512, "2560->512"},
};

const strata::TensorInfo* find_tensor(const strata::GgufFile& g, const Form& f) {
    for (const auto& t : g.tensors())
        if (t.name.rfind("blk.", 0) == 0 && t.name.ends_with(f.suffix) && t.shape.size() == 2 &&
            (int64_t) t.shape[0] == f.K && (int64_t) t.shape[1] == f.N)
            return &t;
    return nullptr;
}

// one form, one token count: planes from the model's weights, the product on the card, the reference in double
void run_form(const strata::GgufFile& g, const Form& f, int64_t T, bool& all) {
    const strata::TensorInfo* t = find_tensor(g, f);
    if (t == nullptr) {
        std::printf("%s T=%lld: no tensor of that shape in the GGUF - the form is NOT measured\n", f.shape, (long long) T);
        all = false;
        return;
    }
    const int64_t K = f.K, N = f.N;
    const uint64_t n = (uint64_t) K * (uint64_t) N;
    if (!sq::can_dequant((sq::Type) t->type) || sq::row_bytes((int) t->type, K) == 0) {
        std::printf("%s T=%lld: type %s cannot be dequantized - the form is NOT measured\n", f.shape, (long long) T,
                    t->type_name());
        all = false;
        return;
    }
    const size_t rb = sq::row_bytes((int) t->type, K);
    const uint8_t* src = g.tensor_data(*t);
    std::vector<float> w32((size_t) n);
    double wabs = 0.0;
    for (int64_t r = 0; r < N; ++r) {
        if (!sq::dequant_row((sq::Type) t->type, src + r * rb, w32.data() + (size_t) r * K, K)) {
            std::printf("%s T=%lld: row %lld did not dequantize\n", f.shape, (long long) T, (long long) r);
            all = false;
            return;
        }
    }
    for (size_t i = 0; i < w32.size(); ++i) wabs += std::fabs((double) w32[i]);
    if (wabs == 0.0) {   // a tensor that read as zeros makes every product zero: that is not a parity, it is nothing to compare
        std::printf("%s T=%lld: %s row 0 of %s read as all zeros - the form is NOT measured\n", f.shape, (long long) T,
                    t->name.c_str(), t->type_name());
        all = false;
        return;
    }
    std::printf("  %s (%s) rows %lld x K %lld: sum of |values| %.1f, first row %.4f %.4f %.4f\n", t->name.c_str(),
                t->type_name(), (long long) N, (long long) K, wabs, w32[0], w32[1], w32[2]);
    const std::vector<uint8_t> q8 = q8_0_of(w32.data(), n);
    w32.clear();
    w32.shrink_to_fit();
    const std::vector<float> wref = f32_of_q8(q8.data(), n);   // the reference's weights: exactly what from_q8 reads

    // the rows: the same values for every form, a fixed seed, so the numbers of two runs are comparable
    std::vector<float> x((size_t) T * K);
    std::mt19937 rng(7777);
    std::normal_distribution<float> g01(0.0f, 1.0f);
    for (size_t i = 0; i < x.size(); ++i) x[i] = 2.0f * g01(rng);
    // **la source des activations doit etre sur la carte** : `fp4_dense::quantize` est un kernel CUDA, il lit `x` sur
    // le peripherique. Passer `x.data()` (la memoire de l'hote) declenche un `illegal memory access` et empoisonne le
    // contexte - c'est ce qui a fait echouer toute la parite le 2026-11-16, avec l'echelle lue a zero.
    float* dx = nullptr;

    uint8_t* dq8 = nullptr;
    uint8_t* planes = nullptr;
    float* wscale = nullptr;
    void* act = nullptr;
    float* gs = nullptr;
    float* Y = nullptr;
    const cudaError_t fails[] = {cudaMalloc(&dq8, q8.size()), cudaMalloc(&planes, np::bytes(N, K)),
                                 cudaMalloc((void**) &wscale, 4), cudaMalloc(&act, fd::act_bytes(T, K)),
                                 cudaMalloc((void**) &gs, (size_t) T * sizeof(float)),
                                 cudaMalloc((void**) &dx, x.size() * sizeof(float)),
                                 cudaMalloc((void**) &Y, (size_t) T * N * sizeof(float))};
    const char* bad = nullptr;
    for (const cudaError_t& e : fails) if (e != cudaSuccess) bad = cudaGetErrorString(e);
    if (bad != nullptr) {
        // la cause de l'echec doit etre dite : `the device buffers did not fit` sans le code CUDA ne prouve rien,
        // et un echec d'allocation laisse le produit a zero, ce qui ferait une parite vide.
        std::printf("%s T=%lld: the device buffers did not fit (%s) - the form is NOT measured\n", f.shape, (long long) T, bad);
        all = false;
    } else {
        // **l'upload se fait sur le tampon alloue, pas sur un tampon que cudaMalloc remplace ensuite** : le 16
        // novembre, `upload(q8)` puis `cudaMalloc(&dq8, ...)` a laisse `from_q8` lire un tampon neuf et vide, d'ou
        // l'echelle a 1.0, la sortie FP4 a zero, et un PASS qui comparait deux zeros.
        cudaMemcpy(dq8, q8.data(), q8.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(dx, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice);
        np::from_q8(dq8, N, K, planes, wscale, nullptr);
        fd::quantize(dx, T, K, K, act, gs, nullptr);
        fd::product(planes, wscale, N, K, act, gs, T, Y, N, false, nullptr);
        const cudaError_t le = cudaGetLastError();
        cudaDeviceSynchronize();
        const std::vector<float> y = download(Y, (size_t) T * N);
        const float s = download(wscale, 1)[0];
        if (le != cudaSuccess || s <= 0.0f) {
            // **une echelle a zero rend la parite vide** : le produit FP4 est multiplie par cette echelle, et la
            // reference aussi, donc les deux cotes sortent a zero et la moyenne des erreurs relatives est 0. C'est
            // ce qui a donne un PASS sans comparaison reelle le 2026-11-16. Un zero n'est pas une parite.
            std::printf("%s T=%lld: the FP4 product did not run (scale %.6e, %s) - the form is NOT measured\n",
                        f.shape, (long long) T, s, le != cudaSuccess ? cudaGetErrorString(le) : "kernels ok");
            all = false;
            cudaFree(dq8); cudaFree(planes); cudaFree(wscale); cudaFree(act); cudaFree(gs); cudaFree(Y);
            return;
        }

        // the reference: Y[t][r] = s * sum_k wref[r][k] * x[t][k], in double, split by weight rows
        std::vector<double> ref((size_t) T * N, 0.0);
        const unsigned nt = (std::min)(16u, (std::max)(1u, std::thread::hardware_concurrency()));
        std::vector<std::thread> pool;
        for (unsigned i = 0; i < nt; ++i) {
            const int64_t r0 = N * (int64_t) i / nt, r1 = N * (int64_t) (i + 1) / nt;
            pool.emplace_back([&, r0, r1] {
                for (int64_t r = r0; r < r1; ++r) {
                    const float* wr = wref.data() + (size_t) r * K;
                    for (int64_t tt = 0; tt < T; ++tt) {
                        const float* xr = x.data() + (size_t) tt * K;
                        double sum = 0.0;
                        for (int64_t k = 0; k < K; ++k) sum += (double) wr[k] * (double) xr[k];
                        ref[(size_t) tt * N + r] = (double) s * sum;
                    }
                }
            });
        }
        for (auto& th : pool) th.join();

        double sum_rel = 0.0, max_rel = 0.0, num = 0.0, den = 0.0, mean_abs = 0.0;
        std::vector<double> sref((size_t) T, 0.0), sfar((size_t) T, 0.0);
        for (int64_t tt = 0; tt < T; ++tt)
            for (int64_t r = 0; r < N; ++r) {
                const double a = ref[(size_t) tt * N + r], b = (double) y[(size_t) tt * N + r];
                const double d = std::fabs(b - a);
                sum_rel += d / (std::fabs(a) + 1e-6);
                max_rel = (std::max)(max_rel, d / (std::fabs(a) + 1e-6));
                num += d * d;
                den += a * a;
                mean_abs += std::fabs(a);
                sref[(size_t) tt] += a;
                sfar[(size_t) tt] += b;
            }
        double row_sum = 0.0, row_max = 0.0;
        for (int64_t tt = 0; tt < T; ++tt) {
            const double d = std::fabs(sfar[(size_t) tt] - sref[(size_t) tt]);
            row_sum += d / (std::fabs(sref[(size_t) tt]) + 1e-6);
            row_max = (std::max)(row_max, d / (std::fabs(sref[(size_t) tt]) + 1e-6));
        }
        const double mean_rel = 100.0 * sum_rel / ((double) T * (double) N);
        const double resid = 100.0 * std::sqrt(num) / (std::sqrt(den) + 1e-12);
        const double row_rel = 100.0 * row_sum / (double) T;
        const bool pass = mean_rel < 1.0;
        all = all && pass;
        std::printf("%s T=%4lld (%s, %.1f MiB of weights): scale %.6e  mean rel %.3f %%  max rel %.3f %%  resid norm %.3f %%  "
                    "row sums: mean rel %.3f %% max %.3f %%  first row ref %.4e fp4 %.4e  mean |ref| %.4e  -> %s\n",
                    f.shape, (long long) T, t->type_name(), (double) q8.size() / 1048576.0, s, mean_rel, 100.0 * max_rel,
                    resid, row_rel, 100.0 * row_max, sref[0], sfar[0], mean_abs / ((double) T * (double) N), pass ? "PASS" : "FAIL");
        std::fflush(stdout);
    }
    cudaFree(dq8); cudaFree(planes); cudaFree(wscale); cudaFree(act); cudaFree(gs); cudaFree(Y); cudaFree(dx);
}

}  // namespace

int main(int argc, char** argv) {
    std::string gguf;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--gguf") == 0 && i + 1 < argc) gguf = argv[++i];
    }
    // `--probe` : une matrice dont les valeurs sont connues (100.0 et -50.0 par bloc de 32), pour verifier ce que
    // `from_q8` pose comme echelle. La matrice est [128 x 256] (la plus petite forme que le kernel prend). Une
    // echelle correcte est amax / (6 * 448) = 100 / 2688 = 3.7202e-2. Un 1.0 veut dire que l'amax est reste a zero.
    // `--probe-gguf <shard>` : le chemin du test seul, sur le premier formulaire : dequant des poids reels, Q8_0 sur
    // l'hote, upload, `from_q8`. Si l'echelle sort a 1.0, l'amax est reste a zero, et la cause est dans les blocs
    // que le test construit - pas dans le kernel (la sonde `--probe` le prouve).
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--probe-gguf") == 0 && i + 1 < argc) {
            strata::GgufFile g(argv[i + 1]);
            const Form& f = kForms[0];
            const strata::TensorInfo* t = find_tensor(g, f);
            if (t == nullptr) { std::printf("probe-gguf: no %s tensor\n", f.suffix); return 1; }
            const int64_t K = f.K, N = f.N;
            const uint64_t n = (uint64_t) K * (uint64_t) N;
            const size_t rb = strata::quant::row_bytes((int) t->type, K);
            const uint8_t* src = g.tensor_data(*t);
            std::vector<float> w32((size_t) n);
            for (int64_t r = 0; r < N; ++r)
                if (!strata::quant::dequant_row((strata::quant::Type) t->type, src + r * rb, w32.data() + (size_t) r * K, K))
                    { std::printf("probe-gguf: row %lld did not dequantize\n", (long long) r); return 1; }
            double wabs = 0.0; for (size_t i = 0; i < w32.size(); ++i) wabs += std::fabs((double) w32[i]);
            const std::vector<uint8_t> q8 = q8_0_of(w32.data(), n);
            std::printf("probe-gguf: sum |values| %.1f, q8 bytes %llu, premier bloc d %.6f codes %d %d %d\n",
                        wabs, (unsigned long long) q8.size(),
                        (float) (uint16_t) (q8[0] | (q8[1] << 8)), (int8_t) q8[2], (int8_t) q8[3], (int8_t) q8[4]);
            uint8_t* dq8 = upload(q8);
            void* planes = nullptr;
            float* scale = nullptr;
            if (!dq8 || cudaMalloc(&planes, np::bytes(N, K)) != cudaSuccess || cudaMalloc((void**) &scale, 4) != cudaSuccess) {
                std::printf("probe-gguf: buffers did not fit (%s)\n", cudaGetErrorString(cudaGetLastError())); return 1;
            }
            np::from_q8(dq8, N, K, planes, scale, nullptr);
            const cudaError_t le = cudaGetLastError();
            cudaDeviceSynchronize();
            std::printf("probe-gguf: scale %.6e (1.0 veut dire amax a zero), %s\n", download(scale, 1)[0],
                        le != cudaSuccess ? cudaGetErrorString(le) : "kernels ok");
            return 0;
        }
    }
    // `--probe-q8 <shard>` : l'aller-retour bloc seul. Le premier bloc Q8_0 des poids reels est ecrit par `q8_0_of`,
    // relu par `f32_of_q8`, et compare aux valeurs dequantifiees. Si les deux ne correspondent pas, le format du
    // bloc dans le test est faux - et la parite ne mesure rien.
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--probe-q8") == 0 && i + 1 < argc) {
            strata::GgufFile g(argv[i + 1]);
            const Form& f = kForms[0];
            const strata::TensorInfo* t = find_tensor(g, f);
            if (t == nullptr) { std::printf("probe-q8: no %s tensor\n", f.suffix); return 1; }
            const int64_t K = f.K, N = f.N;
            const uint64_t n = (uint64_t) K * (uint64_t) N;
            const size_t rb = strata::quant::row_bytes((int) t->type, K);
            const uint8_t* src = g.tensor_data(*t);
            std::vector<float> w32((size_t) n);
            for (int64_t r = 0; r < N; ++r)
                strata::quant::dequant_row((strata::quant::Type) t->type, src + r * rb, w32.data() + (size_t) r * K, K);
            const std::vector<uint8_t> q8 = q8_0_of(w32.data(), n);
            const std::vector<float> wref = f32_of_q8(q8.data(), n);
            float amax = 0.0f; for (int j = 0; j < 32; ++j) amax = (std::max)(amax, std::fabs(w32[j]));
            const uint16_t hb = (uint16_t) (q8[0] | (q8[1] << 8));
            double wabs = 0.0; for (size_t i = 0; i < wref.size(); ++i) wabs += std::fabs((double) wref[i]);
            double wmax = 0.0; for (size_t i = 0; i < wref.size(); ++i) wmax = (std::max)(wmax, std::fabs((double) wref[i]));
            std::printf("probe-q8: echelle bits 0x%04x = %.6f, amax des 32 premieres valeurs %.6f, attendu d = %.6f\n",
                        hb, (float) hb, amax, amax / 127.0f);
            // les octets bruts du bloc 0: les deux premiers sont l'echelle, les 32 suivants les codes
            std::printf("probe-q8: octets 0..5 %d %d %d %d %d %d, octets 2..4 lus comme codes int8 %d %d %d\n",
                        q8[0], q8[1], q8[2], q8[3], q8[4], q8[5], (int) (int8_t) q8[2], (int) (int8_t) q8[3], (int) (int8_t) q8[4]);
            std::printf("probe-q8: valeurs 0..3: %.6f %.6f %.6f ; reconstruites: %.6f %.6f %.6f\n",
                        w32[0], w32[1], w32[2], wref[0], wref[1], wref[2]);
            std::printf("probe-q8: somme |reconstruit| %.1f, max |reconstruit| %.6f, max |poids| %.6f\n",
                        wabs, wmax, [&]{ double m = 0; for (size_t i = 0; i < w32.size(); ++i) m = (std::max)(m, std::fabs((double) w32[i])); return m; }());
            return 0;
        }
    }
    bool probe = false;
    for (int i = 1; i < argc; ++i) probe = probe || std::strcmp(argv[i], "--probe") == 0;
    if (probe) {
        const int64_t rows = 128, K = 256;
        std::vector<float> w((size_t) rows * K, 0.0f);
        for (size_t i = 0; i < w.size(); ++i) w[i] = (i % 2 == 0) ? 100.0f : -50.0f;
        const std::vector<uint8_t> q8 = q8_0_of(w.data(), w.size());
        uint8_t* dq8 = upload(q8);
        void* planes = nullptr;
        float* scale = nullptr;
        if (!dq8 || cudaMalloc(&planes, np::bytes(rows, K)) != cudaSuccess || cudaMalloc((void**) &scale, 4) != cudaSuccess) {
            std::printf("probe: the buffers did not fit (%s)\n", cudaGetErrorString(cudaGetLastError()));
            return 1;
        }
        np::from_q8(dq8, rows, K, planes, scale, nullptr);
        const cudaError_t le = cudaGetLastError();
        cudaDeviceSynchronize();
        const float s = download(scale, 1)[0];
        std::printf("probe: scale %.6e (attendu 3.7202e-2), %s\n", s, le != cudaSuccess ? cudaGetErrorString(le) : "kernels ok");
        return 0;
    }
    int ndev = 0;
    cudaGetDeviceCount(&ndev);
    if (gguf.empty()) { std::printf("dense fp4 parity: no --gguf PATH given, nothing to compare\n"); return 77; }
    if (ndev == 0) { std::printf("dense fp4 parity: no CUDA device\n"); return 77; }
    strata::GgufFile g(gguf);
    std::printf("dense fp4 parity: %s\n", gguf.c_str());
    bool all = true;
    for (const Form& f : kForms)
        for (int64_t T : {256ll, 1024ll}) run_form(g, f, T, all);
    if (!all) { std::printf("dense fp4 parity: FAILED (a form's mean relative error is 1 %% or more, or a form was not measured)\n"); return 1; }
    std::printf("dense fp4 parity: every form's mean relative error is under 1 %%\n");
    return 0;
}
