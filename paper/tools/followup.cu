// Follow-up experiments for the paper (rev.txt items 8-10):
//   followup timing            - FAST vs LUT kernel wall time per FFT size
//   followup conv              - multiprecision convolution driven to digit failure
//   followup iter2d            - iterated 2D FFT round trips, error growth
// Radix-2 DIT FFT identical to experiment1.cu; only twiddle source differs.
#include <cuda_runtime.h>
#include <math_constants.h>
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        cudaError_t err__ = (expr);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::cerr << "CUDA error: " << cudaGetErrorString(err__) << " at "  \
                      << __FILE__ << ":" << __LINE__ << "\n";                   \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

namespace fft {

enum class TwiddleSource { kLUT, kFastApprox };

inline int log2_int(size_t n) {
    int bits = 0;
    while ((size_t(1) << bits) < n) ++bits;
    return bits;
}

__device__ __forceinline__ unsigned reverse_bits(unsigned v, int bits) {
    return __brev(v) >> (32 - bits);
}

__global__ void bit_reverse_kernel(float2* data, int n, int bits) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    unsigned reversed = reverse_bits(static_cast<unsigned>(idx), bits);
    if (reversed > static_cast<unsigned>(idx)) {
        float2 tmp = data[idx];
        data[idx] = data[reversed];
        data[reversed] = tmp;
    }
}

__host__ __device__ __forceinline__ float2 complex_mul(const float2& a, const float2& b) {
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

template <TwiddleSource Source>
__global__ void fft_stage_kernel(float2* data, const float2* twiddles, int n,
                                 int half_size, int stride, bool inverse) {
    int total = n >> 1;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    int group = tid / half_size;
    int j = tid - group * half_size;
    int span = half_size << 1;
    int offset = group * span + j;
    int mate = offset + half_size;
    float2 even = data[offset];
    float2 odd = data[mate];
    int exponent_index = j * stride;
    float2 tw;
    if constexpr (Source == TwiddleSource::kLUT) {
        tw = twiddles[exponent_index];
        if (inverse) tw.y = -tw.y;
    } else {
        float base = (inverse ? 1.0f : -1.0f) * (2.0f * CUDART_PI_F / static_cast<float>(span));
        float angle = base * static_cast<float>(j);
        float s, c;
        __sincosf(angle, &s, &c);
        tw = make_float2(c, s);
    }
    float2 t = complex_mul(tw, odd);
    data[offset] = make_float2(even.x + t.x, even.y + t.y);
    data[mate] = make_float2(even.x - t.x, even.y - t.y);
}

__global__ void normalize_kernel(float2* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float scale = 1.0f / static_cast<float>(n);
    data[idx].x *= scale;
    data[idx].y *= scale;
}

__global__ void pointwise_multiply_kernel(float2* a, const float2* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    a[idx] = complex_mul(a[idx], b[idx]);
}

inline void launch_fft(float2* data, int n, bool inverse, TwiddleSource src,
                       const float2* twiddles = nullptr) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    int bits = log2_int(n);
    bit_reverse_kernel<<<blocks, threads>>>(data, n, bits);
    for (int stage = 0; stage < bits; ++stage) {
        int half_size = 1 << stage;
        int stride = n >> (stage + 1);
        int total = n >> 1;
        int stage_blocks = (total + threads - 1) / threads;
        if (src == TwiddleSource::kLUT)
            fft_stage_kernel<TwiddleSource::kLUT><<<stage_blocks, threads>>>(
                data, twiddles, n, half_size, stride, inverse);
        else
            fft_stage_kernel<TwiddleSource::kFastApprox><<<stage_blocks, threads>>>(
                data, nullptr, n, half_size, stride, inverse);
    }
    if (inverse) normalize_kernel<<<blocks, threads>>>(data, n);
    CUDA_CHECK(cudaGetLastError());
}

// Batched row FFT for 2D transforms: each row of a rows x n matrix.
inline void launch_fft_rows(float2* data, int rows, int n, bool inverse,
                            TwiddleSource src, const float2* twiddles) {
    for (int r = 0; r < rows; ++r)
        launch_fft(data + static_cast<size_t>(r) * n, n, inverse, src, twiddles);
}

__global__ void transpose_kernel(const float2* in, float2* out, int n) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= n || y >= n) return;
    out[static_cast<size_t>(x) * n + y] = in[static_cast<size_t>(y) * n + x];
}

std::vector<float2> make_twiddles(size_t n) {
    std::vector<float2> host(n);
    for (size_t k = 0; k < n; ++k) {
        double angle = -2.0 * M_PI * static_cast<double>(k) / static_cast<double>(n);
        host[k].x = static_cast<float>(std::cos(angle));
        host[k].y = static_cast<float>(std::sin(angle));
    }
    return host;
}

std::vector<std::complex<double>> cpu_fft(std::vector<std::complex<double>> data, bool inverse) {
    const size_t n = data.size();
    int bits = log2_int(n);
    for (size_t i = 0; i < n; ++i) {
        unsigned r = 0; unsigned v = static_cast<unsigned>(i);
        for (int b = 0; b < bits; ++b) { r = (r << 1) | (v & 1u); v >>= 1; }
        if (r > i) std::swap(data[i], data[r]);
    }
    for (size_t len = 2; len <= n; len <<= 1) {
        double angle = (inverse ? 2.0 : -2.0) * M_PI / static_cast<double>(len);
        std::complex<double> wlen(std::cos(angle), std::sin(angle));
        for (size_t i = 0; i < n; i += len) {
            std::complex<double> w(1.0, 0.0);
            size_t half = len >> 1;
            for (size_t j = 0; j < half; ++j) {
                auto u = data[i + j];
                auto v2 = data[i + j + half] * w;
                data[i + j] = u + v2;
                data[i + j + half] = u - v2;
                w *= wlen;
            }
        }
    }
    if (inverse) for (auto& v : data) v /= static_cast<double>(n);
    return data;
}

}  // namespace fft

using fft::TwiddleSource;

// ---------------------------------------------------------------- timing ----
void run_timing() {
    std::printf("log2n,n,lut_ms,fast_ms,fast_over_lut,lut_table_bytes\n");
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (int lg = 10; lg <= 24; lg += 2) {
        size_t n = size_t(1) << lg;
        std::vector<float2> host(n);
        for (auto& v : host) { v.x = dist(rng); v.y = dist(rng); }
        float2 *d_data, *d_tw;
        CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(float2)));
        CUDA_CHECK(cudaMalloc(&d_tw, n * sizeof(float2)));
        auto tw = fft::make_twiddles(n);
        CUDA_CHECK(cudaMemcpy(d_tw, tw.data(), n * sizeof(float2), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_data, host.data(), n * sizeof(float2), cudaMemcpyHostToDevice));

        const int warmup = 5;
        const int iters = (lg <= 16) ? 200 : 50;
        double ms[2] = {0, 0};
        for (int cfg = 0; cfg < 2; ++cfg) {
            TwiddleSource src = cfg == 0 ? TwiddleSource::kLUT : TwiddleSource::kFastApprox;
            for (int i = 0; i < warmup; ++i)
                fft::launch_fft(d_data, static_cast<int>(n), false, src, d_tw);
            CUDA_CHECK(cudaDeviceSynchronize());
            cudaEvent_t t0, t1;
            CUDA_CHECK(cudaEventCreate(&t0));
            CUDA_CHECK(cudaEventCreate(&t1));
            CUDA_CHECK(cudaEventRecord(t0));
            for (int i = 0; i < iters; ++i)
                fft::launch_fft(d_data, static_cast<int>(n), false, src, d_tw);
            CUDA_CHECK(cudaEventRecord(t1));
            CUDA_CHECK(cudaEventSynchronize(t1));
            float total = 0;
            CUDA_CHECK(cudaEventElapsedTime(&total, t0, t1));
            ms[cfg] = total / iters;
            cudaEventDestroy(t0); cudaEventDestroy(t1);
        }
        std::printf("%d,%zu,%.5f,%.5f,%.4f,%zu\n", lg, n, ms[0], ms[1],
                    ms[1] / ms[0], n * sizeof(float2));
        std::fflush(stdout);
        cudaFree(d_data); cudaFree(d_tw);
    }
}

// ------------------------------------------------------------------ conv ----
struct ConvResult { double mean_abs; double max_abs; };

ConvResult gpu_conv_error(const std::vector<int>& a, const std::vector<int>& b,
                          size_t fft_size, TwiddleSource src, const float2* d_tw,
                          const std::vector<double>& reference, size_t conv_len) {
    size_t n = fft_size;
    std::vector<float2> ha(n, make_float2(0, 0)), hb(n, make_float2(0, 0));
    for (size_t i = 0; i < a.size(); ++i) ha[i].x = static_cast<float>(a[i]);
    for (size_t i = 0; i < b.size(); ++i) hb[i].x = static_cast<float>(b[i]);
    float2 *d_a, *d_b;
    CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(float2)));
    CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(float2)));
    CUDA_CHECK(cudaMemcpy(d_a, ha.data(), n * sizeof(float2), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, hb.data(), n * sizeof(float2), cudaMemcpyHostToDevice));
    fft::launch_fft(d_a, static_cast<int>(n), false, src, d_tw);
    fft::launch_fft(d_b, static_cast<int>(n), false, src, d_tw);
    int threads = 256, blocks = static_cast<int>((n + threads - 1) / threads);
    fft::pointwise_multiply_kernel<<<blocks, threads>>>(d_a, d_b, static_cast<int>(n));
    fft::launch_fft(d_a, static_cast<int>(n), true, src, d_tw);
    std::vector<float2> out(n);
    CUDA_CHECK(cudaMemcpy(out.data(), d_a, n * sizeof(float2), cudaMemcpyDeviceToHost));
    cudaFree(d_a); cudaFree(d_b);
    double sum = 0, mx = 0;
    for (size_t i = 0; i < conv_len; ++i) {
        double err = std::abs(static_cast<double>(out[i].x) - reference[i]);
        sum += err;
        mx = std::max(mx, err);
    }
    return {sum / static_cast<double>(conv_len), mx};
}

void run_conv(const std::vector<int>& digit_list) {
    std::printf("digits,fft_size,lut_mean_abs,lut_max_abs,fast_mean_abs,fast_max_abs\n");
    for (int digits : digit_list) {
        std::mt19937 rng(1337u + static_cast<unsigned>(digits));
        std::uniform_int_distribution<int> dist(0, 9);
        std::vector<int> a(digits), b(digits);
        for (auto& v : a) v = dist(rng);
        for (auto& v : b) v = dist(rng);
        if (digits > 0) { a.back() = std::max(1, a.back()); b.back() = std::max(1, b.back()); }
        size_t conv_len = a.size() + b.size();
        size_t fft_size = 1;
        while (fft_size < conv_len) fft_size <<= 1;

        // CPU double reference via same-radix FFT convolution
        std::vector<std::complex<double>> ca(fft_size), cb(fft_size);
        for (size_t i = 0; i < a.size(); ++i) ca[i] = static_cast<double>(a[i]);
        for (size_t i = 0; i < b.size(); ++i) cb[i] = static_cast<double>(b[i]);
        auto fa = fft::cpu_fft(std::move(ca), false);
        auto fb = fft::cpu_fft(std::move(cb), false);
        for (size_t i = 0; i < fft_size; ++i) fa[i] *= fb[i];
        auto conv = fft::cpu_fft(std::move(fa), true);
        std::vector<double> reference(conv_len);
        for (size_t i = 0; i < conv_len; ++i) reference[i] = conv[i].real();

        auto tw = fft::make_twiddles(fft_size);
        float2* d_tw;
        CUDA_CHECK(cudaMalloc(&d_tw, fft_size * sizeof(float2)));
        CUDA_CHECK(cudaMemcpy(d_tw, tw.data(), fft_size * sizeof(float2), cudaMemcpyHostToDevice));

        auto lut = gpu_conv_error(a, b, fft_size, TwiddleSource::kLUT, d_tw, reference, conv_len);
        auto fast = gpu_conv_error(a, b, fft_size, TwiddleSource::kFastApprox, nullptr, reference, conv_len);
        cudaFree(d_tw);
        std::printf("%d,%zu,%.6e,%.6e,%.6e,%.6e\n", digits, fft_size,
                    lut.mean_abs, lut.max_abs, fast.mean_abs, fast.max_abs);
        std::fflush(stdout);
    }
}

// ---------------------------------------------------------------- iter2d ----
// Synthetic 256x256 grayscale test image in [0,1]: sharp vertical edge plus a
// diagonal gradient (edge-dominated content — the worst case in Table 6).
std::vector<float> make_image(int n) {
    std::vector<float> img(static_cast<size_t>(n) * n);
    for (int y = 0; y < n; ++y)
        for (int x = 0; x < n; ++x) {
            float v = (x < n / 2) ? 0.15f : 0.85f;
            v += 0.10f * (static_cast<float>(x + y) / (2.0f * (n - 1)));
            img[static_cast<size_t>(y) * n + x] = v;
        }
    return img;
}

void fft2d(float2* d_data, float2* d_tmp, int n, bool inverse, TwiddleSource src,
           const float2* d_tw) {
    dim3 bt(16, 16), gt((n + 15) / 16, (n + 15) / 16);
    fft::launch_fft_rows(d_data, n, n, inverse, src, d_tw);
    fft::transpose_kernel<<<gt, bt>>>(d_data, d_tmp, n);
    fft::launch_fft_rows(d_tmp, n, n, inverse, src, d_tw);
    fft::transpose_kernel<<<gt, bt>>>(d_tmp, d_data, n);
    CUDA_CHECK(cudaGetLastError());
}

void run_iter2d(int n, int max_iters) {
    auto img = make_image(n);
    size_t total = static_cast<size_t>(n) * n;
    auto tw = fft::make_twiddles(n);
    float2* d_tw;
    CUDA_CHECK(cudaMalloc(&d_tw, n * sizeof(float2)));
    CUDA_CHECK(cudaMemcpy(d_tw, tw.data(), n * sizeof(float2), cudaMemcpyHostToDevice));

    std::printf("config,iters,max_abs,psnr_db\n");
    std::vector<int> checkpoints{1, 2, 5, 10, 20, 50, 100};
    for (int cfg = 0; cfg < 2; ++cfg) {
        TwiddleSource src = cfg == 0 ? TwiddleSource::kLUT : TwiddleSource::kFastApprox;
        std::vector<float2> host(total);
        for (size_t i = 0; i < total; ++i) host[i] = make_float2(img[i], 0.f);
        float2 *d_data, *d_tmp;
        CUDA_CHECK(cudaMalloc(&d_data, total * sizeof(float2)));
        CUDA_CHECK(cudaMalloc(&d_tmp, total * sizeof(float2)));
        CUDA_CHECK(cudaMemcpy(d_data, host.data(), total * sizeof(float2), cudaMemcpyHostToDevice));
        int done = 0;
        for (int cp : checkpoints) {
            if (cp > max_iters) break;
            for (; done < cp; ++done) {
                fft2d(d_data, d_tmp, n, false, src, d_tw);
                fft2d(d_data, d_tmp, n, true, src, d_tw);
            }
            CUDA_CHECK(cudaMemcpy(host.data(), d_data, total * sizeof(float2), cudaMemcpyDeviceToHost));
            double mx = 0, mse = 0;
            for (size_t i = 0; i < total; ++i) {
                double err = std::abs(static_cast<double>(host[i].x) - img[i]);
                mx = std::max(mx, err);
                mse += err * err;
            }
            mse /= static_cast<double>(total);
            double psnr = 10.0 * std::log10(1.0 / mse);
            std::printf("%s%d,%.6e,%.2f\n", cfg == 0 ? "lut," : "fast,", cp, mx, psnr);
            std::fflush(stdout);
        }
        cudaFree(d_data); cudaFree(d_tmp);
    }
    cudaFree(d_tw);
}

// ------------------------------------------------------------------ main ----
int main(int argc, char** argv) {
    std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "timing") {
        run_timing();
    } else if (mode == "conv") {
        std::vector<int> digits{2048, 8192, 16384, 32768, 65536, 131072,
                                262144, 524288, 1048576};
        if (argc > 2) {
            digits.clear();
            std::string list = argv[2];
            size_t pos = 0;
            while (pos < list.size()) {
                size_t comma = list.find(',', pos);
                if (comma == std::string::npos) comma = list.size();
                digits.push_back(std::stoi(list.substr(pos, comma - pos)));
                pos = comma + 1;
            }
        }
        run_conv(digits);
    } else if (mode == "iter2d") {
        int n = argc > 2 ? std::stoi(argv[2]) : 256;
        int iters = argc > 3 ? std::stoi(argv[3]) : 100;
        run_iter2d(n, iters);
    } else {
        std::cerr << "usage: followup timing|conv [digits]|iter2d [n] [iters]\n";
        return 1;
    }
    return 0;
}
