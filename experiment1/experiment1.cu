#include <cuda_runtime.h>
#include <math_constants.h>
#include <algorithm>
#include <cmath>
#include <complex>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <functional>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define CUDA_CHECK(expr)                                                                 \
    do {                                                                                 \
        cudaError_t err__ = (expr);                                                      \
        if (err__ != cudaSuccess) {                                                      \
            std::cerr << "CUDA error: " << cudaGetErrorString(err__) << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            std::exit(EXIT_FAILURE);                                                     \
        }                                                                                \
    } while (0)

struct DeviceBuffer {
    void* ptr = nullptr;
    size_t bytes = 0;
    ~DeviceBuffer() { release(); }
    void allocate(size_t nbytes) {
        if (ptr && bytes == nbytes) return;
        release();
        bytes = nbytes;
        CUDA_CHECK(cudaMalloc(&ptr, bytes));
    }
    void release() {
        if (ptr) {
            cudaFree(ptr);
            ptr = nullptr;
            bytes = 0;
        }
    }
    template <typename T> T* as() { return reinterpret_cast<T*>(ptr); }
};

namespace fft {

enum class TwiddleSource { kLUT, kFastApprox };

inline int log2_int(int n) {
    int bits = 0;
    while ((1 << bits) < n) ++bits;
    return bits;
}

inline unsigned reverse_bits_host(unsigned v, int bits) {
    unsigned r = 0;
    for (int i = 0; i < bits; ++i) {
        r = (r << 1) | (v & 1u);
        v >>= 1;
    }
    return r;
}

struct TwiddleTable {
    std::vector<float2> host;
    DeviceBuffer device;
    void generate(int n) {
        host.resize(n);
        const double kTwoPi = -2.0 * M_PI;
        for (int k = 0; k < n; ++k) {
            double angle = kTwoPi * static_cast<double>(k) / static_cast<double>(n);
            host[k].x = static_cast<float>(std::cos(angle));
            host[k].y = static_cast<float>(std::sin(angle));
        }
        device.allocate(host.size() * sizeof(float2));
        CUDA_CHECK(cudaMemcpy(device.ptr, host.data(), host.size() * sizeof(float2), cudaMemcpyHostToDevice));
    }
};

__device__ __forceinline__ unsigned reverse_bits(unsigned v, int bits) {
    v = __brev(v);
    return v >> (32 - bits);
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
__global__ void fft_stage_kernel(float2* data, const float2* twiddles, int n, int half_size, int stride, bool inverse) {
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

inline void launch_fft(float2* data, int n, bool inverse, TwiddleSource src, const float2* twiddles = nullptr) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    int bits = log2_int(n);
    bit_reverse_kernel<<<blocks, threads>>>(data, n, bits);
    CUDA_CHECK(cudaGetLastError());
    int stages = bits;
    for (int stage = 0; stage < stages; ++stage) {
        int half_size = 1 << stage;
        int stride = n >> (stage + 1);
        int total = n >> 1;
        int stage_blocks = (total + threads - 1) / threads;
        if (src == TwiddleSource::kLUT) {
            fft_stage_kernel<TwiddleSource::kLUT><<<stage_blocks, threads>>>(data, twiddles, n, half_size, stride, inverse);
        } else {
            fft_stage_kernel<TwiddleSource::kFastApprox><<<stage_blocks, threads>>>(data, nullptr, n, half_size, stride, inverse);
        }
        CUDA_CHECK(cudaGetLastError());
    }
    if (inverse) {
        normalize_kernel<<<blocks, threads>>>(data, n);
        CUDA_CHECK(cudaGetLastError());
    }
}

inline void launch_pointwise_multiply(float2* a, const float2* b, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    pointwise_multiply_kernel<<<blocks, threads>>>(a, b, n);
    CUDA_CHECK(cudaGetLastError());
}

inline void upload_complex(const std::vector<std::complex<float>>& host, DeviceBuffer& buf) {
    buf.allocate(host.size() * sizeof(float2));
    std::vector<float2> tmp(host.size());
    for (size_t i = 0; i < host.size(); ++i) {
        tmp[i].x = host[i].real();
        tmp[i].y = host[i].imag();
    }
    CUDA_CHECK(cudaMemcpy(buf.ptr, tmp.data(), tmp.size() * sizeof(float2), cudaMemcpyHostToDevice));
}

inline void download_complex(std::vector<std::complex<float>>& host, const DeviceBuffer& buf) {
    std::vector<float2> tmp(host.size());
    CUDA_CHECK(cudaMemcpy(tmp.data(), buf.ptr, tmp.size() * sizeof(float2), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < host.size(); ++i) {
        host[i] = std::complex<float>(tmp[i].x, tmp[i].y);
    }
}

inline std::vector<std::complex<double>> cpu_fft(std::vector<std::complex<double>> data, bool inverse) {
    const size_t n = data.size();
    unsigned bits = log2_int(static_cast<int>(n));
    for (size_t i = 0; i < n; ++i) {
        size_t j = reverse_bits_host(static_cast<unsigned>(i), bits);
        if (j > i) std::swap(data[i], data[j]);
    }
    for (size_t len = 2; len <= n; len <<= 1) {
        double angle = (inverse ? 2.0 : -2.0) * M_PI / static_cast<double>(len);
        std::complex<double> wlen(std::cos(angle), std::sin(angle));
        for (size_t i = 0; i < n; i += len) {
            std::complex<double> w(1.0, 0.0);
            size_t half = len >> 1;
            for (size_t j = 0; j < half; ++j) {
                auto u = data[i + j];
                auto v = data[i + j + half] * w;
                data[i + j] = u + v;
                data[i + j + half] = u - v;
                w *= wlen;
            }
        }
    }
    if (inverse) {
        for (auto& v : data) v /= static_cast<double>(n);
    }
    return data;
}

}  // namespace fft

struct Options {
    std::vector<int> digit_lengths{2048, 8192};
    int base = 10;
    std::string output_dir = "data";
    unsigned seed = 1337;
};

size_t next_pow2(size_t v) {
    size_t n = 1;
    while (n < v) n <<= 1;
    return n;
}

std::vector<int> random_digits(int digits, int base, std::mt19937& rng) {
    std::uniform_int_distribution<int> dist(0, base - 1);
    std::uniform_int_distribution<int> non_zero(1, base - 1);
    std::vector<int> out(digits);
    for (int i = 0; i < digits; ++i) out[i] = dist(rng);
    if (!out.empty()) out.back() = non_zero(rng);
    return out;
}

template <typename T>
std::vector<std::complex<T>> digits_to_complex(const std::vector<int>& digits, size_t fft_size) {
    std::vector<std::complex<T>> out(fft_size, std::complex<T>(0, 0));
    for (size_t i = 0; i < digits.size(); ++i) out[i] = static_cast<T>(digits[i]);
    return out;
}

std::vector<long long> exact_convolution(const std::vector<int>& a, const std::vector<int>& b) {
    std::vector<long long> result(a.size() + b.size(), 0);
    for (size_t i = 0; i < a.size(); ++i) {
        for (size_t j = 0; j < b.size(); ++j) {
            result[i + j] += static_cast<long long>(a[i]) * static_cast<long long>(b[j]);
        }
    }
    return result;
}

std::vector<double> cpu_double_convolution(const std::vector<int>& a, const std::vector<int>& b, size_t fft_size,
                                           std::vector<std::complex<double>>* fft_stage = nullptr,
                                           std::vector<std::complex<double>>* hadamard_stage = nullptr) {
    auto ca = digits_to_complex<double>(a, fft_size);
    auto cb = digits_to_complex<double>(b, fft_size);
    auto fa = fft::cpu_fft(ca, false);
    if (fft_stage) *fft_stage = fa;
    auto fb = fft::cpu_fft(cb, false);
    for (size_t i = 0; i < fa.size(); ++i) fa[i] *= fb[i];
    if (hadamard_stage) *hadamard_stage = fa;
    auto conv = fft::cpu_fft(fa, true);
    std::vector<double> real(conv.size());
    for (size_t i = 0; i < conv.size(); ++i) real[i] = conv[i].real();
    return real;
}

std::vector<std::complex<float>> gpu_convolution(
    const std::vector<std::complex<float>>& a, const std::vector<std::complex<float>>& b, int n, fft::TwiddleSource src,
    const float2* twiddles,
    std::function<void(const std::string&, const std::vector<std::complex<float>>&)> stage_dump = {}) {
    DeviceBuffer d_a;
    DeviceBuffer d_b;
    fft::upload_complex(a, d_a);
    fft::upload_complex(b, d_b);
    fft::launch_fft(d_a.as<float2>(), n, false, src, twiddles);
    std::vector<std::complex<float>> stage_buffer;
    if (stage_dump) stage_buffer.resize(n);
    auto dump_stage = [&](const std::string& stage) {
        if (!stage_dump) return;
        fft::download_complex(stage_buffer, d_a);
        stage_dump(stage, stage_buffer);
    };
    dump_stage("fft_once");
    fft::launch_fft(d_b.as<float2>(), n, false, src, twiddles);
    fft::launch_pointwise_multiply(d_a.as<float2>(), d_b.as<float2>(), n);
    dump_stage("hadamard");
    fft::launch_fft(d_a.as<float2>(), n, true, src, twiddles);
    std::vector<std::complex<float>> host(n);
    fft::download_complex(host, d_a);
    return host;
}

struct ErrorStats {
    double mean_abs = 0.0;
    double max_abs = 0.0;
    double max_rel = 0.0;
};

ErrorStats compute_stats(const std::vector<double>& ref, const std::vector<float>& obs) {
    double sum = 0.0;
    double max_abs = 0.0;
    double max_rel = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double err = std::abs(static_cast<double>(obs[i]) - ref[i]);
        sum += err;
        max_abs = std::max(max_abs, err);
        double denom = std::max(std::abs(ref[i]), 1e-9);
        max_rel = std::max(max_rel, err / denom);
    }
    return ErrorStats{sum / static_cast<double>(ref.size()), max_abs, max_rel};
}

void write_csv(const std::filesystem::path& path, const std::vector<std::vector<std::string>>& rows) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::trunc);
    for (const auto& row : rows) {
        for (size_t i = 0; i < row.size(); ++i) {
            out << row[i];
            if (i + 1 != row.size()) out << ',';
        }
        out << '\n';
    }
}

std::string format_decimal(double value, int precision) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return oss.str();
}

constexpr int kDoublePrecision = std::numeric_limits<double>::max_digits10;
constexpr int kFloatPrecision = std::numeric_limits<float>::max_digits10;

template <typename T>
void write_complex_txt(const std::filesystem::path& path, const std::vector<std::complex<T>>& values, int precision) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::trunc);
    out << "index real imag\n";
    out << std::scientific << std::setprecision(precision);
    for (size_t i = 0; i < values.size(); ++i) {
        out << i << ' ' << values[i].real() << ' ' << values[i].imag() << '\n';
    }
}

void dump_coefficients(const std::filesystem::path& dir, int digits, const std::vector<double>& reference,
                       const std::vector<std::complex<float>>& lut_vals,
                       const std::vector<std::complex<float>>& fast_vals, const std::vector<long long>& exact,
                       size_t valid_len) {
    std::vector<std::vector<std::string>> rows;
    rows.push_back({"index", "reference", "lut_real", "fast_real", "exact_int"});
    for (size_t i = 0; i < valid_len; ++i) {
        rows.push_back({std::to_string(i), format_decimal(reference[i], kDoublePrecision),
                        format_decimal(static_cast<double>(lut_vals[i].real()), kFloatPrecision),
                        format_decimal(static_cast<double>(fast_vals[i].real()), kFloatPrecision),
                        std::to_string(exact[i])});
    }
    write_csv(dir / ("digits_" + std::to_string(digits) + "_coefficients.csv"), rows);
}

template <typename T>
void dump_stage_txt(const std::filesystem::path& dir, int digits, const std::string& mode,
                    const std::string& stage, const std::vector<std::complex<T>>& data, int precision) {
    std::string filename = "digits_" + std::to_string(digits) + "_" + mode + "_" + stage + ".txt";
    write_complex_txt(dir / filename, data, precision);
}

void run_single_case(int digits, const Options& opts, fft::TwiddleTable& table,
                     std::vector<std::vector<std::string>>& summary_rows) {
    std::mt19937 rng(opts.seed + digits);
    auto a = random_digits(digits, opts.base, rng);
    auto b = random_digits(digits, opts.base, rng);
    size_t conv_len = a.size() + b.size();
    size_t fft_size = next_pow2(conv_len);

    std::cout << "[Experiment1] digits=" << digits << " fft_size=" << fft_size << "\n";

    std::vector<std::complex<double>> double_fft_stage;
    std::vector<std::complex<double>> double_hadamard_stage;
    auto ref_full = cpu_double_convolution(a, b, fft_size, &double_fft_stage, &double_hadamard_stage);
    dump_stage_txt(opts.output_dir, digits, "double", "fft_once", double_fft_stage, kDoublePrecision);
    dump_stage_txt(opts.output_dir, digits, "double", "hadamard", double_hadamard_stage, kDoublePrecision);
    std::vector<double> reference(conv_len);
    std::copy(ref_full.begin(), ref_full.begin() + conv_len, reference.begin());
    auto exact = exact_convolution(a, b);
    auto fa = digits_to_complex<float>(a, fft_size);
    auto fb = digits_to_complex<float>(b, fft_size);

    table.generate(static_cast<int>(fft_size));
    auto make_stage_logger = [&](const std::string& mode) {
        return [&, mode](const std::string& stage, const std::vector<std::complex<float>>& values) {
            dump_stage_txt(opts.output_dir, digits, mode, stage, values, kFloatPrecision);
        };
    };
    auto lut_res = gpu_convolution(fa, fb, static_cast<int>(fft_size), fft::TwiddleSource::kLUT,
                                   table.device.as<float2>(), make_stage_logger("lut"));
    auto fast_res = gpu_convolution(fa, fb, static_cast<int>(fft_size), fft::TwiddleSource::kFastApprox, nullptr,
                                    make_stage_logger("fast"));

    std::vector<float> lut_real(conv_len);
    std::vector<float> fast_real(conv_len);
    for (size_t i = 0; i < conv_len; ++i) {
        lut_real[i] = lut_res[i].real();
        fast_real[i] = fast_res[i].real();
    }

    auto lut_stats = compute_stats(reference, lut_real);
    auto fast_stats = compute_stats(reference, fast_real);
    summary_rows.push_back(
        {std::to_string(digits), "lut", format_decimal(lut_stats.mean_abs, kDoublePrecision),
         format_decimal(lut_stats.max_abs, kDoublePrecision), format_decimal(lut_stats.max_rel, kDoublePrecision)});
    summary_rows.push_back(
        {std::to_string(digits), "fast", format_decimal(fast_stats.mean_abs, kDoublePrecision),
         format_decimal(fast_stats.max_abs, kDoublePrecision), format_decimal(fast_stats.max_rel, kDoublePrecision)});

    dump_coefficients(opts.output_dir, digits, reference, lut_res, fast_res, exact, conv_len);

    std::cout << "  lut  mean_abs=" << lut_stats.mean_abs << " max_abs=" << lut_stats.max_abs
              << " max_rel=" << lut_stats.max_rel << "\n";
    std::cout << "  fast mean_abs=" << fast_stats.mean_abs << " max_abs=" << fast_stats.max_abs
              << " max_rel=" << fast_stats.max_rel << "\n";
}

void run_experiment1(const Options& opts) {
    std::filesystem::create_directories(opts.output_dir);
    fft::TwiddleTable table;
    std::vector<std::vector<std::string>> summary;
    summary.push_back({"digits", "mode", "mean_abs_error", "max_abs_error", "max_rel_error"});
    for (int digits : opts.digit_lengths) {
        run_single_case(digits, opts, table, summary);
    }
    write_csv(std::filesystem::path(opts.output_dir) / "summary.csv", summary);
}

Options parse_args(int argc, char** argv) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--digits" && i + 1 < argc) {
            opts.digit_lengths.clear();
            std::string list = argv[++i];
            std::stringstream ss(list);
            std::string item;
            while (std::getline(ss, item, ',')) {
                if (!item.empty()) opts.digit_lengths.push_back(std::stoi(item));
            }
            if (opts.digit_lengths.empty()) throw std::runtime_error("No digits specified");
        } else if (arg == "--base" && i + 1 < argc) {
            opts.base = std::stoi(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            opts.output_dir = argv[++i];
        } else if (arg == "--seed" && i + 1 < argc) {
            opts.seed = static_cast<unsigned>(std::stoul(argv[++i]));
        } else if (arg == "--help") {
            std::cout << "Usage: experiment1 [--digits 2048,8192] [--base 10] [--output data] [--seed 1337]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }
    return opts;
}

int main(int argc, char** argv) {
    try {
        Options opts = parse_args(argc, argv);
        run_experiment1(opts);
    } catch (const std::exception& ex) {
        std::cerr << "Fatal error: " << ex.what() << "\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
