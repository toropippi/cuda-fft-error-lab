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
#include <random>
#include <stdexcept>
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
            std::cerr << "CUDA error: " << cudaGetErrorString(err__) << " at "           \
                      << __FILE__ << ":" << __LINE__ << "\n";                            \
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

enum class SignalType { kRandom, kSine, kSineSum };

struct Options {
    std::vector<int> lengths{1024, 65536};
    std::vector<SignalType> signals{SignalType::kRandom, SignalType::kSine, SignalType::kSineSum};
    std::string output_dir = "data";
    unsigned seed = 2024;
};

constexpr double kMinRelDenom = 1e-12;

bool is_power_of_two(int n) { return n > 0 && (n & (n - 1)) == 0; }

std::string signal_to_string(SignalType type) {
    switch (type) {
        case SignalType::kRandom: return "random";
        case SignalType::kSine: return "sine";
        case SignalType::kSineSum: return "sine_sum";
    }
    return "unknown";
}

SignalType parse_signal(const std::string& name) {
    if (name == "random") return SignalType::kRandom;
    if (name == "sine") return SignalType::kSine;
    if (name == "sine_sum" || name == "dual" || name == "sum") return SignalType::kSineSum;
    throw std::runtime_error("Unknown signal type: " + name);
}

std::vector<double> generate_signal(SignalType type, int length, std::mt19937& rng) {
    std::vector<double> signal(length, 0.0);
    switch (type) {
        case SignalType::kRandom: {
            std::normal_distribution<double> dist(0.0, 1.0);
            for (double& v : signal) v = dist(rng);
            break;
        }
        case SignalType::kSine: {
            int freq = std::max(1, length / 32);
            double omega = 2.0 * M_PI * static_cast<double>(freq) / static_cast<double>(length);
            for (int n = 0; n < length; ++n) signal[n] = std::sin(omega * static_cast<double>(n));
            break;
        }
        case SignalType::kSineSum: {
            int f1 = std::max(1, length / 21);
            int f2 = std::max(1, length / 9);
            double omega1 = 2.0 * M_PI * static_cast<double>(f1) / static_cast<double>(length);
            double omega2 = 2.0 * M_PI * static_cast<double>(f2) / static_cast<double>(length);
            for (int n = 0; n < length; ++n) {
                double t = static_cast<double>(n);
                signal[n] = 0.6 * std::sin(omega1 * t) + 0.4 * std::sin(omega2 * t + 0.5);
            }
            break;
        }
    }
    return signal;
}

std::vector<std::complex<double>> real_to_complex(const std::vector<double>& real) {
    std::vector<std::complex<double>> out(real.size());
    for (size_t i = 0; i < real.size(); ++i) out[i] = std::complex<double>(real[i], 0.0);
    return out;
}

std::vector<std::complex<float>> real_to_complex_float(const std::vector<double>& real) {
    std::vector<std::complex<float>> out(real.size());
    for (size_t i = 0; i < real.size(); ++i) out[i] = std::complex<float>(static_cast<float>(real[i]), 0.0f);
    return out;
}

std::vector<std::complex<float>> gpu_fft_once(const std::vector<std::complex<float>>& input, bool inverse,
                                              fft::TwiddleSource src, const float2* twiddles) {
    DeviceBuffer d_data;
    fft::upload_complex(input, d_data);
    int fft_size = static_cast<int>(input.size());
    fft::launch_fft(d_data.as<float2>(), fft_size, inverse, src, twiddles);
    std::vector<std::complex<float>> host(input.size());
    fft::download_complex(host, d_data);
    return host;
}

struct ErrorAccumulator {
    double sum_abs = 0.0;
    double sum_rel = 0.0;
    double max_abs = 0.0;
    double max_rel = 0.0;
};

struct ModeSummary {
    double mean_abs = 0.0;
    double max_abs = 0.0;
    double mean_rel = 0.0;
    double max_rel = 0.0;
    double recon_l2 = 0.0;
};

void update_acc(ErrorAccumulator& acc, double abs_err, double rel_err) {
    acc.sum_abs += abs_err;
    acc.sum_rel += rel_err;
    acc.max_abs = std::max(acc.max_abs, abs_err);
    acc.max_rel = std::max(acc.max_rel, rel_err);
}

ModeSummary finalize_acc(const ErrorAccumulator& acc, size_t n) {
    ModeSummary summary;
    summary.mean_abs = acc.sum_abs / static_cast<double>(n);
    summary.mean_rel = acc.sum_rel / static_cast<double>(n);
    summary.max_abs = acc.max_abs;
    summary.max_rel = acc.max_rel;
    return summary;
}

void write_spectrum_csv(const std::filesystem::path& dir, int length, const std::string& signal_name,
                        const std::vector<std::complex<double>>& reference,
                        const std::vector<std::complex<float>>& lut_vals,
                        const std::vector<std::complex<float>>& fast_vals,
                        ModeSummary& lut_summary, ModeSummary& fast_summary) {
    std::filesystem::create_directories(dir);
    auto path = dir / ("signal_" + signal_name + "_n" + std::to_string(length) + "_spectrum.csv");
    std::ofstream out(path, std::ios::trunc);
    out << "index,ref_real,ref_imag,lut_real,lut_imag,lut_abs_err,lut_rel_err,fast_real,fast_imag,fast_abs_err,fast_rel_err\n";
    out << std::scientific << std::setprecision(std::numeric_limits<double>::max_digits10);
    ErrorAccumulator lut_acc;
    ErrorAccumulator fast_acc;
    for (size_t i = 0; i < reference.size(); ++i) {
        std::complex<double> ref(reference[i].real(), reference[i].imag());
        std::complex<double> lut(lut_vals[i].real(), lut_vals[i].imag());
        std::complex<double> fast(fast_vals[i].real(), fast_vals[i].imag());
        double ref_mag = std::abs(ref);
        double lut_abs = std::abs(lut - ref);
        double fast_abs = std::abs(fast - ref);
        double lut_rel = lut_abs / std::max(ref_mag, kMinRelDenom);
        double fast_rel = fast_abs / std::max(ref_mag, kMinRelDenom);
        update_acc(lut_acc, lut_abs, lut_rel);
        update_acc(fast_acc, fast_abs, fast_rel);
        out << std::to_string(i) << ',' << ref.real() << ',' << ref.imag() << ',' << static_cast<double>(lut.real()) << ','
            << static_cast<double>(lut.imag()) << ',' << lut_abs << ',' << lut_rel << ',' << static_cast<double>(fast.real())
            << ',' << static_cast<double>(fast.imag()) << ',' << fast_abs << ',' << fast_rel << '\n';
    }
    lut_summary = finalize_acc(lut_acc, reference.size());
    fast_summary = finalize_acc(fast_acc, reference.size());
}

double reconstruction_error(const std::vector<std::complex<float>>& recon, const std::vector<double>& original) {
    double sum = 0.0;
    for (size_t i = 0; i < original.size(); ++i) {
        std::complex<double> ideal(original[i], 0.0);
        std::complex<double> got(recon[i].real(), recon[i].imag());
        double diff = std::abs(got - ideal);
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

struct SummaryRow {
    int length = 0;
    std::string signal;
    std::string mode;
    ModeSummary summary;
};

void write_summary_csv(const std::filesystem::path& dir, const std::vector<SummaryRow>& rows) {
    std::filesystem::create_directories(dir);
    std::ofstream out(dir / "experiment2_summary.csv", std::ios::trunc);
    out << "length,signal,mode,mean_abs_error,max_abs_error,mean_rel_error,max_rel_error,reconstruction_l2\n";
    out << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (const auto& row : rows) {
        out << std::defaultfloat << row.length << ',' << row.signal << ',' << row.mode << ',';
        out << std::scientific << row.summary.mean_abs << ',' << row.summary.max_abs << ',' << row.summary.mean_rel << ','
            << row.summary.max_rel << ',' << row.summary.recon_l2 << '\n';
    }
}

void run_single_case(int length, SignalType signal_type, const Options& opts, fft::TwiddleTable& table,
                     std::vector<SummaryRow>& summary_rows) {
    if (!is_power_of_two(length)) {
        throw std::runtime_error("Length must be a power of two: " + std::to_string(length));
    }
    std::mt19937 rng(opts.seed + static_cast<unsigned>(length) * 37u + static_cast<unsigned>(signal_type) * 997u);
    auto signal = generate_signal(signal_type, length, rng);
    auto signal_complex = real_to_complex(signal);
    auto reference = fft::cpu_fft(signal_complex, false);

    auto input_float = real_to_complex_float(signal);
    table.generate(length);
    auto lut_freq = gpu_fft_once(input_float, false, fft::TwiddleSource::kLUT, table.device.as<float2>());
    auto fast_freq = gpu_fft_once(input_float, false, fft::TwiddleSource::kFastApprox, nullptr);

    ModeSummary lut_summary;
    ModeSummary fast_summary;
    std::string sig_name = signal_to_string(signal_type);
    write_spectrum_csv(opts.output_dir, length, sig_name, reference, lut_freq, fast_freq, lut_summary, fast_summary);

    auto lut_time = gpu_fft_once(lut_freq, true, fft::TwiddleSource::kLUT, table.device.as<float2>());
    auto fast_time = gpu_fft_once(fast_freq, true, fft::TwiddleSource::kFastApprox, nullptr);
    lut_summary.recon_l2 = reconstruction_error(lut_time, signal);
    fast_summary.recon_l2 = reconstruction_error(fast_time, signal);

    summary_rows.push_back({length, sig_name, "lut", lut_summary});
    summary_rows.push_back({length, sig_name, "fast", fast_summary});

    std::cout << "[Experiment2] length=" << length << " signal=" << sig_name << "\n";
    std::cout << "  lut  mean_abs=" << lut_summary.mean_abs << " max_abs=" << lut_summary.max_abs
              << " mean_rel=" << lut_summary.mean_rel << " max_rel=" << lut_summary.max_rel
              << " recon_l2=" << lut_summary.recon_l2 << "\n";
    std::cout << "  fast mean_abs=" << fast_summary.mean_abs << " max_abs=" << fast_summary.max_abs
              << " mean_rel=" << fast_summary.mean_rel << " max_rel=" << fast_summary.max_rel
              << " recon_l2=" << fast_summary.recon_l2 << "\n";
}

void run_experiment2(const Options& opts) {
    std::filesystem::create_directories(opts.output_dir);
    fft::TwiddleTable table;
    std::vector<SummaryRow> summary_rows;
    for (int length : opts.lengths) {
        for (SignalType signal : opts.signals) {
            run_single_case(length, signal, opts, table, summary_rows);
        }
    }
    write_summary_csv(opts.output_dir, summary_rows);
}

Options parse_args(int argc, char** argv) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--lengths" && i + 1 < argc) {
            opts.lengths.clear();
            std::stringstream ss(argv[++i]);
            std::string item;
            while (std::getline(ss, item, ',')) {
                if (!item.empty()) opts.lengths.push_back(std::stoi(item));
            }
            if (opts.lengths.empty()) throw std::runtime_error("No lengths specified");
        } else if (arg == "--signals" && i + 1 < argc) {
            opts.signals.clear();
            std::stringstream ss(argv[++i]);
            std::string item;
            while (std::getline(ss, item, ',')) {
                if (!item.empty()) opts.signals.push_back(parse_signal(item));
            }
            if (opts.signals.empty()) throw std::runtime_error("No signals specified");
        } else if (arg == "--output" && i + 1 < argc) {
            opts.output_dir = argv[++i];
        } else if (arg == "--seed" && i + 1 < argc) {
            opts.seed = static_cast<unsigned>(std::stoul(argv[++i]));
        } else if (arg == "--help") {
            std::cout << "Usage: experiment2 [--lengths 1024,65536] [--signals random,sine,sine_sum] [--output data] [--seed 2024]\n";
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
        run_experiment2(opts);
    } catch (const std::exception& ex) {
        std::cerr << "Fatal error: " << ex.what() << "\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
