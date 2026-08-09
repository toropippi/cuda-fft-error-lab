// Exhaustive 32-bit sweep of single-precision sin/cos error on NVIDIA GPUs.
// Enumerates all 2^32 float bit patterns, compares __sinf/__cosf (SFU fast path)
// and sinf/cosf (standard path) against double-precision references.
//
// Outputs (CSV):
//   stats.csv        : mean/max ULP and abs error over |x|<=pi and |x|<=100pi
//   bins_pi.csv      : 2048 linear bins over [-pi, pi], mean ULP per bin
//   bins_exp.csv     : per-binary-exponent bins (log2|x| in [-149, 127]), mean abs error
//
// Build: nvcc -O3 -std=c++17 -arch=native sweep_ulp.cu -o sweep_ulp
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

#define CUDA_CHECK(expr)                                                         \
    do {                                                                         \
        cudaError_t err__ = (expr);                                              \
        if (err__ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                          \
                    cudaGetErrorString(err__), __FILE__, __LINE__);              \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

constexpr int kPiBins = 2048;      // linear bins over [-pi, pi]
constexpr int kExpBins = 277;      // binary exponents -149..127
constexpr double kPi = 3.14159265358979323846;

struct Accum {
    // [0]=native_sin, [1]=native_cos, [2]=sin, [3]=cos
    double sum_ulp_pi[4];      // weighted by spacing(x): uniform-x average
    double weight_pi;
    double max_ulp_pi[4];
    unsigned long long count_pi;
    double max_abs_100pi[4];
    double sum_abs_100pi[4];
    unsigned long long count_100pi;
    double max_abs_2pi[4];
    // binned data (weighted by spacing(x) within each bin)
    double bin_pi_sum[4][kPiBins];
    double bin_pi_w[kPiBins];
    double bin_exp_sum[4][kExpBins];   // mean abs error per exponent bin
    unsigned long long bin_exp_cnt[kExpBins];
};

__device__ __forceinline__ double ulp_of(float ref) {
    float a = fabsf(ref);
    float next = nextafterf(a, INFINITY);
    return (double)next - (double)a;   // spacing at ref (min: 2^-149 at 0)
}

__device__ __forceinline__ double ulp_err(float res, double ref_d) {
    float ref_f = (float)ref_d;        // round-to-nearest float reference
    double u = ulp_of(ref_f);
    return fabs((double)res - ref_d) / u;
}

__global__ void sweep_kernel(unsigned chunk_hi, Accum* acc) {
    const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned u = (chunk_hi << 24) | gid;
    const float x = __uint_as_float(u);
    if (!isfinite(x)) return;

    const double xd = (double)x;
    const double ref_s = sin(xd);
    const double ref_c = cos(xd);

    float fast_s, fast_c;
    __sincosf(x, &fast_s, &fast_c);    // SFU fast path (sin.approx / cos.approx)
    const float std_s = sinf(x);       // standard CUDA single-precision path
    const float std_c = cosf(x);

    const double abs_err[4] = {
        fabs((double)fast_s - ref_s), fabs((double)fast_c - ref_c),
        fabs((double)std_s  - ref_s), fabs((double)std_c  - ref_c)
    };

    const double ax = fabs(xd);

    // per-exponent bin (mean abs error over the whole float range)
    int e;
    frexpf(fabsf(x), &e);              // |x| = m * 2^e, m in [0.5,1)
    int eb = e - 1 + 149;              // log2 exponent, shifted to [0, 276]
    if (x == 0.0f) eb = 0;
    if (eb >= 0 && eb < kExpBins) {
        for (int k = 0; k < 4; ++k)
            atomicAdd(&acc->bin_exp_sum[k][eb], abs_err[k]);
        atomicAdd(&acc->bin_exp_cnt[eb], 1ULL);
    }

    if (ax <= 100.0 * kPi) {
        for (int k = 0; k < 4; ++k) {
            atomicAdd(&acc->sum_abs_100pi[k], abs_err[k]);
            // atomicMax on double via CAS
            double* addr = &acc->max_abs_100pi[k];
            double old = *addr;
            while (old < abs_err[k]) {
                unsigned long long assumed = __double_as_longlong(old);
                unsigned long long prev = atomicCAS((unsigned long long*)addr, assumed,
                                                    __double_as_longlong(abs_err[k]));
                if (prev == assumed) break;
                old = __longlong_as_double(prev);
            }
        }
        atomicAdd(&acc->count_100pi, 1ULL);
    }

    if (ax <= 2.0 * kPi) {
        for (int k = 0; k < 4; ++k) {
            double* addr = &acc->max_abs_2pi[k];
            double old = *addr;
            while (old < abs_err[k]) {
                unsigned long long assumed = __double_as_longlong(old);
                unsigned long long prev = atomicCAS((unsigned long long*)addr, assumed,
                                                    __double_as_longlong(abs_err[k]));
                if (prev == assumed) break;
                old = __longlong_as_double(prev);
            }
        }
    }

    if (ax <= kPi) {
        const double ue[4] = {
            ulp_err(fast_s, ref_s), ulp_err(fast_c, ref_c),
            ulp_err(std_s, ref_s),  ulp_err(std_c, ref_c)
        };
        // weight = spacing of the float grid at x -> uniform-x expectation
        const double w = ulp_of(x);
        for (int k = 0; k < 4; ++k) {
            atomicAdd(&acc->sum_ulp_pi[k], ue[k] * w);
            double* addr = &acc->max_ulp_pi[k];
            double old = *addr;
            while (old < ue[k]) {
                unsigned long long assumed = __double_as_longlong(old);
                unsigned long long prev = atomicCAS((unsigned long long*)addr, assumed,
                                                    __double_as_longlong(ue[k]));
                if (prev == assumed) break;
                old = __longlong_as_double(prev);
            }
        }
        atomicAdd(&acc->weight_pi, w);
        atomicAdd(&acc->count_pi, 1ULL);

        int b = (int)((xd + kPi) / (2.0 * kPi) * kPiBins);
        if (b < 0) b = 0;
        if (b >= kPiBins) b = kPiBins - 1;
        for (int k = 0; k < 4; ++k)
            atomicAdd(&acc->bin_pi_sum[k][b], ue[k] * w);
        atomicAdd(&acc->bin_pi_w[b], w);
    }
}

int main(int argc, char** argv) {
    const char* outdir = (argc > 1) ? argv[1] : ".";
    Accum* d_acc;
    CUDA_CHECK(cudaMalloc(&d_acc, sizeof(Accum)));
    CUDA_CHECK(cudaMemset(d_acc, 0, sizeof(Accum)));

    const int threads = 256;
    const int blocks = (1 << 24) / threads;
    for (unsigned hi = 0; hi < 256; ++hi) {
        sweep_kernel<<<blocks, threads>>>(hi, d_acc);
        CUDA_CHECK(cudaGetLastError());
        if (hi % 32 == 31) {
            CUDA_CHECK(cudaDeviceSynchronize());
            printf("chunk %u/256 done\n", hi + 1);
            fflush(stdout);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    Accum* h = (Accum*)malloc(sizeof(Accum));
    CUDA_CHECK(cudaMemcpy(h, d_acc, sizeof(Accum), cudaMemcpyDeviceToHost));

    char path[1024];
    const char* names[4] = {"native_sin", "native_cos", "std_sin", "std_cos"};

    snprintf(path, sizeof(path), "%s/stats.csv", outdir);
    FILE* f = fopen(path, "w");
    fprintf(f, "function,wmean_ulp_pi,max_ulp_pi,count_pi,mean_abs_100pi,max_abs_100pi,max_abs_2pi,count_100pi\n");
    for (int k = 0; k < 4; ++k) {
        fprintf(f, "%s,%.6f,%.1f,%llu,%.9e,%.9e,%.9e,%llu\n", names[k],
                h->sum_ulp_pi[k] / h->weight_pi, h->max_ulp_pi[k], h->count_pi,
                h->sum_abs_100pi[k] / (double)h->count_100pi, h->max_abs_100pi[k],
                h->max_abs_2pi[k], h->count_100pi);
    }
    fclose(f);

    snprintf(path, sizeof(path), "%s/bins_pi.csv", outdir);
    f = fopen(path, "w");
    fprintf(f, "x_center,weight,native_sin,native_cos,std_sin,std_cos\n");
    for (int b = 0; b < kPiBins; ++b) {
        double xc = -kPi + (b + 0.5) * (2.0 * kPi / kPiBins);
        double w = h->bin_pi_w[b];
        fprintf(f, "%.9f,%.9e", xc, w);
        for (int k = 0; k < 4; ++k)
            fprintf(f, ",%.9e", w > 0.0 ? h->bin_pi_sum[k][b] / w : 0.0);
        fprintf(f, "\n");
    }
    fclose(f);

    snprintf(path, sizeof(path), "%s/bins_exp.csv", outdir);
    f = fopen(path, "w");
    fprintf(f, "log2_x,count,native_sin,native_cos,std_sin,std_cos\n");
    for (int b = 0; b < kExpBins; ++b) {
        int e = b - 149;
        unsigned long long c = h->bin_exp_cnt[b];
        fprintf(f, "%d,%llu", e, c);
        for (int k = 0; k < 4; ++k)
            fprintf(f, ",%.9e", c ? h->bin_exp_sum[k][b] / (double)c : 0.0);
        fprintf(f, "\n");
    }
    fclose(f);

    printf("done\n");
    free(h);
    cudaFree(d_acc);
    return 0;
}
