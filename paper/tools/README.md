# Re-measurement tools and data

Tools and measurement outputs for the paper *"An Exhaustive 32-bit Evaluation
of Single-Precision Sine and Cosine on Consumer GPUs and Its Impact on
FFT-Based Computations"* (see the arXiv listing of the author for the paper
itself).

## Tools

| File | Purpose |
|---|---|
| `sweep_ulp.cu` | CUDA: exhaustive 2^32 sweep of `__sincosf` / `sinf` / `cosf` against in-kernel double references (spacing-weighted ulp statistics, per-exponent bins). Build: `nvcc -O3 -std=c++17 -arch=native sweep_ulp.cu -o sweep_ulp` and run with an output directory argument. |
| `sweep_ocl.py` | OpenCL (pyopencl): the same exhaustive sweep plus staircase / subnormal / power-of-two probes, for any OpenCL device. `--device` and `--platform` select the device by substring; on Intel pick the native runtime, e.g. `--device 770 --platform Graphics` (NOT the OpenCLOn12 platform that exposes the same GPU). Add `--full` for the 2^32 sweep. |
| `cliff_probe.py` | Locates the Intel accuracy cliff (round(x/pi) = 2^15) to a single float32 and measures post-cliff mean absolute error by range. |
| `period_probe.py` | Tests the 2^16 scale periodicity of Intel `native_sin` (exact for power-of-two arguments only). |
| `figs_paper.py` | Regenerates every figure in the paper (Okabe-Ito colorblind-safe palette, explicit units). Expects this repository's `experiment1/data` and a sibling clone of [FFTLUTtest](https://github.com/toropippi/FFTLUTtest) with its `output/` populated; both paths can be overridden with `--exp1-data` / `--fft2d-output`. |
| `crtrig_wrapper.c` | Batch wrapper exposing CORE-MATH `cr_sinf`/`cr_cosf` as a DLL. Build with the two [CORE-MATH](https://core-math.gitlabpages.inria.fr/) sources `src/binary32/sin/sinf.c` and `src/binary32/cos/cosf.c` (MIT license): `gcc -O3 -march=x86-64-v2 -shared -static -o crtrig.dll crtrig_wrapper.c sinf.c cosf.c` (GCC/Clang required). |
| `oracle_scan.py` | Exhaustive 2^32 comparison of the rounded-to-float double-libm reference against CORE-MATH; emits `oracle_mismatch_{sin,cos}.csv`. |
| `sweep_cr.py` | Exhaustive 2^32 sweep against the CORE-MATH oracle (`--dll` points at `crtrig.dll`). `--backend cuda` evaluates `__sincosf`/`sinf`/`cosf` via CuPy on NVIDIA; `--backend ocl` evaluates `native_sin`/`native_cos`/`sin`/`cos` via pyopencl. Reports spacing-weighted ulp statistics (denominator = spacing at the CORE-MATH value) and integer ulp distances to the correctly rounded result. |
| `subnormal_compile_probe.cu` | nvcc-compiled probe showing CUDA `sinf` preserves subnormal inputs bit-for-bit offline while the NVRTC route flushes them (paper Section 4.4). |
| `followup.cu` | Follow-up FFT experiments (paper Section 5): `followup timing` (LUT vs FAST forward-FFT wall time), `followup conv [digits,...]` (multiprecision convolution driven to digit failure), `followup iter2d [n] [iters]` (iterated 2D round trips). Build: `nvcc -O3 -std=c++17 -arch=native followup.cu -o followup`. |

## Measurement outputs

| Files | Contents |
|---|---|
| `stats.csv`, `bins_pi.csv`, `bins_exp.csv` | NVIDIA RTX 5090 (CUDA) sweep statistics |
| `ocl_stats_{amd,intel}.csv`, `ocl_bins_{pi,exp}_{amd,intel}.csv` | AMD gfx1036 / Intel UHD 770 (OpenCL) sweep statistics |
| `staircase_*_{nv,amd,intel}.csv` | Fast-path outputs near x = 0 and x = pi/2 |
| `pow2_{nv,amd,intel}.csv` | Fast sine at power-of-two arguments |
| `subnormal_{nv,amd,intel}.txt` | Subnormal-input handling of the fast path |
| `cliff_fine_intel.csv`, `cliff_ranges_intel.csv` | Intel cliff boundary and post-cliff error ranges |
| `period_probe_intel.txt` | 2^16 periodicity test results |
| `crsweep_stats_{nvidia,amd,intel}_cr.csv`, `crsweep_bins_pi_*_cr.csv` | Exhaustive sweep statistics against the CORE-MATH oracle (`sweep_cr.py`), including max/mean integer ulp distance to the correctly rounded result |
| `oracle_mismatch_{sin,cos}.csv` | The only inputs (2 for sin, 4 for cos, out of 2^32) where the rounded double-libm reference differs from CORE-MATH — all by one ulp, all far outside [-pi, pi] |
| `fft_timing_median.csv` | LUT vs FAST forward-FFT wall time (RTX 5090, medians of five runs) |
| `conv_failure_sizes.csv` | Multiprecision-convolution max coefficient error up to 2^20 digits; FAST first exceeds the 0.5 digit-corruption threshold at 32768 digits, LUT at 131072 |
| `iter2d_roundtrips.csv` | Error growth over 1-100 iterated 2D FFT round trips (256^2 edge image), LUT vs FAST |

Column conventions follow the paper: ulp errors use Eq. (1) of the paper
(numerator against the double-precision value, denominator = spacing at the
CORE-MATH correctly rounded reference), `wmean_ulp_pi` is the
spacing-weighted (uniform-x) mean over [-pi, pi].  The earlier
`stats.csv` / `ocl_stats_*.csv` files used the rounded double reference
throughout; `oracle_mismatch_*.csv` documents the exhaustive check that the
two references agree except at six one-ulp hard-to-round cases.
