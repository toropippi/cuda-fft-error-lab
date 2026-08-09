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

Column conventions follow the paper: ulp errors use Eq. (1) of the paper
(spacing at the rounded double-precision reference), `wmean_ulp_pi` is the
spacing-weighted (uniform-x) mean over [-pi, pi].
