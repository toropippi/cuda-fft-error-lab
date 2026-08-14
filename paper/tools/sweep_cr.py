"""Exhaustive 2^32 sweep with CORE-MATH correctly rounded oracle.

Backends:
  --backend cuda : cupy RawKernel evaluating __sincosf / sinf / cosf  (NVIDIA)
  --backend ocl  : pyopencl evaluating native_sin/native_cos/sin/cos (AMD/Intel)

References:
  - correctly rounded float reference: CORE-MATH cr_sinf/cr_cosf via crtrig.dll
  - high-precision numerator: numpy double sin/cos (|err| <= ~2^-29 float ulp)

Outputs ocl_stats-compatible CSV plus:
  - max/mean integer ulp distance of each path to the CORE-MATH reference
  - count of inputs where float32(double ref) != CORE-MATH (oracle cross-check)
"""
import argparse, ctypes, os, time
import numpy as np
from concurrent.futures import ThreadPoolExecutor

CHUNK = 1 << 24
PI = np.pi
K_PI_BINS = 2048

HERE = os.path.dirname(os.path.abspath(__file__))


def load_dll(path):
    dll = ctypes.CDLL(path)
    for fn in ("cr_sinf_range", "cr_cosf_range"):
        getattr(dll, fn).argtypes = [ctypes.c_uint32,
                                     np.ctypeslib.ndpointer(np.uint32),
                                     ctypes.c_int64]
        getattr(dll, fn).restype = None
    return dll


def cr_range(dll, fn, start, n, workers=8):
    out = np.empty(n, dtype=np.uint32)
    f = getattr(dll, fn)
    step = n // workers
    def part(i):
        s = i * step
        e = n if i == workers - 1 else s + step
        f(np.uint32((start + s) & 0xFFFFFFFF), out[s:e], e - s)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(part, range(workers)))
    return out


def ulp32_of_f32(v32):
    """float64 spacing of the float32 grid at float32 value v32 (array)."""
    a = np.abs(v32)
    return (np.nextafter(a, np.float32(np.inf)) - a).astype(np.float64)


def ordkey(bits):
    """Monotone int64 key for float32 bit patterns (total order, +-0 equal)."""
    b = bits.astype(np.int64)
    mag = b & 0x7FFFFFFF
    return np.where(b & 0x80000000 != 0, -mag, mag)


CUDA_SRC = r"""
extern "C" __global__ void sweep(float* ons, float* onc, float* os, float* oc,
                                 unsigned int hi) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int u = (hi << 24) | gid;
    float x = __uint_as_float(u);
    float s, c;
    __sincosf(x, &s, &c);
    ons[gid] = s; onc[gid] = c;
    os[gid] = sinf(x); oc[gid] = cosf(x);
}
"""

OCL_SRC = r"""
__kernel void sweep(__global float* ons, __global float* onc,
                    __global float* os,  __global float* oc,
                    const uint chunk_hi) {
    const uint gid = (uint)get_global_id(0);
    const uint u = (chunk_hi << 24) | gid;
    const float x = as_float(u);
    ons[gid] = native_sin(x);
    onc[gid] = native_cos(x);
    os[gid]  = sin(x);
    oc[gid]  = cos(x);
}
"""


class CudaBackend:
    def __init__(self):
        import cupy as cp
        self.cp = cp
        self.kern = cp.RawKernel(CUDA_SRC, "sweep")
        self.bufs = [cp.empty(CHUNK, cp.float32) for _ in range(4)]
        self.name = "CUDA:" + cp.cuda.runtime.getDeviceProperties(0)["name"].decode()

    def chunk(self, hi):
        cp = self.cp
        self.kern((CHUNK // 256,), (256,), (*self.bufs, np.uint32(hi)))
        return [cp.asnumpy(b) for b in self.bufs]


class OclBackend:
    def __init__(self, device, platform):
        import pyopencl as cl
        self.cl = cl
        dev = None
        for p in cl.get_platforms():
            if platform.lower() not in p.name.lower():
                continue
            for d in p.get_devices():
                if device.lower() in d.name.lower() and dev is None:
                    dev, plat = d, p
        if dev is None:
            raise SystemExit(f"device {device} not found")
        self.name = plat.name + "::" + dev.name
        self.ctx = cl.Context([dev])
        self.q = cl.CommandQueue(self.ctx)
        self.prg = cl.Program(self.ctx, OCL_SRC).build(options=[])
        mf = cl.mem_flags
        self.bufs = [cl.Buffer(self.ctx, mf.WRITE_ONLY, CHUNK * 4) for _ in range(4)]
        self.hosts = [np.empty(CHUNK, np.float32) for _ in range(4)]

    def chunk(self, hi):
        cl = self.cl
        self.prg.sweep(self.q, (CHUNK,), None, *self.bufs, np.uint32(hi))
        for h, b in zip(self.hosts, self.bufs):
            cl.enqueue_copy(self.q, h, b)
        self.q.finish()
        return [h.copy() for h in self.hosts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["cuda", "ocl"], required=True)
    ap.add_argument("--device", default="")
    ap.add_argument("--platform", default="")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dll", default=os.path.join(HERE, "crtrig.dll"))
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()

    dll = load_dll(a.dll)
    be = CudaBackend() if a.backend == "cuda" else OclBackend(a.device, a.platform)
    print("using:", be.name, flush=True)

    names = (["fast_sin", "fast_cos", "std_sin", "std_cos"] if a.backend == "cuda"
             else ["native_sin", "native_cos", "std_sin", "std_cos"])

    sum_ulp_pi = np.zeros(4); w_pi = 0.0
    max_ulp_pi = np.zeros(4)
    max_abs_2pi = np.zeros(4)
    max_abs_100pi = np.zeros(4)
    sum_abs_100pi = np.zeros(4); n_100pi = 0
    bin_pi_sum = np.zeros((4, K_PI_BINS)); bin_pi_w = np.zeros(K_PI_BINS)
    max_dist = np.zeros(4, dtype=np.int64)
    max_dist_normal = np.zeros(4, dtype=np.int64)
    argmax_dist_normal = np.zeros(4, dtype=np.uint32)
    sum_dist_norm = np.zeros(4, dtype=np.float64); n_norm = 0
    argmax_dist = np.zeros(4, dtype=np.uint32)
    oracle_mism = [0, 0]  # sin, cos: float32(double) != cr

    t0 = time.time()
    for hi in range(256):
        outs = be.chunk(hi)
        u = (np.uint32(hi) << np.uint32(24)) | np.arange(CHUNK, dtype=np.uint32)
        x = u.view(np.float32)
        finite = np.isfinite(x)

        crs = cr_range(dll, "cr_sinf_range", hi << 24, CHUNK, a.threads)
        crc = cr_range(dll, "cr_cosf_range", hi << 24, CHUNK, a.threads)
        crs_f = crs.view(np.float32); crc_f = crc.view(np.float32)

        xd = x[finite].astype(np.float64)
        ref_s = np.sin(xd); ref_c = np.cos(xd)
        oracle_mism[0] += int((ref_s.astype(np.float32).view(np.uint32)
                               != crs[finite]).sum())
        oracle_mism[1] += int((ref_c.astype(np.float32).view(np.uint32)
                               != crc[finite]).sum())

        refs = [ref_s, ref_c, ref_s, ref_c]
        crefs = [crs_f, crc_f, crs_f, crc_f]
        vals = [o[finite].astype(np.float64) for o in outs]
        abserr = [np.abs(v - r) for v, r in zip(vals, refs)]
        ax = np.abs(xd)

        # integer ulp distance to CORE-MATH reference
        exp = (u >> np.uint32(23)) & np.uint32(0xFF)
        normal = (exp >= 1) & (exp <= 254)
        n_norm += int(normal.sum())
        for k in range(4):
            key_o = ordkey(outs[k].view(np.uint32)[finite])
            key_c = ordkey(crefs[k].view(np.uint32)[finite])
            dist = np.abs(key_o - key_c)
            i = int(np.argmax(dist))
            if dist[i] > max_dist[k]:
                max_dist[k] = dist[i]
                argmax_dist[k] = u[finite][i]
            dn = np.abs(ordkey(outs[k].view(np.uint32)[normal])
                        - ordkey(crefs[k].view(np.uint32)[normal]))
            sum_dist_norm[k] += float(dn.sum())
            j = int(np.argmax(dn))
            if dn[j] > max_dist_normal[k]:
                max_dist_normal[k] = dn[j]
                argmax_dist_normal[k] = u[normal][j]

        m100 = ax <= 100 * PI
        if m100.any():
            n_100pi += int(m100.sum())
            for k in range(4):
                e = abserr[k][m100]
                sum_abs_100pi[k] += e.sum()
                max_abs_100pi[k] = max(max_abs_100pi[k], e.max())
        m2 = ax <= 2 * PI
        if m2.any():
            for k in range(4):
                max_abs_2pi[k] = max(max_abs_2pi[k], abserr[k][m2].max())

        mpi = ax <= PI
        if mpi.any():
            w = ulp32_of_f32(xd[mpi].astype(np.float32))
            w_pi += w.sum()
            b = np.clip(((xd[mpi] + PI) / (2 * PI) * K_PI_BINS).astype(np.int64),
                        0, K_PI_BINS - 1)
            np.add.at(bin_pi_w, b, w)
            for k in range(4):
                ue = abserr[k][mpi] / ulp32_of_f32(crefs[k][finite][mpi])
                sum_ulp_pi[k] += (ue * w).sum()
                max_ulp_pi[k] = max(max_ulp_pi[k], ue.max())
                np.add.at(bin_pi_sum[k], b, ue * w)
        if hi % 32 == 31:
            print(f"chunk {hi+1}/256  ({time.time()-t0:.0f}s)", flush=True)

    with open(os.path.join(HERE, f"crsweep_stats_{a.tag}.csv"), "w") as f:
        f.write("function,wmean_ulp_pi,max_ulp_pi,mean_abs_100pi,max_abs_100pi,"
                "max_abs_2pi,max_dist_cr,argmax_dist_bits,"
                "max_dist_cr_normal,argmax_dist_normal_bits,mean_dist_cr_normal\n")
        for k in range(4):
            f.write(f"{names[k]},{sum_ulp_pi[k]/w_pi:.6f},{max_ulp_pi[k]:.1f},"
                    f"{sum_abs_100pi[k]/n_100pi:.9e},{max_abs_100pi[k]:.9e},"
                    f"{max_abs_2pi[k]:.9e},{max_dist[k]},"
                    f"{int(argmax_dist[k]):08x},"
                    f"{max_dist_normal[k]},{int(argmax_dist_normal[k]):08x},"
                    f"{sum_dist_norm[k]/n_norm:.6f}\n")
    with open(os.path.join(HERE, f"crsweep_bins_pi_{a.tag}.csv"), "w") as f:
        f.write("x_center,weight," + ",".join(names) + "\n")
        for b in range(K_PI_BINS):
            xc = -PI + (b + 0.5) * 2 * PI / K_PI_BINS
            row = [f"{bin_pi_sum[k][b]/bin_pi_w[b]:.9e}" if bin_pi_w[b] > 0 else "0"
                   for k in range(4)]
            f.write(f"{xc:.9f},{bin_pi_w[b]:.9e}," + ",".join(row) + "\n")
    print(f"oracle cross-check: double-vs-CORE-MATH mismatches "
          f"sin={oracle_mism[0]} cos={oracle_mism[1]}", flush=True)
    print("SWEEP_DONE", time.time() - t0, "s", flush=True)


if __name__ == "__main__":
    main()
