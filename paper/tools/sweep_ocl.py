"""Exhaustive 32-bit sweep of native_sin/native_cos/sin/cos via OpenCL.

Runs on any OpenCL device (used here for the AMD gfx1036 iGPU, optionally the
RTX 5090 for cross-checking the CUDA sweep).  References are computed on the
host in double precision with NumPy.  Outputs CSVs analogous to sweep_ulp.cu.

Usage:
  python sweep_ocl.py --device gfx1036 --tag amd            # probes only
  python sweep_ocl.py --device gfx1036 --tag amd --full     # + full 2^32 sweep
"""
import argparse
import numpy as np
import pyopencl as cl

KERNEL = r"""
__kernel void sweep(__global float* out_ns, __global float* out_nc,
                    __global float* out_s,  __global float* out_c,
                    const uint chunk_hi) {
    const uint gid = (uint)get_global_id(0);
    const uint u = (chunk_hi << 24) | gid;
    const float x = as_float(u);
    out_ns[gid] = native_sin(x);
    out_nc[gid] = native_cos(x);
    out_s[gid]  = sin(x);
    out_c[gid]  = cos(x);
}
__kernel void eval_pts(__global const float* xs, __global float* out_ns,
                       __global float* out_nc, __global float* out_s,
                       __global float* out_c) {
    const uint gid = (uint)get_global_id(0);
    const float x = xs[gid];
    out_ns[gid] = native_sin(x);
    out_nc[gid] = native_cos(x);
    out_s[gid]  = sin(x);
    out_c[gid]  = cos(x);
}
"""

CHUNK = 1 << 24
PI = np.pi
K_PI_BINS = 2048
K_EXP_BINS = 277  # binary exponents -149..127


def ulp32(v64):
    """Spacing of the float32 grid at the float32 rounding of v64."""
    v32 = np.abs(v64.astype(np.float32))
    return (np.nextafter(v32, np.float32(np.inf)) - v32).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--platform", default="", help="substring filter on platform name")
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()

    dev = None
    for p in cl.get_platforms():
        if a.platform.lower() not in p.name.lower():
            continue
        for d in p.get_devices():
            if a.device.lower() in d.name.lower() and dev is None:
                dev = d
                plat = p
    if dev is None:
        raise SystemExit(f"device {a.device} not found")
    print("using:", plat.name, "::", dev.name)
    ctx = cl.Context([dev])
    q = cl.CommandQueue(ctx)
    prg = cl.Program(ctx, KERNEL).build(options=[])  # no fast-math options

    mf = cl.mem_flags

    def eval_points(xs):
        xs = np.ascontiguousarray(xs, dtype=np.float32)
        n = len(xs)
        bx = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=xs)
        outs = [cl.Buffer(ctx, mf.WRITE_ONLY, n * 4) for _ in range(4)]
        prg.eval_pts(q, (n,), None, bx, *outs)
        res = []
        for b in outs:
            h = np.empty(n, np.float32)
            cl.enqueue_copy(q, h, b)
            res.append(h)
        q.finish()
        return res  # ns, nc, s, c

    # ---- probe 1: staircase samples near 0 and near pi/2 ----
    xs0 = np.linspace(0.0, 2e-6, 4001).astype(np.float32)
    ns, nc, s, c = eval_points(xs0)
    np.savetxt(f"staircase_near0_{a.tag}.csv",
               np.column_stack([xs0, ns, s]), delimiter=",",
               header="x,native_sin,std_sin", comments="")

    xph = (np.float64(np.pi) / 2 + np.linspace(-2e-6, 2e-6, 4001)).astype(np.float32)
    ns, nc, s, c = eval_points(xph)
    np.savetxt(f"staircase_nearhalfpi_{a.tag}.csv",
               np.column_stack([xph, nc, c]), delimiter=",",
               header="x,native_cos,std_cos", comments="")

    # ---- probe 2: subnormal handling of native_sin ----
    sub_bits = np.arange(1, 1 << 20, 7, dtype=np.uint32)  # sample of subnormals
    xs = sub_bits.view(np.float32)
    ns, nc, s, c = eval_points(xs)
    ref = np.sin(xs.astype(np.float64))
    dist_ulp = np.abs(ns.astype(np.float64) - ref) / ulp32(ref)
    flushed = np.mean(ns == 0.0)
    print(f"subnormals: native_sin flushed-to-zero fraction = {flushed:.4f}, "
          f"max ulp dist = {dist_ulp.max():.2f}")
    with open(f"subnormal_{a.tag}.txt", "w") as f:
        f.write(f"flushed_fraction,{flushed}\nmax_ulp,{dist_ulp.max()}\n")

    # ---- probe 3: large-argument behavior along 2^k ----
    ks = np.arange(1, 127)
    xs = (2.0 ** ks).astype(np.float32)
    ns, nc, s, c = eval_points(xs)
    ref = np.sin(xs.astype(np.float64))
    with open(f"pow2_{a.tag}.csv", "w") as f:
        f.write("log2x,native_sin,ref\n")
        for k, v, r in zip(ks, ns, ref):
            f.write(f"{k},{v!r},{r!r}\n")

    if not a.full:
        return

    # ---- full 2^32 sweep ----
    sum_ulp_pi = np.zeros(4)
    w_pi = 0.0
    max_ulp_pi = np.zeros(4)
    max_abs_2pi = np.zeros(4)
    max_abs_100pi = np.zeros(4)
    sum_abs_100pi = np.zeros(4)
    n_100pi = 0
    bin_pi_sum = np.zeros((4, K_PI_BINS))
    bin_pi_w = np.zeros(K_PI_BINS)
    bin_exp_sum = np.zeros((4, K_EXP_BINS))
    bin_exp_cnt = np.zeros(K_EXP_BINS, dtype=np.int64)

    outs = [cl.Buffer(ctx, mf.WRITE_ONLY, CHUNK * 4) for _ in range(4)]
    hosts = [np.empty(CHUNK, np.float32) for _ in range(4)]

    for hi in range(256):
        prg.sweep(q, (CHUNK,), None, *outs, np.uint32(hi))
        for h, b in zip(hosts, outs):
            cl.enqueue_copy(q, h, b)
        q.finish()
        u = (np.uint32(hi) << np.uint32(24)) | np.arange(CHUNK, dtype=np.uint32)
        x = u.view(np.float32)
        finite = np.isfinite(x)
        xd = x[finite].astype(np.float64)
        vals = [h[finite].astype(np.float64) for h in hosts]  # ns, nc, s, c
        ref_s = np.sin(xd)
        ref_c = np.cos(xd)
        refs = [ref_s, ref_c, ref_s, ref_c]
        abserr = [np.abs(v - r) for v, r in zip(vals, refs)]
        ax = np.abs(xd)

        # exponent bins
        eb = np.clip(np.floor(np.log2(np.maximum(ax, 1e-300))), -149, 127).astype(np.int64) + 149
        eb[ax == 0.0] = 0
        for k in range(4):
            np.add.at(bin_exp_sum[k], eb, abserr[k])
        np.add.at(bin_exp_cnt, eb, 1)

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
            w = ulp32(xd[mpi])  # spacing of x -> uniform-x weighting
            w_pi += w.sum()
            b = np.clip(((xd[mpi] + PI) / (2 * PI) * K_PI_BINS).astype(np.int64),
                        0, K_PI_BINS - 1)
            np.add.at(bin_pi_w, b, w)
            for k in range(4):
                ue = abserr[k][mpi] / ulp32(refs[k][mpi])
                sum_ulp_pi[k] += (ue * w).sum()
                max_ulp_pi[k] = max(max_ulp_pi[k], ue.max())
                np.add.at(bin_pi_sum[k], b, ue * w)
        if hi % 32 == 31:
            print(f"chunk {hi+1}/256")

    names = ["native_sin", "native_cos", "std_sin", "std_cos"]
    with open(f"ocl_stats_{a.tag}.csv", "w") as f:
        f.write("function,wmean_ulp_pi,max_ulp_pi,mean_abs_100pi,max_abs_100pi,max_abs_2pi\n")
        for k in range(4):
            f.write(f"{names[k]},{sum_ulp_pi[k]/w_pi:.6f},{max_ulp_pi[k]:.1f},"
                    f"{sum_abs_100pi[k]/n_100pi:.9e},{max_abs_100pi[k]:.9e},"
                    f"{max_abs_2pi[k]:.9e}\n")
    with open(f"ocl_bins_pi_{a.tag}.csv", "w") as f:
        f.write("x_center,weight," + ",".join(names) + "\n")
        for b in range(K_PI_BINS):
            xc = -PI + (b + 0.5) * 2 * PI / K_PI_BINS
            row = [f"{bin_pi_sum[k][b]/bin_pi_w[b]:.9e}" if bin_pi_w[b] > 0 else "0"
                   for k in range(4)]
            f.write(f"{xc:.9f},{bin_pi_w[b]:.9e}," + ",".join(row) + "\n")
    with open(f"ocl_bins_exp_{a.tag}.csv", "w") as f:
        f.write("log2_x,count," + ",".join(names) + "\n")
        for b in range(K_EXP_BINS):
            row = [f"{bin_exp_sum[k][b]/bin_exp_cnt[b]:.9e}" if bin_exp_cnt[b] else "0"
                   for k in range(4)]
            f.write(f"{b-149},{bin_exp_cnt[b]}," + ",".join(row) + "\n")
    print("full sweep done")


if __name__ == "__main__":
    main()
