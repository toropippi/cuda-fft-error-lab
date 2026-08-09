"""Probe the Intel fixed-point cliff around x where round(x/pi) = 2^15.

Evaluates native_sin at consecutive float32 values around x = 102942..102944
and along a coarse grid up to 2^20 to locate where the error explodes.
Also samples mean |error| beyond the cliff (decorrelation check).

Usage: python cliff_probe.py --device 770 --tag intel
"""
import argparse
import numpy as np
import pyopencl as cl

KERNEL = r"""
__kernel void eval_pts(__global const float* xs, __global float* out_ns,
                       __global float* out_nc) {
    const uint gid = (uint)get_global_id(0);
    const float x = xs[gid];
    out_ns[gid] = native_sin(x);
    out_nc[gid] = native_cos(x);
}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--platform", default="", help="substring filter on platform name")
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
    prg = cl.Program(ctx, KERNEL).build(options=[])
    mf = cl.mem_flags
    kern = cl.Kernel(prg, "eval_pts")

    def eval_points(xs):
        xs = np.ascontiguousarray(xs, dtype=np.float32)
        n = len(xs)
        bx = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=xs)
        outs = [cl.Buffer(ctx, mf.WRITE_ONLY, n * 4) for _ in range(2)]
        kern(q, (n,), None, bx, *outs)
        res = []
        for b in outs:
            h = np.empty(n, np.float32)
            cl.enqueue_copy(q, h, b)
            res.append(h)
        q.finish()
        return res

    # consecutive float32 values spanning the predicted cliff
    lo = np.float32(102940.0)
    xs = [lo]
    for _ in range(5000):
        xs.append(np.nextafter(xs[-1], np.float32(np.inf)))
    xs = np.array(xs, dtype=np.float32)
    ns, nc = eval_points(xs)
    ref = np.sin(xs.astype(np.float64))
    err = np.abs(ns.astype(np.float64) - ref)
    with open(f"cliff_fine_{a.tag}.csv", "w") as f:
        f.write("x,x_over_pi,native_sin,ref,abs_err\n")
        for x, v, r, e in zip(xs, ns, ref, err):
            f.write(f"{x!r},{np.float64(x)/np.pi:.6f},{v!r},{r!r},{e:.6e}\n")
    bad = err > 1e-2
    if bad.any():
        i = int(np.argmax(bad))
        print(f"first bad x = {xs[i]!r} (x/pi = {np.float64(xs[i])/np.pi:.4f}), "
              f"err = {err[i]:.3e}")
        if i > 0:
            print(f"last good x = {xs[i-1]!r} (x/pi = {np.float64(xs[i-1])/np.pi:.4f}), "
                  f"err = {err[i-1]:.3e}")
    else:
        print("no cliff found in fine window", float(xs[0]), float(xs[-1]))

    # mean |error| before/after the cliff (decorrelation check)
    rng_bins = [(1e3, 1e4), (1e4, 1e5), (1.03e5, 2e5), (2e5, 1e6), (1e6, 1e7)]
    with open(f"cliff_ranges_{a.tag}.csv", "w") as f:
        f.write("range_lo,range_hi,mean_abs_err_native_sin,max_abs_err\n")
        for lo_, hi_ in rng_bins:
            xs = np.linspace(lo_, hi_, 200001).astype(np.float32)
            ns, nc = eval_points(xs)
            ref = np.sin(xs.astype(np.float64))
            e = np.abs(ns.astype(np.float64) - ref)
            f.write(f"{lo_:g},{hi_:g},{e.mean():.6e},{e.max():.6e}\n")
            print(f"[{lo_:g},{hi_:g}] mean={e.mean():.4e} max={e.max():.4e}")


if __name__ == "__main__":
    main()
