"""Test whether Intel native_sin is exactly periodic in the binary exponent:
does native_sin(2^16 * x) == native_sin(x) hold in general (not just x=2^k)?

Usage: python period_probe.py --device 770 --platform Graphics --tag intel
"""
import argparse
import numpy as np
import pyopencl as cl

KERNEL = r"""
__kernel void eval_pts(__global const float* xs, __global float* out_ns) {
    const uint gid = (uint)get_global_id(0);
    out_ns[gid] = native_sin(xs[gid]);
}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--platform", default="")
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
        ob = cl.Buffer(ctx, mf.WRITE_ONLY, n * 4)
        kern(q, (n,), None, bx, ob)
        h = np.empty(n, np.float32)
        cl.enqueue_copy(q, h, ob)
        q.finish()
        return h

    rng = np.random.default_rng(12345)
    with open(f"period_probe_{a.tag}.txt", "w") as f:
        for lo, hi, note in [
            (3.0, 100.0, "pre-cliff base"),
            (2.0**17, 2.0**20, "post-cliff base"),
            (2.0**21, 2.0**22, "post-cliff base 2"),
        ]:
            x = rng.uniform(lo, hi, 20000).astype(np.float32)
            x16 = (x.astype(np.float64) * 65536.0).astype(np.float32)
            # keep only pairs where the *32 multiplication was exact
            exact = (x16.astype(np.float64) == x.astype(np.float64) * 65536.0)
            x, x16 = x[exact], x16[exact]
            y, y16 = eval_points(x), eval_points(x16)
            eq = np.mean(y == y16)
            d = np.abs(y.astype(np.float64) - y16.astype(np.float64))
            line = (f"{note} [{lo:g},{hi:g}]: n={len(x)}, "
                    f"bitwise-equal fraction = {eq:.6f}, max |diff| = {d.max():.3e}")
            print(line)
            f.write(line + "\n")


if __name__ == "__main__":
    main()
