"""Exhaustive oracle comparison: float32(double-libm sin/cos) vs CORE-MATH
cr_sinf/cr_cosf over all 2^32 bit patterns.  Lists every finite input where
the double-rounded reference differs from the correctly rounded result."""
import ctypes, os, sys, time
import numpy as np
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
DLL = ctypes.CDLL(os.path.join(HERE, "crtrig.dll"))
for fn in ("cr_sinf_range", "cr_cosf_range"):
    getattr(DLL, fn).argtypes = [ctypes.c_uint32,
                                 np.ctypeslib.ndpointer(np.uint32),
                                 ctypes.c_int64]
    getattr(DLL, fn).restype = None

CHUNK = 1 << 24

def scan_chunk(args):
    h, func = args
    start = np.uint32(h << 24)
    bits = (np.uint64(h << 24) + np.arange(CHUNK, dtype=np.uint64)).astype(np.uint32)
    x = bits.view(np.float32)
    finite = np.isfinite(x)
    cr = np.empty(CHUNK, dtype=np.uint32)
    if func == "sin":
        DLL.cr_sinf_range(start, cr, CHUNK)
        ref = np.sin(x.astype(np.float64)).astype(np.float32)
    else:
        DLL.cr_cosf_range(start, cr, CHUNK)
        ref = np.cos(x.astype(np.float64)).astype(np.float32)
    neq = (cr != ref.view(np.uint32)) & finite
    idx = np.nonzero(neq)[0]
    return [(int(bits[i]), int(ref.view(np.uint32)[i]), int(cr[i])) for i in idx]

def main():
    for func in ("sin", "cos"):
        t0 = time.time()
        mism = []
        with ThreadPoolExecutor(max_workers=24) as ex:
            for r in ex.map(scan_chunk, [(h, func) for h in range(256)]):
                mism.extend(r)
        mism.sort()
        print(f"{func}: {len(mism)} mismatches, {time.time()-t0:.0f}s", flush=True)
        with open(os.path.join(HERE, f"mismatch_{func}.csv"), "w") as f:
            f.write("input_bits,double_ref_bits,cr_bits,input_val,double_ref,cr_val\n")
            for b, rd, rc in mism:
                xv = np.uint32(b).view(np.float32)
                f.write(f"{b:08x},{rd:08x},{rc:08x},{xv!r},"
                        f"{np.uint32(rd).view(np.float32)!r},"
                        f"{np.uint32(rc).view(np.float32)!r}\n")
    print("SCAN_DONE", flush=True)

if __name__ == "__main__":
    main()
