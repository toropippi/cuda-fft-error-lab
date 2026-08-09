"""Regenerate all paper figures with a colorblind-safe (Okabe-Ito) palette,
explicit units, and colorbars.

Inputs:
  - staircase_*.csv, bins_*.csv, ocl_bins_*_amd.csv   (sweep tools, this dir)
  - cuda-fft-error-lab experiment1 coefficient CSVs
  - FFTLUTtest output/ (summary.csv and per-case .npy)
"""
import argparse
import json
import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Okabe-Ito colorblind-safe palette
C_LUT = "#0072B2"    # blue
C_FAST = "#D55E00"   # vermillion
C_STD = "#009E73"    # bluish green
C_INTEL = "#CC79A7"  # reddish purple
C_REF = "#000000"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.size": 10,
    "axes.grid": True, "grid.alpha": 0.3,
})

# Default data locations, relative to this file:
#   EXP1  -> <this repo>/experiment1/data
#   FFT2D -> a sibling clone of https://github.com/toropippi/FFTLUTtest, after
#            running `fftlut_experiment --run-all --output-dir output`
_HERE = os.path.dirname(os.path.abspath(__file__))
EXP1 = os.path.normpath(os.path.join(_HERE, "..", "..", "experiment1", "data"))
FFT2D = os.path.normpath(os.path.join(_HERE, "..", "..", "..",
                                      "FFTLUTtest", "output"))


def fig_staircase(out):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    for ax, fname, col, ref_col, title, xl, x0 in [
        (axes[0], "staircase_near0", "native_sin", "std_sin",
         "fast sine near $x=0$", "x (radians)", 0.0),
        (axes[1], "staircase_nearhalfpi", "native_cos", "std_cos",
         r"fast cosine near $x=\pi/2$", r"$x - \pi/2$ (radians)",
         np.pi / 2),
    ]:
        nv = pd.read_csv(f"{fname}_nv.csv")
        amd = pd.read_csv(f"{fname}_amd.csv")
        itl = pd.read_csv(f"{fname}_intel.csv")
        xref = nv["x"].to_numpy(dtype=np.float64)
        if "near0" in fname:
            ref = np.sin(xref)
        else:
            ref = np.cos(xref)
        xs = {k: v["x"].to_numpy(dtype=np.float64) - x0
              for k, v in [("nv", nv), ("amd", amd), ("itl", itl)]}
        ax.plot(xs["nv"], ref, color=C_REF, lw=1.0, label="exact")
        ax.plot(xs["nv"], nv[col], color=C_LUT, lw=1.1, ls="-",
                label="NVIDIA RTX 5090")
        ax.plot(xs["amd"], amd[col], color=C_FAST, lw=1.1, ls=(0, (3, 1)),
                label="AMD gfx1036")
        ax.plot(xs["itl"], itl[col], color=C_INTEL, lw=1.3, ls=(0, (1, 0.8)),
                label="Intel UHD 770")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xl)
        ax.set_ylabel("function value (dimensionless)")
        ax.ticklabel_format(style="sci", scilimits=(-2, 3))
    axes[0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_intel_cliff(out):
    import re as _re
    xs, ns, ref = [], [], []
    for line in open("cliff_fine_intel.csv").read().splitlines()[1:]:
        p = line.split(",")
        xs.append(float(_re.sub(r"np\.float32\(|\)", "", p[0])))
        ns.append(float(_re.sub(r"np\.float32\(|\)", "", p[2])))
        ref.append(float(_re.sub(r"np\.float64\(|\)", "", p[3])))
    xs, ns, ref = np.asarray(xs), np.asarray(ns), np.asarray(ref)
    m = xs <= 102950.0
    cliff = 102942.14
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.plot(xs[m], ref[m], color=C_REF, lw=1.2, label="exact sin x")
    ax.plot(xs[m], ns[m], color=C_INTEL, lw=1.1, ls="-",
            label="Intel fast (native_sin)")
    ax.axvline(cliff, color="gray", ls="-.", lw=1.0)
    ax.annotate(r"first float with round$(x/\pi) = 2^{15}$"
                "\n(x = 102942.14)",
                xy=(cliff, 0.62), xytext=(102940.05, 0.92), fontsize=8,
                color="dimgray",
                arrowprops=dict(arrowstyle="->", color="dimgray", lw=0.8))
    ax.set_xlabel("x (radians)")
    ax.set_ylabel("function value (dimensionless)")
    ax.set_ylim(-1.3, 1.25)
    ax.legend(loc="lower center", fontsize=8, ncol=2)
    ax.ticklabel_format(style="plain", useOffset=False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_exponent(out):
    nv = pd.read_csv("bins_exp.csv")           # CUDA sweep (RTX 5090)
    amd = pd.read_csv("ocl_bins_exp_amd.csv")  # OpenCL sweep (gfx1036)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    m = (nv["log2_x"] >= -30) & (nv["log2_x"] <= 127)
    ax.semilogy(nv["log2_x"][m], np.maximum(nv["native_sin"][m], 1e-12),
                lw=1.5, ls="-", color=C_LUT, label="NVIDIA fast (__sinf)")
    ma = (amd["log2_x"] >= -30) & (amd["log2_x"] <= 127)
    ax.semilogy(amd["log2_x"][ma], np.maximum(amd["native_sin"][ma], 1e-12),
                lw=1.5, ls=(0, (4, 1.5)), color=C_FAST,
                label="AMD fast (native_sin)")
    itl = pd.read_csv("ocl_bins_exp_intel.csv")
    mi = (itl["log2_x"] >= -30) & (itl["log2_x"] <= 127)
    ax.semilogy(itl["log2_x"][mi], np.maximum(itl["native_sin"][mi], 1e-12),
                lw=1.6, ls=(0, (3, 1, 1, 1)), color=C_INTEL,
                label="Intel fast (native_sin)")
    ax.semilogy(nv["log2_x"][m], np.maximum(nv["std_sin"][m], 1e-12),
                lw=1.6, ls=(0, (1, 1)), color=C_STD,
                label="NVIDIA standard (sinf)")
    ax.axvline(np.log2(2 * np.pi), color="gray", ls="--", lw=0.8)
    ax.text(np.log2(2 * np.pi) + 1, 1e-11, r"$2\pi$", color="gray")
    ax.axvline(np.log2(2**15 * np.pi), color="gray", ls="-.", lw=0.8)
    ax.text(np.log2(2**15 * np.pi) - 5.6, 1e-11, r"$2^{15}\pi$", color="gray")
    ax.axvline(np.log2(2**23 * np.pi), color="gray", ls=":", lw=0.8)
    ax.text(np.log2(2**23 * np.pi) + 1, 1e-11, r"$2^{23}\pi$", color="gray")
    ax.set_xlabel(r"$\log_2 |x|$  (per-binary-exponent bins over all finite floats)")
    ax.set_ylabel("mean absolute error (dimensionless)")
    ax.legend(loc="center right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_ulp_profile(out):
    nv = pd.read_csv("bins_pi.csv")
    amd = pd.read_csv("ocl_bins_pi_amd.csv")
    itl = pd.read_csv("ocl_bins_pi_intel.csv")
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.8), sharex=True)
    for ax, col, title in [
        (axes[0], "native_sin", "fast sine (CUDA __sinf / OpenCL native_sin)"),
        (axes[1], "native_cos", "fast cosine (CUDA __cosf / OpenCL native_cos)"),
    ]:
        ax.semilogy(nv["x_center"], np.maximum(nv[col], 1e-2), lw=1.0,
                    color=C_LUT, label="NVIDIA RTX 5090")
        ax.semilogy(amd["x_center"], np.maximum(amd[col], 1e-2), lw=1.0,
                    ls=(0, (3, 1)), color=C_FAST, label="AMD gfx1036")
        ax.semilogy(itl["x_center"], np.maximum(itl[col], 1e-2), lw=1.2,
                    ls=(0, (1, 0.8)), color=C_INTEL, label="Intel UHD 770")
        ax.set_ylabel("mean error (ulp)")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(1e-2, 1e5)
    axes[0].legend(loc="upper right", fontsize=9)
    axes[1].set_xlabel("x (radians)")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def load_conv(digits):
    df = pd.read_csv(f"{EXP1}/digits_{digits}_coefficients.csv")
    err_lut = np.abs(df["lut_real"] - df["reference"])
    err_fast = np.abs(df["fast_real"] - df["reference"])
    return df["index"], err_lut, err_fast




def fig_conv_hist(out):
    _, el, ef = load_conv(8192)
    hi = max(ef.max(), el.max())
    bins = np.linspace(0, hi * 1.04, 40)
    fig, (axl, axr) = plt.subplots(
        1, 2, figsize=(7.2, 3.8), sharey=True,
        gridspec_kw={"width_ratios": [5, 1], "wspace": 0.06})
    for ax in (axl, axr):
        ax.hist(el, bins=bins, color=C_LUT, alpha=0.85, label="LUT twiddles",
                edgecolor="white", linewidth=0.3)
        ax.hist(ef, bins=bins, color=C_FAST, alpha=0.6, label="FAST twiddles",
                hatch="///", edgecolor="white", linewidth=0.3)
    axl.set_xlim(0, hi * 1.08)
    axr.set_xlim(0.468, 0.532)
    axr.axvline(0.5, color=C_REF, ls="--", lw=1.2)
    axr.text(0.497, axl.get_ylim()[1] * 0.55, "digit-corruption\nthreshold (0.5)",
             fontsize=8, va="center", ha="right", rotation=90)
    axr.set_xticks([0.5])
    # broken-axis marks
    axl.spines["right"].set_visible(False)
    axr.spines["left"].set_visible(False)
    axr.tick_params(left=False)
    d = 0.013
    kw = dict(transform=axl.transAxes, color="k", clip_on=False, lw=0.8)
    axl.plot((1 - d / 5, 1 + d / 5), (-d, +d), **kw)
    kw = dict(transform=axr.transAxes, color="k", clip_on=False, lw=0.8)
    axr.plot((-d, +d), (-d, +d), **kw)
    axl.set_xlabel("absolute coefficient error (digit units)")
    axl.xaxis.set_label_coords(0.6, -0.12)
    axl.set_ylabel("count")
    axl.legend(fontsize=9)
    axr.grid(False)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)



CASES_2D = [
    ("horizontal gradient", "case_horizontal_gradient_0_256x256"),
    ("sharp edge vertical", "case_sharp_edge_vertical_0_256x256"),
    ("text small", "case_text_small_0_256x256"),
    ("game-like scene", "case_game_like_scene_simple_0_256x256"),
]


def fig_2d_diff(out):
    from PIL import Image
    fig, axes = plt.subplots(2, len(CASES_2D), figsize=(11.4, 5.6),
                             constrained_layout=True)
    norm = LogNorm(vmin=1e-10, vmax=1e-6)
    im = None
    for col, (name, case) in enumerate(CASES_2D):
        orig = Image.open(f"{FFT2D}/{case}/original.png")
        diff = np.load(f"{FFT2D}/{case}/absdiff_fast_vs_lut.npy")
        axes[0][col].imshow(orig, cmap="gray", vmin=0, vmax=255)
        axes[0][col].set_title(name, fontsize=10)
        im = axes[1][col].imshow(np.maximum(diff, 1e-12), cmap="viridis",
                                 norm=norm)
        for row in range(2):
            axes[row][col].set_xticks([])
            axes[row][col].set_yticks([])
            axes[row][col].grid(False)
    axes[0][0].set_ylabel("original image\n(intensity in [0,1])", fontsize=9)
    axes[1][0].set_ylabel("|FAST $-$ LUT|", fontsize=9)
    cbar = fig.colorbar(im, ax=axes[1, :], shrink=0.92, pad=0.015)
    cbar.set_label("absolute difference\n(intensity units, log)", fontsize=9)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main():
    global EXP1, FFT2D
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../figures")
    ap.add_argument("--exp1-data", default=EXP1,
                    help="cuda-fft-error-lab experiment1/data directory")
    ap.add_argument("--fft2d-output", default=FFT2D,
                    help="FFTLUTtest output directory (run --run-all first)")
    ap.add_argument("--skip-sweep-figs", action="store_true",
                    help="skip figures that need the AMD full-sweep CSVs")
    a = ap.parse_args()
    EXP1, FFT2D = a.exp1_data, a.fft2d_output
    fig_staircase(f"{a.out}/fig_staircase.png")
    fig_intel_cliff(f"{a.out}/fig_intel_cliff.png")
    fig_conv_hist(f"{a.out}/fig_conv_hist.png")
    fig_2d_diff(f"{a.out}/fig_2dfft_orig_vs_diff.png")
    if not a.skip_sweep_figs:
        fig_exponent(f"{a.out}/fig_abs_error_exponent.png")
        fig_ulp_profile(f"{a.out}/fig_ulp_profile_pi.png")
    print("figures written")


if __name__ == "__main__":
    main()
