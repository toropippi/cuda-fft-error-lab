import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SPECTRUM_PATTERN = re.compile(r"signal_(?P<signal>.+)_n(?P<length>\d+)_spectrum\.csv")


def parse_args():
    parser = argparse.ArgumentParser(description="plotting helper")
    parser.add_argument("--data", default="../data", help="Directory containing spectrum CSVs")
    parser.add_argument("--figures", default="../figures", help="Directory to write PNG figures")
    parser.add_argument("--bins", type=int, default=120, help="Histogram bins for abs-error plots")
    return parser.parse_args()


def find_spectrum_files(data_dir: Path):
    files = []
    for path in data_dir.glob("signal_*_n*_spectrum.csv"):
        match = SPECTRUM_PATTERN.fullmatch(path.name)
        if not match:
            continue
        meta = match.groupdict()
        files.append((path, meta["signal"], int(meta["length"])))
    files.sort(key=lambda item: (item[1], item[2]))
    return files


def plot_error_histograms(errors, bins, output):
    modes = ["lut", "fast"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, mode, color in zip(axes, modes, ["tab:blue", "tab:orange"]):
        data = errors[mode]
        ax.hist(data, bins=bins, color=color, alpha=0.85, log=True)
        ax.set_title(f"{mode.upper()} abs-error distribution")
        ax.set_xlabel("absolute error")
        ax.set_ylabel("count")
        ax.grid(True, which="both", linestyle=":", linewidth=0.6)
    fig.suptitle("absolute error histograms")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_frequency_error(df, signal, length, output):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["index"], df["lut_abs_err"], label="LUT abs error", linewidth=1.0)
    ax.plot(df["index"], df["fast_abs_err"], label="Fast abs error", linewidth=1.0)
    ax.set_title(f"Abs error vs frequency index ({signal}, n={length})")
    ax.set_xlabel("frequency index")
    ax.set_ylabel("absolute error")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_reconstruction(summary_df, output):
    tmp = summary_df.copy()
    tmp["recon_mse"] = (tmp["reconstruction_l2"] ** 2) / tmp["length"]
    agg = (
        tmp.groupby(["signal", "mode"])[["mean_abs_error", "mean_rel_error", "recon_mse"]]
        .mean()
        .reset_index()
    )

    signals = sorted(agg["signal"].unique())
    mode_order = ["lut", "fast"]
    modes = [m for m in mode_order if m in set(agg["mode"])]
    x = np.arange(len(signals))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = {"lut": "tab:blue", "fast": "tab:orange"}
    for idx, mode in enumerate(modes):
        values = []
        for sig in signals:
            row = agg[(agg["signal"] == sig) & (agg["mode"] == mode)]
            values.append(row["recon_mse"].values[0] if not row.empty else np.nan)
        ax.bar(
            x + (idx - 0.5) * width,
            values,
            width=width,
            label=f"{mode.upper()} recon MSE",
            color=colors.get(mode, None),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(signals)
    ax.set_ylabel("reconstruction MSE")
    ax.set_yscale("log")
    ax.set_title("Reconstruction error per signal type (log scale)")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)

    return agg[["signal", "mode", "mean_abs_error", "mean_rel_error", "recon_mse"]]


def main():
    args = parse_args()
    data_dir = Path(args.data).resolve()
    figs_dir = Path(args.figures).resolve()
    figs_dir.mkdir(parents=True, exist_ok=True)

    spectrum_files = find_spectrum_files(data_dir)
    if not spectrum_files:
        raise FileNotFoundError(f"No spectrum CSV found in {data_dir}")

    errors = {"lut": [], "fast": []}
    for path, signal, length in spectrum_files:
        df = pd.read_csv(path)
        for col in ["index", "lut_abs_err", "fast_abs_err"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        errors["lut"].append(df["lut_abs_err"].to_numpy(dtype=np.float64))
        errors["fast"].append(df["fast_abs_err"].to_numpy(dtype=np.float64))
        freq_plot = figs_dir / f"exp2_abs_error_{signal}_n{length}.png"
        plot_frequency_error(df, signal, length, freq_plot)

    errors = {k: np.concatenate(v) for k, v in errors.items() if v}
    hist_path = figs_dir / "exp2_abs_error_hist.png"
    plot_error_histograms(errors, args.bins, hist_path)

    summary_path = data_dir / "experiment2_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"{summary_path} not found")
    summary_df = pd.read_csv(summary_path)
    summary_df["length"] = pd.to_numeric(summary_df["length"], errors="coerce")
    recon_plot = figs_dir / "exp2_reconstruction_mse.png"
    agg = plot_reconstruction(summary_df, recon_plot)

    analysis_csv = data_dir / "analysis_experiment2_summary.csv"
    agg.to_csv(analysis_csv, index=False)

    print("=== Experiment 2 aggregate stats ===")
    print(agg.to_string(index=False))
    print(f"Saved histogram to {hist_path}")
    print(f"Saved per-signal plots ({len(spectrum_files)}) into {figs_dir}")
    print(f"Saved reconstruction comparison to {recon_plot}")
    print(f"Aggregate CSV: {analysis_csv}")


if __name__ == "__main__":
    main()
