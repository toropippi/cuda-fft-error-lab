import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_coefficients(data_dir: Path) -> pd.DataFrame:
    rows = []
    for csv_path in sorted(data_dir.glob("digits_*_coefficients.csv")):
        digits = int(csv_path.stem.split("_")[1])
        df = pd.read_csv(csv_path)
        df["digits"] = digits
        df["index"] = df["index"].astype(int)
        rows.append(df)
    if not rows:
        raise FileNotFoundError(f"No coefficient CSV files found in {data_dir}")
    return pd.concat(rows, ignore_index=True)


def compute_mode_stats(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    stats = []
    for digits, group in df.groupby("digits"):
        values = group[f"{mode}_real"]
        exact = group["exact_int"]
        abs_err = (values - exact).abs()
        rel_mask = exact != 0
        rel_err = pd.Series(np.nan, index=group.index, dtype=float)
        rel_err.loc[rel_mask] = abs_err.loc[rel_mask] / exact.loc[rel_mask].abs()
        rounded = values.round().astype(np.int64)
        digit_error = (rounded - exact).abs()
        stats.append(
            {
                "digits": digits,
                "mode": mode,
                "count": len(group),
                "mean_abs_error": abs_err.mean(),
                "max_abs_error": abs_err.max(),
                "mean_rel_error": rel_err.mean(),
                "max_rel_error": rel_err.max(),
                "error_rate": (rounded != exact).mean(),
                "max_digit_error": digit_error.max(),
            }
        )
    return pd.DataFrame(stats)


def plot_grouped_bar(df: pd.DataFrame, value_col: str, title: str, ylabel: str, output: Path) -> None:
    digits = sorted(df["digits"].unique())
    modes = ["lut", "fast"]
    x = np.arange(len(digits))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, mode in enumerate(modes):
        subset = df[df["mode"] == mode]
        ax.bar(x + (i - 0.5) * width, subset[value_col], width, label=mode)
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in digits])
    ax.set_xlabel("digits")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_abs_error_hist(group: pd.DataFrame, digits: int, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    lut_err = (group["lut_real"] - group["exact_int"]).abs().to_numpy()
    fast_err = (group["fast_real"] - group["exact_int"]).abs().to_numpy()
    all_err = np.concatenate([lut_err, fast_err])
    max_err = all_err.max()
    if max_err == 0:
        max_display = 1e-6
    else:
        q99 = np.quantile(all_err, 0.995)
        max_display = max_err if max_err <= q99 * 1.5 else q99
    bins = np.linspace(0, max_display, 40 if max_display > 0 else 20)
    for mode in ["lut", "fast"]:
        abs_err = (group[f"{mode}_real"] - group["exact_int"]).abs()
        ax.hist(abs_err, bins=bins, alpha=0.6, label=f"{mode} abs err")
    if max_err > max_display:
        ax.axvline(max_err, color="k", linestyle="--", label=f"max {max_err:.3g}")
    ax.set_xlabel("absolute error")
    ax.set_ylabel("count")
    ax.set_title(f"Abs error histogram (digits={digits})")
    ax.set_xlim(0, bins[-1])
    xticks = np.linspace(0, bins[-1], num=6)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{tick:.3g}" for tick in xticks])
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_index_error(group: pd.DataFrame, digits: int, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    for mode in ["lut", "fast"]:
        abs_err = (group[f"{mode}_real"] - group["exact_int"]).abs()
        ax.plot(group["index"], abs_err, label=f"{mode}")
    ax.set_xlabel("coefficient index")
    ax.set_ylabel("absolute error")
    ax.set_title(f"Error vs index (digits={digits})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


BASE_DIR = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment1 FFT convolution outputs.")
    parser.add_argument("--data", type=Path, default=None, help="Path to data directory")
    parser.add_argument("--figures", type=Path, default=None, help="Directory to save figures")
    parser.add_argument("--summary", type=Path, default=None, help="CSV path for derived summary metrics")
    args = parser.parse_args()

    data_dir = args.data if args.data is not None else BASE_DIR / "data"
    figures_dir = args.figures if args.figures is not None else BASE_DIR / "figures"
    summary_path = args.summary if args.summary is not None else BASE_DIR / "data" / "analysis_summary.csv"

    coeff_df = load_coefficients(data_dir.resolve())
    stats_frames = []
    for mode in ["lut", "fast"]:
        stats_frames.append(compute_mode_stats(coeff_df, mode))
    summary_df = pd.concat(stats_frames, ignore_index=True)
    summary_df.to_csv(summary_path, index=False)

    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_grouped_bar(summary_df, "error_rate", "Rounding error rate by digits", "error rate",
                     figures_dir / "error_rate_bar.png")
    plot_grouped_bar(summary_df, "max_digit_error", "Maximum digit error by digits", "digits difference",
                     figures_dir / "max_digit_error_bar.png")

    for digits, group in coeff_df.groupby("digits"):
        plot_abs_error_hist(group, digits, figures_dir / f"abs_error_hist_{digits}.png")
        plot_index_error(group, digits, figures_dir / f"error_vs_index_{digits}.png")

    print("Saved summary to", summary_path)
    print("Figures written to", figures_dir)


if __name__ == "__main__":
    main()
