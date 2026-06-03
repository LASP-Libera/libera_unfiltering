"""
Quick inspection tool for a generated unfiltering coefficients .nc file.

Usage:
    python tools/inspect_coefficients.py coefficients/unfiltering_coefficients_v0.1.0_srf-0-0-1_modtran-3.7.nc

    # show a geometry plot of the linear (a1) coefficient
    python tools/inspect_coefficients.py <file.nc> --plot
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr


def _bin_label(lo, hi):
    return f"{lo:.1f}–{hi:.1f}°"


def inspect(path: str, plot: bool = False):
    ds = xr.open_dataset(path)

    # ── Metadata ────────────────────────────────────────────────────────────
    print("=" * 60)
    print("METADATA")
    print("=" * 60)
    for k, v in ds.attrs.items():
        print(f"  {k:<25} {v}")

    # ── Bin geometry ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("VIEWING GEOMETRY BINS")
    print("=" * 60)
    sza_lo, sza_hi = ds["sza_lo"].values, ds["sza_hi"].values
    vza_lo, vza_hi = ds["vza_lo"].values, ds["vza_hi"].values
    raz_lo, raz_hi = ds["raz_lo"].values, ds["raz_hi"].values
    print(f"  SZA bins ({len(sza_lo)}): " + "  ".join(_bin_label(lo, hi) for lo, hi in zip(sza_lo, sza_hi)))
    print(f"  VZA bins ({len(vza_lo)}): " + "  ".join(_bin_label(lo, hi) for lo, hi in zip(vza_lo, vza_hi)))
    print(f"  RAZ bins ({len(raz_lo)}): " + "  ".join(_bin_label(lo, hi) for lo, hi in zip(raz_lo, raz_hi)))

    # ── Coverage ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("BIN COVERAGE  (NaN = no data / sparse bin)")
    print("=" * 60)
    total_bins = len(sza_lo) * len(vza_lo) * len(raz_lo)

    for var, label in [("sw_coefficients", "SW"), ("lw_coefficients", "LW"), ("ssw_coefficients", "SSW")]:
        arr = ds[var].values
        # a bin is populated if its first coefficient is not NaN
        populated = ~np.isnan(arr[:, :, :, 0])
        n_filled = int(populated.sum())
        print(f"  {label}: {n_filled}/{total_bins} bins populated  ({100*n_filled/total_bins:.0f}%)")

    # ── Coefficient stats ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("COEFFICIENT STATS  (populated bins only)")
    print("=" * 60)
    for var, label, idx_names in [
        ("sw_coefficients",  "SW",  ["a0 (offset)", "a1 (linear)", "a2 (quadratic)"]),
        ("lw_coefficients",  "LW",  ["a0 (offset)", "a1 (linear)", "a2 (quadratic)"]),
        ("ssw_coefficients", "SSW", ["intercept", "c1 (const)", "c2 (ssw_f)", "c3 (sw_f)",
                                     "c4 (ssw_f²)", "c5 (cross)", "c6 (sw_f²)"]),
    ]:
        arr = ds[var].values
        print(f"\n  {label}:")
        for i, name in enumerate(idx_names):
            vals = arr[:, :, :, i].ravel()
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                print(f"    idx {i} {name:<20}  no data")
            else:
                print(f"    idx {i} {name:<20}  "
                      f"min={vals.min():+.6f}  mean={vals.mean():+.6f}  max={vals.max():+.6f}")

    # ── Sample bin ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SAMPLE: SZA bin 0, VZA bin 0, RAZ bin 0")
    print(f"  ({_bin_label(sza_lo[0], sza_hi[0])} / "
          f"{_bin_label(vza_lo[0], vza_hi[0])} / "
          f"{_bin_label(raz_lo[0], raz_hi[0])})")
    print("=" * 60)
    for var, label in [("sw_coefficients", "SW"), ("lw_coefficients", "LW"), ("ssw_coefficients", "SSW")]:
        coefs = ds[var].values[0, 0, 0, :]
        coef_str = "  ".join(f"{v:+.6f}" for v in coefs)
        print(f"  {label}: [{coef_str}]")

    # ── Optional plot ───────────────────────────────────────────────────────
    if plot:
        _plot(ds, sza_lo, sza_hi, vza_lo, vza_hi)


def _plot(ds, sza_lo, sza_hi, vza_lo, vza_hi):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not available — skipping plot")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Linear coefficient (a1) averaged over RAZ bins\n"
                 "Expected near 1.0 — deviations show where SRF correction is largest")

    for ax, (var, label) in zip(axes, [
        ("sw_coefficients",  "SW  a1"),
        ("lw_coefficients",  "LW  a1"),
        ("ssw_coefficients", "SSW  c2 (ssw_f weight)"),
    ]):
        # For SW/LW: idx 1 is a1 (linear). For SSW: idx 2 is c2 (ssw_f weight).
        coef_idx = 1 if var != "ssw_coefficients" else 2
        arr = ds[var].values[:, :, :, coef_idx]   # shape (5, 5, 5)
        grid = np.nanmean(arr, axis=2)             # average over RAZ bins → (5, 5)

        sza_labels = [_bin_label(lo, hi) for lo, hi in zip(sza_lo, sza_hi)]
        vza_labels = [_bin_label(lo, hi) for lo, hi in zip(vza_lo, vza_hi)]

        im = ax.imshow(grid.T, aspect="auto", origin="lower")
        ax.set_xticks(range(len(sza_labels)))
        ax.set_xticklabels(sza_labels, rotation=30, ha="right", fontsize=7)
        ax.set_yticks(range(len(vza_labels)))
        ax.set_yticklabels(vza_labels, fontsize=7)
        ax.set_xlabel("SZA bin")
        ax.set_ylabel("VZA bin")
        ax.set_title(label)
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Inspect a Libera unfiltering coefficients .nc file")
    parser.add_argument("path", help="Path to the .nc file")
    parser.add_argument("--plot", action="store_true", help="Show a geometry heatmap of the linear coefficients")
    args = parser.parse_args()

    if not Path(args.path).exists():
        print(f"File not found: {args.path}", file=sys.stderr)
        sys.exit(1)

    inspect(args.path, plot=args.plot)


if __name__ == "__main__":
    main()
