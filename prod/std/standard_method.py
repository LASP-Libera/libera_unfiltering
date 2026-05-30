"""
Generates unfiltering coefficients for the traditional regression method.

Workflow:
  1. load_dataset()                      — parse all .tp7 files into a single DataFrame
  2. generate_unfiltering_coefficients() — fit regression per SZA/VZA/RAZ bin
  3. serialize_coefficients()            — write the result to a NetCDF file

Run end-to-end:
  from prod.std.standard_method import run
  run(data_dir="data/", output_path="coefficients.nc")
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from tp7.tp7 import Tape7, _as_path

logger = logging.getLogger(__name__)

# Viewing geometry bins (degrees) — consistent with Loeb et al. (2001)
SZA_BINS = [
    (0.0, 22.2),
    (22.2, 41.4),
    (41.4, 60.0),
    (60.0, 75.5),
    (75.5, 85.0),
]

VZA_BINS = [
    (0.0, 15.0),
    (15.0, 30.0),
    (30.0, 45.0),
    (45.0, 60.0),
    (60.0, 90.0),
]

RAZ_BINS = [
    (0.0, 15.0),
    (15.0, 60.0),
    (60.0, 120.0),
    (120.0, 165.0),
    (165.0, 180.0),
]


def load_dataset(data_dir, srf_dir=None) -> pd.DataFrame:
    """
    Parse every .tp7 file found under data_dir and return a single concatenated
    describer_df with integrated filtered and unfiltered radiances.

    Parameters
    ----------
    data_dir : str | Path | S3Path
        Root directory containing the MODTRAN .tp7 files (searched recursively).
    srf_dir : str | Path | S3Path, optional
        Directory containing the Libera SRF CSV files.  Defaults to the local
        data/SRF/ directory relative to the repo root.

    Returns
    -------
    pd.DataFrame
        Combined dataset with one row per MODTRAN run across all scene types.
    """
    data_path = _as_path(data_dir)
    tp7_files = sorted(data_path.rglob("*.tp7"))

    if not tp7_files:
        raise FileNotFoundError(f"No .tp7 files found under {data_dir}")

    frames = []
    for tp7_file in tp7_files:
        logger.info(f"Loading {tp7_file.name}")
        t7 = Tape7(tp7_file, srf_path=srf_dir)
        frames.append(t7.describer_df)

    dataset = pd.concat(frames, ignore_index=True)
    logger.info(f"Loaded {len(dataset)} runs from {len(tp7_files)} files")
    return dataset


def generate_unfiltering_coefficients(dataset: pd.DataFrame) -> dict:
    """
    Fit quadratic regression coefficients per SZA/VZA/RAZ bin.

    For SW and LW channels a univariate degree-2 polynomial is fit:
        m_u = a0 + a1*m_f + a2*m_f^2

    For SSW a multivariate degree-2 polynomial is fit using both the SSW and SW
    filtered radiances as predictors.

    Parameters
    ----------
    dataset : pd.DataFrame
        Output of load_dataset() — must contain the six integrated radiance columns.

    Returns
    -------
    dict
        Keys are (sza_bin, vza_bin, raz_bin) tuples.
        Values are (sw_coef, ssw_coef, lw_coef) where:
          sw_coef  : ndarray shape (3,)  — [a0, a1, a2]
          lw_coef  : ndarray shape (3,)  — [a0, a1, a2]
          ssw_coef : ndarray shape (7,)  — [intercept, c1..c6] where c1..c6 are
                     PolynomialFeatures(degree=2) weights for
                     [1, ssw_f, sw_f, ssw_f^2, ssw_f*sw_f, sw_f^2]
          Any of the three may be None if the bin contains insufficient data.
    """
    coefficients = {}

    angular_bins = [
        (sza_bin, vza_bin, raz_bin)
        for sza_bin in SZA_BINS
        for vza_bin in VZA_BINS
        for raz_bin in RAZ_BINS
    ]

    for sza_bin, vza_bin, raz_bin in angular_bins:
        binned = dataset[
            (dataset["SZA"].between(*sza_bin)) &
            (dataset["VZA"].between(*vza_bin)) &
            (dataset["RAZ"].between(*raz_bin))
        ]

        sw_coef = ssw_coef = lw_coef = None

        if len(binned) >= 3:
            try:
                filtered_sw   = binned["Shortwave Filtered Rads (Integrated)"]
                unfiltered_sw = binned["Shortwave Unfiltered Rads (Integrated)"]
                filtered_lw   = binned["Longwave Filtered Rads (Integrated)"]
                unfiltered_lw = binned["Longwave Unfiltered Rads (Integrated)"]
                filtered_ssw  = binned["Split Shortwave Filtered Rads (Integrated)"]

                fit_sw = np.polynomial.polynomial.Polynomial.fit(filtered_sw, unfiltered_sw, 2)
                sw_coef = fit_sw.convert().coef

                fit_lw = np.polynomial.polynomial.Polynomial.fit(filtered_lw, unfiltered_lw, 2)
                lw_coef = fit_lw.convert().coef

                X = np.vstack([filtered_ssw, filtered_sw]).T
                poly = PolynomialFeatures(degree=2)
                X_poly = poly.fit_transform(X)
                reg = LinearRegression()
                reg.fit(X_poly, unfiltered_sw.values)
                # Store intercept as index 0, feature weights as indices 1-6
                ssw_coef = np.concatenate([[reg.intercept_], reg.coef_])

            except Exception as e:
                logger.warning(f"Fit failed for bin {(sza_bin, vza_bin, raz_bin)}: {e}")
        else:
            logger.debug(f"Skipping sparse bin {(sza_bin, vza_bin, raz_bin)} ({len(binned)} samples)")

        coefficients[(sza_bin, vza_bin, raz_bin)] = (sw_coef, ssw_coef, lw_coef)

    filled = sum(1 for v in coefficients.values() if v[0] is not None)
    logger.info(f"Coefficients generated for {filled}/{len(angular_bins)} bins")
    return coefficients


def serialize_coefficients(coefficients: dict, output_path) -> Path:
    """
    Write the coefficient dict to a NetCDF file.

    Dimensions
    ----------
    sza_bin, vza_bin, raz_bin : int (0–4)
        Bin indices.  Corresponding angle ranges stored as auxiliary
        coordinates sza_lo/sza_hi, vza_lo/vza_hi, raz_lo/raz_hi.
    sw_coef_idx  : 0–2  — polynomial order [a0, a1, a2]
    lw_coef_idx  : 0–2  — polynomial order [a0, a1, a2]
    ssw_coef_idx : 0–6  — [intercept, 1, ssw_f, sw_f, ssw_f^2, ssw_f*sw_f, sw_f^2]

    Sparse bins (no data) are stored as NaN.

    Parameters
    ----------
    coefficients : dict
        Output of generate_unfiltering_coefficients().
    output_path : str | Path
        Destination path for the .nc file.

    Returns
    -------
    Path
        Resolved path to the written file.
    """
    n_sza, n_vza, n_raz = len(SZA_BINS), len(VZA_BINS), len(RAZ_BINS)

    sw_arr  = np.full((n_sza, n_vza, n_raz, 3), np.nan)
    lw_arr  = np.full((n_sza, n_vza, n_raz, 3), np.nan)
    ssw_arr = np.full((n_sza, n_vza, n_raz, 7), np.nan)

    for (sza_bin, vza_bin, raz_bin), (sw_coef, ssw_coef, lw_coef) in coefficients.items():
        i = SZA_BINS.index(sza_bin)
        j = VZA_BINS.index(vza_bin)
        k = RAZ_BINS.index(raz_bin)
        if sw_coef is not None:
            sw_arr[i, j, k]  = sw_coef
        if lw_coef is not None:
            lw_arr[i, j, k]  = lw_coef
        if ssw_coef is not None:
            ssw_arr[i, j, k] = ssw_coef

    ds = xr.Dataset(
        {
            "sw_coefficients": (
                ["sza_bin", "vza_bin", "raz_bin", "sw_coef_idx"],
                sw_arr,
                {
                    "long_name": "Shortwave unfiltering polynomial coefficients",
                    "description": "Quadratic fit: m_u = a0 + a1*m_f + a2*m_f^2",
                    "units": "W m-2 sr-1",
                },
            ),
            "lw_coefficients": (
                ["sza_bin", "vza_bin", "raz_bin", "lw_coef_idx"],
                lw_arr,
                {
                    "long_name": "Longwave unfiltering polynomial coefficients",
                    "description": "Quadratic fit: m_u = a0 + a1*m_f + a2*m_f^2",
                    "units": "W m-2 sr-1",
                },
            ),
            "ssw_coefficients": (
                ["sza_bin", "vza_bin", "raz_bin", "ssw_coef_idx"],
                ssw_arr,
                {
                    "long_name": "Split-shortwave unfiltering multivariate polynomial coefficients",
                    "description": (
                        "Multivariate degree-2 fit using SSW and SW filtered radiances. "
                        "Index 0 = intercept; indices 1-6 = PolynomialFeatures weights "
                        "for [1, ssw_f, sw_f, ssw_f^2, ssw_f*sw_f, sw_f^2]."
                    ),
                    "units": "W m-2 sr-1",
                },
            ),
        },
        coords={
            "sza_bin": range(n_sza),
            "vza_bin": range(n_vza),
            "raz_bin": range(n_raz),
            "sw_coef_idx":  range(3),
            "lw_coef_idx":  range(3),
            "ssw_coef_idx": range(7),
            "sza_lo": ("sza_bin", [b[0] for b in SZA_BINS]),
            "sza_hi": ("sza_bin", [b[1] for b in SZA_BINS]),
            "vza_lo": ("vza_bin", [b[0] for b in VZA_BINS]),
            "vza_hi": ("vza_bin", [b[1] for b in VZA_BINS]),
            "raz_lo": ("raz_bin", [b[0] for b in RAZ_BINS]),
            "raz_hi": ("raz_bin", [b[1] for b in RAZ_BINS]),
        },
        attrs={
            "title": "Libera unfiltering regression coefficients",
            "source": "MODTRAN 3.7 radiative transfer simulations",
            "method": "Scene-stratified quadratic regression (Loeb et al. 2001)",
            "created_utc": datetime.now(timezone.utc).isoformat(),
        },
    )

    output_path = Path(output_path)
    ds.to_netcdf(output_path)
    logger.info(f"Coefficients written to {output_path}")
    return output_path


def run(data_dir, output_path="coefficients.nc", srf_dir=None) -> Path:
    """
    End-to-end coefficient generation: load data, fit, and write .nc file.

    Parameters
    ----------
    data_dir : str | Path | S3Path
        Root directory containing MODTRAN .tp7 files.
    output_path : str | Path
        Destination for the output NetCDF file.
    srf_dir : str | Path | S3Path, optional
        SRF file directory.  Defaults to local data/SRF/.

    Returns
    -------
    Path
        Path to the written coefficients file.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    logger.info("Step 1: Loading dataset")
    dataset = load_dataset(data_dir, srf_dir=srf_dir)

    logger.info("Step 2: Generating coefficients")
    coefficients = generate_unfiltering_coefficients(dataset)

    logger.info("Step 3: Serializing to NetCDF")
    return serialize_coefficients(coefficients, output_path)
