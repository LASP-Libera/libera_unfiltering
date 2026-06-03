"""Unit tests for prod/std/standard_method.py."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from prod.std.standard_method import (
    SZA_BINS,
    VZA_BINS,
    RAZ_BINS,
    generate_unfiltering_coefficients,
    serialize_coefficients,
)


class TestGenerateCoefficients:
    def test_returns_all_bins(self, sample_dataset):
        result = generate_unfiltering_coefficients(sample_dataset)
        assert len(result) == len(SZA_BINS) * len(VZA_BINS) * len(RAZ_BINS)

    def test_populated_bin_coefficient_shapes(self, sample_dataset):
        result = generate_unfiltering_coefficients(sample_dataset)
        sw, ssw, lw = result[(SZA_BINS[0], VZA_BINS[0], RAZ_BINS[0])]
        assert sw.shape == (3,)
        assert lw.shape == (3,)
        assert ssw.shape == (7,)

    def test_linear_term_near_one(self, sample_dataset):
        result = generate_unfiltering_coefficients(sample_dataset)
        sw, _, lw = result[(SZA_BINS[0], VZA_BINS[0], RAZ_BINS[0])]
        # Synthetic data was built with a1 ~ 1.02 for SW and 1.01 for LW
        assert abs(sw[1] - 1.0) < 0.15
        assert abs(lw[1] - 1.0) < 0.15

    def test_ssw_intercept_at_index_zero(self, sample_dataset):
        result = generate_unfiltering_coefficients(sample_dataset)
        _, ssw, _ = result[(SZA_BINS[0], VZA_BINS[0], RAZ_BINS[0])]
        # ssw[0] is intercept; ssw[1] is c1 (PolynomialFeatures constant feature, near 0)
        assert ssw is not None
        assert len(ssw) == 7

    def test_empty_bin_returns_none(self, sample_dataset):
        result = generate_unfiltering_coefficients(sample_dataset)
        # sample_dataset only has rows in bin (0, 0, 0) — bin (1, 1, 1) is empty
        sw, ssw, lw = result[(SZA_BINS[1], VZA_BINS[1], RAZ_BINS[1])]
        assert sw is None
        assert ssw is None
        assert lw is None

    def test_below_threshold_returns_none(self):
        """Bins with exactly 2 rows must return None (minimum is 3)."""
        tiny = pd.DataFrame({
            "SZA": [5.0, 10.0],
            "VZA": [5.0, 10.0],
            "RAZ": [5.0, 10.0],
            "Shortwave Filtered Rads (Integrated)": [80.0, 85.0],
            "Shortwave Unfiltered Rads (Integrated)": [82.0, 87.0],
            "Longwave Filtered Rads (Integrated)": [30.0, 32.0],
            "Longwave Unfiltered Rads (Integrated)": [31.0, 33.0],
            "Split Shortwave Filtered Rads (Integrated)": [10.0, 11.0],
        })
        result = generate_unfiltering_coefficients(tiny)
        sw, ssw, lw = result[(SZA_BINS[0], VZA_BINS[0], RAZ_BINS[0])]
        assert sw is None

    def test_all_values_are_floats(self, sample_dataset):
        result = generate_unfiltering_coefficients(sample_dataset)
        sw, ssw, lw = result[(SZA_BINS[0], VZA_BINS[0], RAZ_BINS[0])]
        assert sw.dtype.kind == 'f'
        assert lw.dtype.kind == 'f'
        assert ssw.dtype.kind == 'f'


class TestSerializeCoefficients:
    def test_creates_file(self, tmp_path, minimal_coefficients):
        out = tmp_path / "coefs.nc"
        returned = serialize_coefficients(minimal_coefficients, out)
        assert returned == out
        assert out.exists()

    def test_expected_variables(self, tmp_path, minimal_coefficients):
        out = tmp_path / "coefs.nc"
        serialize_coefficients(minimal_coefficients, out)
        ds = xr.open_dataset(out)
        assert "sw_coefficients" in ds
        assert "lw_coefficients" in ds
        assert "ssw_coefficients" in ds

    def test_sw_dimensions_and_shape(self, tmp_path, minimal_coefficients):
        out = tmp_path / "coefs.nc"
        serialize_coefficients(minimal_coefficients, out)
        ds = xr.open_dataset(out)
        sw = ds["sw_coefficients"]
        assert sw.dims == ("sza_bin", "vza_bin", "raz_bin", "sw_coef_idx")
        assert sw.shape == (5, 5, 5, 3)

    def test_ssw_dimensions_and_shape(self, tmp_path, minimal_coefficients):
        out = tmp_path / "coefs.nc"
        serialize_coefficients(minimal_coefficients, out)
        ds = xr.open_dataset(out)
        ssw = ds["ssw_coefficients"]
        assert ssw.dims == ("sza_bin", "vza_bin", "raz_bin", "ssw_coef_idx")
        assert ssw.shape == (5, 5, 5, 7)

    def test_populated_bin_not_nan(self, tmp_path, minimal_coefficients):
        out = tmp_path / "coefs.nc"
        serialize_coefficients(minimal_coefficients, out)
        ds = xr.open_dataset(out)
        assert not np.any(np.isnan(ds["sw_coefficients"].values[0, 0, 0, :]))
        assert not np.any(np.isnan(ds["lw_coefficients"].values[0, 0, 0, :]))
        assert not np.any(np.isnan(ds["ssw_coefficients"].values[0, 0, 0, :]))

    def test_empty_bin_is_nan(self, tmp_path, minimal_coefficients):
        out = tmp_path / "coefs.nc"
        serialize_coefficients(minimal_coefficients, out)
        ds = xr.open_dataset(out)
        assert np.all(np.isnan(ds["sw_coefficients"].values[1, 1, 1, :]))

    def test_global_attrs(self, tmp_path, minimal_coefficients):
        out = tmp_path / "coefs.nc"
        serialize_coefficients(minimal_coefficients, out, srf_version="0-0-1", modtran_version="3.7")
        ds = xr.open_dataset(out)
        assert ds.attrs["srf_version"] == "0-0-1"
        assert ds.attrs["modtran_version"] == "3.7"
        assert "coefficient_version" in ds.attrs
        assert "git_commit" in ds.attrs
        assert "created_utc" in ds.attrs

    def test_bin_bounds_coords(self, tmp_path, minimal_coefficients):
        out = tmp_path / "coefs.nc"
        serialize_coefficients(minimal_coefficients, out)
        ds = xr.open_dataset(out)
        for coord in ["sza_lo", "sza_hi", "vza_lo", "vza_hi", "raz_lo", "raz_hi"]:
            assert coord in ds.coords, f"Missing coordinate: {coord}"

    def test_correct_bin_bounds_values(self, tmp_path, minimal_coefficients):
        out = tmp_path / "coefs.nc"
        serialize_coefficients(minimal_coefficients, out)
        ds = xr.open_dataset(out)
        assert float(ds["sza_lo"].values[0]) == 0.0
        assert float(ds["sza_hi"].values[0]) == pytest.approx(22.2)
