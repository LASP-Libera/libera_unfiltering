"""Unit tests for tp7/modtran6.py — Modtran6NC class."""
import numpy as np
import pytest
import xarray as xr
from pathlib import Path

from tp7.modtran6 import Modtran6NC, CERES_SCENE_MAP, DCC_CLDC_THRESHOLD, DCC_CLD_OT_THRESHOLD

_REPO_ROOT = Path(__file__).parent.parent.parent
_SRF_DIR = _REPO_ROOT / "data" / "SRF"

_EXPECTED_RADIANCE_COLS = {
    "Shortwave Unfiltered Rads (Integrated)",
    "Longwave Unfiltered Rads (Integrated)",
    "Total Unfiltered Rads (Integrated)",
    "Split Shortwave Unfiltered Rads (Integrated)",
    "Shortwave Filtered Rads (Integrated)",
    "Longwave Filtered Rads (Integrated)",
    "Split Shortwave Filtered Rads (Integrated)",
    "Total Filtered Rads (Integrated)",
}

_EXPECTED_COLS = _EXPECTED_RADIANCE_COLS | {"Scene", "Cloud", "SZA", "VZA", "RAZ", "Run #"}


def _make_nc(tmp_path, filename="scene.nc", scene_id=5, cldc=0.0, cld_ot=0.0):
    """Write a minimal MODTRAN 6-like .nc to tmp_path and return its path."""
    n_vza, n_raa, n_sza = 2, 2, 2
    n_sw, n_lw = 3, 3

    ds = xr.Dataset(
        {
            "CERES_TRMM_SZA":  (["ceres_trmm_solar_zenith_angle"],     [10.0, 20.0]),
            "CERES_TRMM_VZA":  (["ceres_trmm_viewing_zenith_angle"],   [5.0,  15.0]),
            "CERES_TRMM_RAA":  (["ceres_trmm_relative_azimuth_angle"], [10.0, 20.0]),
            "CERES_TRMM_Scene_ID": float(scene_id),
            "FV3_CLDC":        float(cldc),
            "FV3_CLD_OT":      float(cld_ot),
            "MODTRAN6_SPECTRAL_RADIANCE_TOA_SW_WVL_CERES_TRMM": (
                ["ceres_trmm_viewing_zenith_angle",
                 "ceres_trmm_relative_azimuth_angle",
                 "ceres_trmm_solar_zenith_angle",
                 "wavelength_sw"],
                np.ones((n_vza, n_raa, n_sza, n_sw)),
            ),
            "MODTRAN6_SPECTRAL_RADIANCE_TOA_LW_WVL_CERES_TRMM": (
                ["ceres_trmm_viewing_zenith_angle",
                 "ceres_trmm_relative_azimuth_angle",
                 "ceres_trmm_solar_zenith_angle",
                 "wavelength_lw"],
                np.ones((n_vza, n_raa, n_sza, n_lw)),
            ),
        },
        coords={
            "wavelength_sw": (["wavelength_sw"], [400.0, 1000.0, 4000.0]),    # nm → 0.4–4.0 µm
            "wavelength_lw": (["wavelength_lw"], [6000.0, 20000.0, 50000.0]), # nm → 6–50 µm
        },
    )
    nc_path = tmp_path / filename
    ds.to_netcdf(nc_path)
    return nc_path


class TestDescriber:
    def test_row_count(self, tmp_path):
        nc = _make_nc(tmp_path)
        m = Modtran6NC(nc, srf_path=_SRF_DIR)
        assert len(m.describer_df) == 8  # 2 VZA × 2 RAA × 2 SZA

    def test_all_columns_present(self, tmp_path):
        nc = _make_nc(tmp_path)
        m = Modtran6NC(nc, srf_path=_SRF_DIR)
        assert _EXPECTED_COLS.issubset(set(m.describer_df.columns))

    def test_angles_match_ceres_grid(self, tmp_path):
        nc = _make_nc(tmp_path)
        m = Modtran6NC(nc, srf_path=_SRF_DIR)
        assert set(m.describer_df["SZA"].unique()) == {10.0, 20.0}
        assert set(m.describer_df["VZA"].unique()) == {5.0, 15.0}
        assert set(m.describer_df["RAZ"].unique()) == {10.0, 20.0}

    def test_run_number_is_sequential(self, tmp_path):
        nc = _make_nc(tmp_path)
        m = Modtran6NC(nc, srf_path=_SRF_DIR)
        assert list(m.describer_df["Run #"]) == list(range(8))

    def test_radiance_columns_nonnegative(self, tmp_path):
        nc = _make_nc(tmp_path)
        m = Modtran6NC(nc, srf_path=_SRF_DIR)
        for col in _EXPECTED_RADIANCE_COLS:
            assert (m.describer_df[col] >= 0).all(), f"Negative values in {col}"


class TestSceneAndCloud:
    def test_clear_ocean(self, tmp_path):
        nc = _make_nc(tmp_path, scene_id=5, cldc=0.0)
        m = Modtran6NC(nc, srf_path=_SRF_DIR)
        assert (m.describer_df["Scene"] == "Clear Ocean").all()
        assert (m.describer_df["Cloud"] == 0).all()

    def test_cloudy_ocean_scene_id(self, tmp_path):
        nc = _make_nc(tmp_path, scene_id=18, cldc=0.5)
        m = Modtran6NC(nc, srf_path=_SRF_DIR)
        assert (m.describer_df["Scene"] == "Cloudy Ocean").all()

    def test_cloud_flag_one_when_cldc_positive(self, tmp_path):
        nc = _make_nc(tmp_path, scene_id=18, cldc=0.5)
        m = Modtran6NC(nc, srf_path=_SRF_DIR)
        assert (m.describer_df["Cloud"] == 1).all()

    def test_cloud_flag_zero_when_cldc_zero(self, tmp_path):
        nc = _make_nc(tmp_path, scene_id=5, cldc=0.0)
        m = Modtran6NC(nc, srf_path=_SRF_DIR)
        assert (m.describer_df["Cloud"] == 0).all()

    def test_dcc_override_takes_precedence_over_scene_id(self, tmp_path):
        # Scene_ID=5 is normally Clear Ocean, but DCC thresholds win
        nc = _make_nc(tmp_path, scene_id=5, cldc=DCC_CLDC_THRESHOLD, cld_ot=DCC_CLD_OT_THRESHOLD + 1.0)
        m = Modtran6NC(nc, srf_path=_SRF_DIR)
        assert (m.describer_df["Scene"] == "Deep Convective Cloud").all()
        assert (m.describer_df["Cloud"] == 1).all()

    def test_dcc_not_triggered_below_ot_threshold(self, tmp_path):
        # Full cloud cover but low OT → not DCC, falls through to Scene_ID
        nc = _make_nc(tmp_path, scene_id=18, cldc=DCC_CLDC_THRESHOLD, cld_ot=DCC_CLD_OT_THRESHOLD - 1.0)
        m = Modtran6NC(nc, srf_path=_SRF_DIR)
        assert (m.describer_df["Scene"] == "Cloudy Ocean").all()

    def test_unknown_scene_id_raises(self, tmp_path):
        nc = _make_nc(tmp_path, scene_id=999)
        with pytest.raises(ValueError, match="Unknown CERES_TRMM_Scene_ID: 999"):
            Modtran6NC(nc, srf_path=_SRF_DIR)