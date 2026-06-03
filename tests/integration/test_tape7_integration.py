"""
Integration tests for Tape7 using real .tp7 files.

Run integration tests:   pytest -m integration
Skip them:               pytest -m "not integration"
"""
from pathlib import Path

import pytest

from tp7.tp7 import Tape7

_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "Modtran_3-7_data"

pytestmark = pytest.mark.integration

RADIANCE_COLS = [
    "Shortwave Unfiltered Rads (Integrated)",
    "Longwave Unfiltered Rads (Integrated)",
    "Shortwave Filtered Rads (Integrated)",
    "Longwave Filtered Rads (Integrated)",
    "Split Shortwave Filtered Rads (Integrated)",
    "Total Filtered Rads (Integrated)",
]


@pytest.fixture(scope="module")
def tape7():
    files = sorted(_DATA_DIR.rglob("*.tp7"))
    if not files:
        pytest.skip("No .tp7 files found under data/Modtran_3-7_data/")
    return Tape7(files[0])


class TestTape7Integration:
    def test_describer_df_is_nonempty(self, tape7):
        assert len(tape7.describer_df) > 0

    def test_required_radiance_columns_present(self, tape7):
        for col in RADIANCE_COLS:
            assert col in tape7.describer_df.columns, f"Missing column: {col}"

    def test_no_nan_in_radiance_columns(self, tape7):
        for col in RADIANCE_COLS:
            assert tape7.describer_df[col].notna().all(), f"NaN found in: {col}"

    def test_geometry_columns_present(self, tape7):
        for col in ["Scene", "SZA", "VZA", "RAZ"]:
            assert col in tape7.describer_df.columns

    def test_sza_in_valid_range(self, tape7):
        assert tape7.describer_df["SZA"].between(0, 90).all()

    def test_vza_in_valid_range(self, tape7):
        assert tape7.describer_df["VZA"].between(0, 90).all()

    def test_rads_shape(self, tape7):
        n = len(tape7.describer_df)
        assert tape7.rads.shape == (n, 3, 4000)

    def test_radiances_are_positive(self, tape7):
        # Integrated radiances should be non-negative physical quantities
        assert (tape7.describer_df["Shortwave Unfiltered Rads (Integrated)"] >= 0).all()
        assert (tape7.describer_df["Longwave Unfiltered Rads (Integrated)"] >= 0).all()
