"""
Integration tests for load_dataset — requires local .tp7 files.

Run integration tests:   pytest -m integration
Skip them:               pytest -m "not integration"
"""
from pathlib import Path

import pytest

from prod.std.standard_method import load_dataset

_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "Modtran_3-7_data"

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def full_dataset():
    if not list(_DATA_DIR.rglob("*.tp7")):
        pytest.skip("No .tp7 files found under data/Modtran_3-7_data/")
    return load_dataset(_DATA_DIR)


class TestLoadDataset:
    def test_multiple_scene_types(self, full_dataset):
        scenes = full_dataset["Scene"].unique()
        assert len(scenes) > 1, f"Expected multiple scene types, got: {scenes}"

    def test_row_count_sanity(self, full_dataset):
        # Each .tp7 file has many runs; across all files we expect hundreds of rows
        assert len(full_dataset) > 100

    def test_required_columns_present(self, full_dataset):
        required = [
            "SZA", "VZA", "RAZ",
            "Shortwave Unfiltered Rads (Integrated)",
            "Longwave Unfiltered Rads (Integrated)",
            "Shortwave Filtered Rads (Integrated)",
            "Longwave Filtered Rads (Integrated)",
            "Split Shortwave Filtered Rads (Integrated)",
        ]
        for col in required:
            assert col in full_dataset.columns, f"Missing column: {col}"

    def test_no_nan_in_radiances(self, full_dataset):
        for col in ["Shortwave Filtered Rads (Integrated)", "Longwave Filtered Rads (Integrated)"]:
            assert full_dataset[col].notna().all()

    def test_raises_on_empty_directory(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No .tp7 files found"):
            load_dataset(tmp_path)
