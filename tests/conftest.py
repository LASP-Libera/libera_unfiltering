"""Shared fixtures for all tests."""
import numpy as np
import pandas as pd
import pytest

from prod.std.standard_method import SZA_BINS, VZA_BINS, RAZ_BINS


@pytest.fixture
def sample_dataset():
    """Synthetic describer_df with 20 rows all inside bin index (0, 0, 0)."""
    rng = np.random.default_rng(42)
    n = 20
    sw_f = rng.uniform(70, 90, n)
    lw_f = rng.uniform(25, 35, n)
    ssw_f = rng.uniform(8, 12, n)
    sw_u = 0.5 + 1.02 * sw_f + rng.normal(0, 0.1, n)
    lw_u = 0.3 + 1.01 * lw_f + rng.normal(0, 0.05, n)

    return pd.DataFrame({
        "SZA": rng.uniform(1, 20, n),
        "VZA": rng.uniform(1, 14, n),
        "RAZ": rng.uniform(1, 14, n),
        "Shortwave Filtered Rads (Integrated)": sw_f,
        "Shortwave Unfiltered Rads (Integrated)": sw_u,
        "Longwave Filtered Rads (Integrated)": lw_f,
        "Longwave Unfiltered Rads (Integrated)": lw_u,
        "Split Shortwave Filtered Rads (Integrated)": ssw_f,
    })


@pytest.fixture
def minimal_coefficients():
    """Coefficient dict with only bin (0, 0, 0) populated; all others None."""
    bin_key = (SZA_BINS[0], VZA_BINS[0], RAZ_BINS[0])
    return {
        (sza, vza, raz): (
            np.array([0.5, 1.02, 0.0]),
            np.concatenate([[0.5], np.array([0.0, 1.02, 0.01, 0.0, 0.0, 0.0])]),
            np.array([0.3, 1.01, 0.0]),
        ) if (sza, vza, raz) == bin_key else (None, None, None)
        for sza in SZA_BINS
        for vza in VZA_BINS
        for raz in RAZ_BINS
    }
