"""Shared fixtures for all tests."""
import numpy as np
import pandas as pd
import pytest

from prod.std.standard_method import SZA_BINS, VZA_BINS, RAZ_BINS, SCENE_TYPES, CLOUD_VALUES


@pytest.fixture
def sample_dataset():
    """Synthetic describer_df with 20 rows all inside bin (scene=Land, cloud=0, sza=0, vza=0, raz=0)."""
    rng = np.random.default_rng(42)
    n = 20
    sw_f = rng.uniform(70, 90, n)
    lw_f = rng.uniform(25, 35, n)
    ssw_f = rng.uniform(8, 12, n)
    sw_u = 0.5 + 1.02 * sw_f + rng.normal(0, 0.1, n)
    lw_u = 0.3 + 1.01 * lw_f + rng.normal(0, 0.05, n)
    ssw_u = 0.2 + 1.015 * ssw_f + rng.normal(0, 0.05, n)
    tot_f = sw_f + lw_f
    tot_u = sw_u + lw_u

    return pd.DataFrame({
        "Scene": ["Land"] * n,
        "Cloud": [0] * n,
        "SZA": rng.uniform(1, 20, n),
        "VZA": rng.uniform(1, 14, n),
        "RAZ": rng.uniform(1, 14, n),
        "Shortwave Filtered Rads (Integrated)": sw_f,
        "Shortwave Unfiltered Rads (Integrated)": sw_u,
        "Longwave Filtered Rads (Integrated)": lw_f,
        "Longwave Unfiltered Rads (Integrated)": lw_u,
        "Split Shortwave Filtered Rads (Integrated)": ssw_f,
        "Split Shortwave Unfiltered Rads (Integrated)": ssw_u,
        "Total Filtered Rads (Integrated)": tot_f,
        "Total Unfiltered Rads (Integrated)": tot_u,
    })


@pytest.fixture
def minimal_coefficients():
    """Coefficient dict with only bin (scene=0, cloud=0, sza=0, vza=0, raz=0) populated; all others None."""
    populated_key = (0, 0, SZA_BINS[0], VZA_BINS[0], RAZ_BINS[0])
    sw_coef  = np.concatenate([[0.5], np.array([0.0, 1.02, 0.01, 0.0, 0.0, 0.0])])
    ssw_coef = np.concatenate([[0.2], np.array([0.0, 1.015, 0.005, 0.0, 0.0, 0.0])])
    lw_coef  = np.array([0.3, 1.01, 0.0])
    tot_coef = np.array([0.4, 1.015, 0.0])

    return {
        (scene_idx, cloud, sza, vza, raz): (
            sw_coef, ssw_coef, lw_coef, tot_coef
        ) if (scene_idx, cloud, sza, vza, raz) == populated_key else (None, None, None, None)
        for scene_idx in range(len(SCENE_TYPES))
        for cloud in CLOUD_VALUES
        for sza in SZA_BINS
        for vza in VZA_BINS
        for raz in RAZ_BINS
    }
