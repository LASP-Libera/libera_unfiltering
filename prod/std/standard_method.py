"""
Generates the Coefficients per specified bin for the traditional method

Author: Caleb Kumar
Date: 02/05/2026
Version: 1.0.0
"""
# Standard Lib imports
import os

# Third Party imports
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

#Custom Imports
from tp7.tp7 import Tape7
from srfs.srfs import SRFS
from matt_code.convert_tp7 import load_all_runs_from_tp7
from matt_code.make_srf import get_interpolated_srf



def generate_multivariate_coefficients(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Uses the available data to generate coefficients for the traditional unfiltering algorithm
    :param dataset:
    :return:
    """

    coefficients = {}

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

    angular_bins = [
        (sza_bin, vza_bin, raz_bin)
        for sza_bin in SZA_BINS
        for vza_bin in VZA_BINS
        for raz_bin in RAZ_BINS
    ]

    for sza_bin, vza_bin, raz_bin in angular_bins:
        binned_data = dataset[
            (dataset["SZA"].between(*sza_bin)) &
            (dataset["VZA"].between(*vza_bin)) &
            (dataset["RAZ"].between(*raz_bin))
        ]
        filtered_sw = binned_data["Shortwave Filtered Rads (Integrated)"]
        unfiltered_sw = binned_data["Shortwave Unfiltered Rads (Integrated)"]
        filtered_lw = binned_data["Longwave Filtered Rads (Integrated)"]
        unfiltered_lw = binned_data["Longwave Unfiltered Rads (Integrated)"]

        filtered_ssw = binned_data["Split Shortwave Filtered Rads (Integrated)"]
        sw_coef, ssw_coef, lw_coef = None, None, None

        try:
            fit_sw = np.polynomial.polynomial.Polynomial.fit(filtered_sw, unfiltered_sw, 2)
            fit_lw = np.polynomial.polynomial.Polynomial.fit(filtered_lw, unfiltered_lw, 2)
            sw_coef = fit_sw.convert().coef
            lw_coef = fit_lw.convert().coef

            input = np.vstack((filtered_sw, filtered_ssw)).transpose()
            output = unfiltered_sw
            poly_model = PolynomialFeatures(degree=2)
            poly_x = poly_model.fit_transform(input)
            poly_model.fit(poly_x, output)
            regression = LinearRegression()
            regression.fit(poly_x, output)
            ssw_coef = regression.coef_

        except:
            print("No data for: ", (sza_bin, vza_bin, raz_bin))

        coefficients[(sza_bin, vza_bin, raz_bin)] = (sw_coef, ssw_coef, lw_coef)

    return coefficients