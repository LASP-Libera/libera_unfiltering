import numpy as np
from importlib import resources

import pandas as pd


def load_srf(channel:str, version:str = "0-0-1"):
    file_name = f"./data/SRF/libera_srf_{channel}_v{version}.csv"

    return pd.read_csv(file_name, header=1, names=["Wavelength", "Response"])

def get_interpolated_srf(channel:str,
                         interpolation_points=np.linspace(0.3,100, 500),
                         version:str = "0-0-1"):
    base_srf_data = load_srf(channel, version)
    return np.interp(interpolation_points, base_srf_data.Wavelength, base_srf_data.Response)