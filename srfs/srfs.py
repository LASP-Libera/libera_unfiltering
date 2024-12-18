import pandas as pd
import numpy as np
from typing import Union

class SRFS:
    def __init__(self, file_path):
        self.file_path = file_path
        self.srf_vals = None
        self.conversion_df = {}
        self.process_srf()

    def process_srf(self):
        srf = pd.read_csv(self.file_path)

        # loop through and create a column in the dataframe that holds the end of the wavelength range
        for index, row in srf.iterrows():
            next_index = index + 1
            if next_index not in srf.index:
                continue
            else:
                srf.loc[index, 'End_wv_range'] = srf.loc[next_index, 'wavelength [um]']

        # create the first row in the dataframe starting at wavelenght 0.0 and ending at the first wavelength from the csv
        new_row = srf.iloc[0].copy()
        new_row.loc['End_wv_range'] = new_row['wavelength [um]']
        new_row.loc['wavelength [um]'] = 0.0
        data, cols = new_row.values, new_row.index
        tdf = pd.DataFrame([data], columns=cols)
        srf = pd.concat([tdf, srf])

        # add the final end value to replace the NAN
        srf.loc[999, 'End_wv_range'] = 1001

        self.srf_vals = srf

        return
