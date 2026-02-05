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
from sklearn.metrics import r2_score, mean_squared_error

#Custom Imports
from tp7.tp7 import Tape7
from srfs.srfs import SRFS
from matt_code.convert_tp7 import load_all_runs_from_tp7
from matt_code.make_srf import get_interpolated_srf

