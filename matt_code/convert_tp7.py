import io

import numpy as np
import pandas as pd
from pathlib import Path


def create_single_run_data(array_of_string_data, header_length):
    headerdata = array_of_string_data[0:header_length - 1]

    check_line = headerdata[10]
    if check_line.split()[0] == "sza":
        # Only tested for land sza41
        sza = float(headerdata[-1].split()[2])
        raz = float(headerdata[-1].split()[5])
        cloud_fraction = float(headerdata[-1].split()[8])

        if len(headerdata[5].split()) == 7:
            vza = float(headerdata[5].split()[2])
        elif len(headerdata[5].split()) == 6:
            poorly_formatted_header_pieces = headerdata[5].split()
            vza = poorly_formatted_header_pieces[2][:9]
        else:
            raise ValueError("")
        vza = 180 - float(vza)

        data_string = "".join(array_of_string_data[header_length - 1:-1])

        data_columns = ["Frequency", "TotalTransmission", "ThermalPath", "ThermalScattering", "SurfaceEmission",
                        "SolarScattering", "SingleScattering", "GroundReflected", "DirectReflected", "TotalRadiance",
                        "ReflectedSolar", "SolarObservation", "Depth", "thmsfcref"]
        keep_cols = ["Frequency", "SolarScattering", "GroundReflected", "TotalRadiance", "thmsfcref"]
        text_widths = (8, 11, 11, 11, 11, 11, 11, 11, 11, 11, 9, 9, 8, 11)
    elif check_line.split()[0] == "SOLZEN":
        # Only tested for snow sza00
        sza = float(headerdata[-1].split()[0])
        vza = float(headerdata[-1].split()[1])
        raz = float(headerdata[-1].split()[2])

        # Only care about clouds or not
        cloud_fraction = 1
        if float(headerdata[1].split()[4]) == 0:
            cloud_fraction = 0

        data_string = "".join(array_of_string_data[header_length - 1:-1])

        data_columns = ["Wavelength", "Frequency", "BRDF", "SurfaceAlbedo", "TotalRadiance", "SolarScattering",
                        "GroundReflected", "SurfaceDirected", "AtmosphericEmission", "SurfaceEmission", "thmsfcref",
                        "TotalTransmission"]
        keep_cols = ["Frequency", "GroundReflected", "thmsfcref", "SolarScattering", "TotalRadiance"]
        text_widths = (10, 8, 12, 12, 13, 13, 13, 13, 13, 13, 14, 10)
    else:
        raise ValueError("Check line was not as expected in the header.")
    dat = pd.read_fwf(io.StringIO(data_string), header=0, names=data_columns, usecols=keep_cols, widths=text_widths)
    dat.attrs = {"SZA": sza, "RAA": raz, "VZA": vza, "CloudFraction": cloud_fraction}

    # Calculated columns
    dat["Reflected"] = dat.SolarScattering + dat.GroundReflected - dat.thmsfcref
    dat["Emitted"] = dat.TotalRadiance - dat.Reflected
    dat["Wavelength"] = (1 / dat.Frequency) * 1e4
    dat["ShortWaveRadiance"] = ((dat.Frequency ** 2) * dat.Reflected)
    dat["LongWaveRadiance"] = ((dat.Frequency ** 2) * dat.Emitted)
    dat["TotalRadianceWatts"] = ((dat.Frequency ** 2) * dat.TotalRadiance)

    # Sort the whole DataFrame to match with ascending wavelength order
    dat = dat.sort_values(by="Wavelength")
    return dat


def load_all_runs_from_tp7(filepath: str or Path, header_lines: int):
    """ Land 41 had 12 header lines, snow00 has 13"""

    file_object = open(filepath, "r")

    all_lines = file_object.readlines()

    lines_per_run = 4000
    footer_lines = 1
    total_lines_per_run = lines_per_run + header_lines + footer_lines

    line_count = len(all_lines)
    run_count = line_count / total_lines_per_run

    if line_count % total_lines_per_run != 0:
        raise ValueError("Mismatch of line number to expected")

    modtran_runs = []
    for i in range(int(run_count)):
        line_start = i*total_lines_per_run
        line_end = (i+1)*total_lines_per_run
        line_data = all_lines[line_start:line_end]
        modtran_runs.append(create_single_run_data(line_data, header_lines))

    return modtran_runs


def create_metadata_arrays(run_data):
    vzas = []
    szas = []
    raas = []
    cfs = []
    for i in range(len(run_data)):
        vzas.append(run_data[i].attrs["VZA"])
        szas.append(run_data[i].attrs["SZA"])
        raas.append(run_data[i].attrs["RAA"])
        cfs.append(run_data[i].attrs["CloudFraction"])
    vzas = np.array(vzas)
    szas = np.array(szas)
    raas = np.array(raas)
    cfs = np.array(cfs)

    vzas = np.where(vzas==0, 0.1, vzas)
    szas = np.where(szas==0, 0.1, szas)
    raas = np.where(raas==0, 0.1, raas)
    return((szas, vzas, raas, cfs))