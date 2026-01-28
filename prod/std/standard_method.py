from tp7.tp7 import Tape7
from srfs.srfs import SRFS
from matt_code.convert_tp7 import load_all_runs_from_tp7
from matt_code.make_srf import get_interpolated_srf

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

land_sza_41_runs = load_all_runs_from_tp7("./data/Modtran_Unfiltering_Tape7s_SZA41/lnd_sz41L.tp7", header_lines=12)

vzas = []
szas = []
raas = []
cfs = []
for i in range(len(land_sza_41_runs)):
    vzas.append(land_sza_41_runs[i].attrs["VZA"])
    szas.append(land_sza_41_runs[i].attrs["SZA"])
    raas.append(land_sza_41_runs[i].attrs["RAA"])
    cfs.append(land_sza_41_runs[i].attrs["CloudFraction"])
vzas = np.array(vzas)
szas = np.array(szas)
raas = np.array(raas)
cfs = np.array(cfs)

found_runs = []
first_found = -1
for i in range(len(land_sza_41_runs)):
    if 15 <= vzas[i] <= 30:
        if 41.4 <= szas[i] <= 60:
            if 15 <= raas[i] <= 60:
                if cfs[i] == 1:
                    found_runs.append(land_sza_41_runs[i])
                    if first_found == -1:
                        first_found = i
                    j=i
print(f"Found {len(found_runs)} runs")
print(f"First run found was index {first_found}")
print(f"Last run found was {j}")

wavelengths = found_runs[0].Wavelength

ssw_filter = get_interpolated_srf("ssw", interpolation_points=wavelengths)
sw_filter = get_interpolated_srf("sw", interpolation_points=wavelengths)
lw_filter = get_interpolated_srf("lw", interpolation_points=wavelengths)
total_filter = get_interpolated_srf("total", interpolation_points=wavelengths)

unfiltered_sw = []
filtered_sw = []
filtered_ssw = []

for i in range(len(found_runs)):
    sw_radiance = np.where(
        (wavelengths >= 0.35) & (wavelengths < 4.5),
        found_runs[i].TotalRadianceWatts,
        0
    )
    unfiltered_sw.append(np.trapz(sw_radiance, wavelengths))
    filtered_sw.append(np.trapz(sw_radiance*sw_filter, wavelengths))
    filtered_ssw.append(np.trapz(sw_radiance*ssw_filter, wavelengths))

filtered_sw = np.array(filtered_sw)
filtered_ssw = np.array(filtered_ssw)
unfiltered_sw = np.array(unfiltered_sw)

fit = np.polynomial.polynomial.Polynomial.fit(filtered_sw, unfiltered_sw, 2)
estimated = fit(filtered_sw)
r2_score(unfiltered_sw, estimated)
print(f"r^2 Value: {r2_score(unfiltered_sw, estimated)}")
print(f"Mean Squared Error: {mean_squared_error(unfiltered_sw, estimated)}")