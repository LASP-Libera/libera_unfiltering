"""Tape7 class, turns tp7 data file into its own object"""
import numpy as np
import pandas as pd
from pathlib import Path
import scipy.integrate as integrate

try:
    from cloudpathlib import AnyPath
except ImportError:
    AnyPath = None


def _as_path(p):
    """Normalize a str/Path/S3Path to a path object with .open() and .name."""
    if isinstance(p, str):
        return AnyPath(p) if AnyPath is not None else Path(p)
    return p


_DEFAULT_SRF_DIR = Path(__file__).parent.parent / "data" / "SRF"


class Tape7:
    def __init__(self, filepath, srf_path=None):
        self.filepath = _as_path(filepath)
        self.srf_path = _as_path(srf_path) if srf_path is not None else _DEFAULT_SRF_DIR
        self.tests = []
        self.num_runs = 0
        self.header_data = None
        self.file_data = None
        self.rads = []
        self.describer_df = None
        self.title = None
        self.integrated_rads = None
        self._init_data()

    def _init_data(self):
        read_result = self._read_tp7(self.filepath)
        self.header_data, self.file_data = read_result[0], read_result[1]
        self.title = self._get_title(self.filepath)
        self.describer_df = self._build_scene_description()
        self.rads = self._compute_radiences()
        self.integrated_rads = self._integrate_radiances()

    @staticmethod
    def _parse_metadata(lines):
        day_or_night = lines[0]
        size = 4000 if day_or_night[0] == 'F' else 4996

        sample_row = lines[-2].split()
        num_cols = len(sample_row)
        num_runs = len([line for line in lines if line.strip() == lines[-1].strip()])

        start_index = 13 if num_cols == 12 else 12

        return size, num_cols, num_runs, start_index

    @staticmethod
    def _get_title(filepath):
        scene_type = filepath.name.split('_')[0]

        file_to_title = {
            "lnd": "Land",
            "ocecld": "Cloudy Ocean",
            "oceclr": "Clear Ocean",
            "sno": "Snow",
            "dc": "Deep Convective Cloud"
        }
        title = file_to_title.get(scene_type, -1)

        if title == -1:
            raise ValueError(f"Unrecognised scene prefix in filename: {filepath.name}")

        return title

    def _read_tp7(self, filepath):
        with filepath.open('r') as file:
            lines = file.readlines()

        size, num_cols, num_runs, start_index = self._parse_metadata(lines)

        grid = np.zeros((size, num_cols, num_runs))

        for i in range(num_runs):
            idxs = start_index + (i * ((size + 1) + start_index))
            run_lines = lines[idxs:idxs + size]
            grid[:, :, i] = np.genfromtxt(run_lines, dtype=float, usecols=None)

        lst = [['#' for col in range(start_index)] for row in range(num_runs)]

        for i in np.arange(0, num_runs, 1):
            idxs = (i * ((size + 1) + start_index))
            idxl = idxs + start_index
            thing = lines[idxs:idxl]
            for j in range(start_index):
                lst[i][j] = thing[j]

        return [lst, grid]

    def _build_scene_description(self):
        num_runs = self.num_runs = np.shape(self.file_data)[2]
        header_data = self.header_data

        scene, runnumber, sza, vza, raz = [], [], np.zeros(num_runs), np.zeros(num_runs), np.zeros(num_runs)
        cldtype, cldalt, cldthick, cldext = [], np.zeros(num_runs), np.zeros(num_runs), np.zeros(num_runs)
        surfacetemp, albedomodel, met, windspd = np.zeros(num_runs), np.zeros(num_runs), np.zeros(num_runs), np.zeros(
            num_runs)
        atmmodel, season, aerosoltype = [], [], []

        title = self._get_title(self.filepath)

        for i in range(num_runs):
            self._populate_scene_data(i, title, scene, runnumber, sza, vza, raz, header_data, cldtype,
                                     cldalt, cldthick, cldext, surfacetemp, albedomodel, met, windspd,
                                     atmmodel, season, aerosoltype)

        if title == "Land" or title == "Deep Convective Cloud":
            vza = 180 - vza
        df = self._create_description_dataframe(title, scene, sza, vza, raz, cldtype, cldalt, cldthick, cldext,
                                               surfacetemp, albedomodel, met, windspd, atmmodel, season, aerosoltype,
                                               runnumber)

        self.describer_df = df
        return self.describer_df

    def _populate_scene_data(self, i, title, scene, runnumber, sza, vza, raz, headerdata, cldtype,
                            cldalt, cldthick, cldext, surfacetemp, albedomodel, met, windspd,
                            atmmodel, season, aerosoltype):
        scene.append(title)
        runnumber.append(i)

        if title == "Land":
            sza[i] = headerdata[i][10].split()[2]
            raz[i] = headerdata[i][10].split()[5]

            card3 = headerdata[i][5].split()
            if len(card3) != 7:
                thing1 = card3[2][:9]
                thing2 = card3[2][9:]
                card3.append(card3[5])
                card3[5] = card3[4]
                card3[4] = card3[3]
                card3[3] = thing2
                card3[2] = thing1
            vza[i] = card3[2]

            if headerdata[i][1].split()[4] == '0':
                ctype = 'No cloud'
                calt = 0
                cthick = 0
                cext = 0
            if headerdata[i][1].split()[4] == '18':
                if headerdata[i][2].split()[0] == '1.000':
                    ctype = 'ICLD 18: cirrus cloud'
                    calt = 7
                    cthick = 1
                    cext = 4
                if headerdata[i][2].split()[0] == '2.000':
                    ctype = 'ICLD 18: cirrus cloud'
                    calt = 7
                    cthick = 2
                    cext = 6
            if headerdata[i][1].split()[4] == '3':
                ctype = 'ICLD 3: stratus cloud'
                calt = 0.5
                cthick = 0.2
                cext = 28
            if headerdata[i][1].split()[4] == '1':
                ctype = 'ICLD 1: cumulus cloud'
                calt = 0.66
                cthick = 2.34
                cext = 92.6

            cldtype.append(ctype)
            cldalt[i] = calt
            cldthick[i] = cthick
            cldext[i] = cext

            if headerdata[i][0].split()[1] == '1':
                atmmodel.append('Tropical')
            if headerdata[i][0].split()[1] == '2':
                atmmodel.append('Midlatitude Summer')
            if headerdata[i][0].split()[1] == '6':
                atmmodel.append('1976 US Standard')
            if headerdata[i][0].split()[1] == '3':
                atmmodel.append('Midlatitude Winter')

            if headerdata[i][1].split()[0] == '10':
                aerosoltype.append('Desert')
            if headerdata[i][1].split()[0] == '1':
                aerosoltype.append('Rural')

            if headerdata[i][1].split()[1] == '1':
                season.append('Spring/Summer')
            if headerdata[i][1].split()[1] == '2':
                season.append('Fall/Winter')

            surfacetemp[i] = headerdata[i][0].split()[14]
            albedomodel[i] = headerdata[i][0].split()[15]
            met[i] = headerdata[i][1].split()[6]
            windspd[i] = headerdata[i][1].split()[7]

        elif title == "Cloudy Ocean":

            sza[i] = headerdata[i][11].split()[0]
            vza[i] = headerdata[i][11].split()[1]
            raz[i] = headerdata[i][11].split()[2]

            if headerdata[i][1].split()[4] == '3':
                ctype = 'ICLD 3: stratus cloud'
                calt = 0.5
                cthick = 0.2
                cext = 28
            if headerdata[i][1].split()[4] == '1':
                ctype = 'ICLD 1: cumulus cloud'
                calt = 0.66
                cthick = 2.34
                cext = 92.6
            if headerdata[i][1].split()[4] == '18':
                if headerdata[i][2].split()[0] == '1.000':
                    ctype = 'ICLD 18: cirrus cloud'
                    calt = 7
                    cthick = 1
                    cext = 4
                if headerdata[i][2].split()[0] == '2.000':
                    ctype = 'ICLD 18: cirrus cloud'
                    calt = 7
                    cthick = 2
                    cext = 6

            cldtype.append(ctype)
            cldalt[i] = calt
            cldthick[i] = cthick
            cldext[i] = cext

        elif title == "Clear Ocean":
            sza[i] = headerdata[i][11].split()[0]
            vza[i] = headerdata[i][11].split()[1]
            raz[i] = headerdata[i][11].split()[2]
            met[i] = headerdata[i][1].split()[6]

        elif title == "Snow":
            sza[i] = headerdata[i][11].split()[0]
            vza[i] = headerdata[i][11].split()[1]
            raz[i] = headerdata[i][11].split()[2]
            surfacetemp[i] = headerdata[i][0].split()[14]
            albedomodel[i] = headerdata[i][0].split()[15]

            if headerdata[i][1].split()[4] == '0':
                ctype = 'No cloud'
                calt = 0
                cthick = 0
                cext = 0
            if headerdata[i][1].split()[4] == '19':
                ctype = 'ICLD 19: subvisual cirrus cloud'
                calt = 10
                cthick = 0.3
                cext = 1
            if headerdata[i][1].split()[4] == '18':
                ctype = 'ICLD 18: cirrus cloud'
                calt = 0.5
                cthick = 1
                cext = 2
            if headerdata[i][1].split()[4] == '3':
                ctype = 'ICLD 3: stratus cloud'
                calt = 0.1
                cthick = 3
                cext = 30
            if headerdata[i][1].split()[4] == '4':
                ctype = 'ICLD 4: strato-cumulus cloud'
                calt = 0.1
                cthick = 0.4
                cext = 10

            cldtype.append(ctype)
            cldalt[i] = calt
            cldthick[i] = cthick
            cldext[i] = cext

        elif title == "Deep Convective Cloud":
            sza[i] = headerdata[i][10].split()[2]
            raz[i] = headerdata[i][10].split()[5]
            cldalt[i] = headerdata[i][2].split()[1]
            card3 = headerdata[i][5].split()
            if len(card3) != 7:
                thing1 = card3[2][:9]
                thing2 = card3[2][9:]
                card3.append(card3[5])
                card3[5] = card3[4]
                card3[4] = card3[3]
                card3[3] = thing2
                card3[2] = thing1
            vza[i] = card3[2]

    def _create_description_dataframe(self, title, scene, sza, vza, raz, cldtype, cldalt, cldthick, cldext, surfacetemp,
                                     albedomodel, met, windspd, atmmodel, season, aerosoltype, runnumber):
        mapping = {"No Cloud": 0, "Anycloud": 1}

        data = None
        if title in ["Land", "Cloudy Ocean", "Snow"]:
            data = {'Scene': scene, 'SZA': sza, 'VZA': vza, 'RAZ': raz, 'Cloud': cldtype, 'Run #': runnumber}
            df = pd.DataFrame(data)
            df["Cloud"] = df["Cloud"].apply(lambda x: "No Cloud" if x.lower() == "no cloud" else "Anycloud")
            df["Cloud"] = df["Cloud"].map(mapping)
            return df

        elif title == "Clear Ocean":
            data = {'Scene': scene, 'SZA': sza, 'VZA': vza, 'RAZ': raz, "Cloud": [0] * len(sza), 'Run #': runnumber}

        elif title == "Deep Convective Cloud":
            data = {'Scene': scene, 'SZA': sza, 'VZA': vza, 'RAZ': raz, 'Cloud': [1] * len(sza), 'Run #': runnumber}

        return pd.DataFrame(data)

    def _compute_radiences(self):
        num_runs = self.num_runs
        tp7data = self.file_data
        colnums = self._get_column_numbers(tp7data)

        self.rads = np.zeros((4000, 3, num_runs))

        for i in range(num_runs):
            freq, refl, emit = self._calculate_radiances_for_run(i, tp7data, colnums)
            self.tests.append(freq[0])
            lam = (1.0000000000000000000000000 / freq) * 1E4
            swrad = (freq ** 2.00) * refl
            lwrad = (freq ** 2.00) * emit

            self.rads[:, 0, i] = lam[::-1]
            self.rads[:, 1, i] = swrad[::-1]
            self.rads[:, 2, i] = lwrad[::-1]

        self.rads = np.transpose(self.rads, (2, 1, 0))

        return self.rads

    def _get_column_numbers(self, tp7data):
        if tp7data.shape[1] == 12:
            return [1, 6, 10, 5, 4]
        elif tp7data.shape[1] == 14:
            return [0, 7, 13, 5, 9]
        else:
            raise ValueError("Error with tp7data: Unexpected number of columns")

    def _calculate_radiances_for_run(self, run_index, tp7data, colnums):
        freq = tp7data[:, int(colnums[0]), run_index]
        tot = tp7data[:, int(colnums[4]), run_index]
        sfctot = tp7data[:, int(colnums[1]), run_index]
        thmsfcref = tp7data[:, int(colnums[2]), run_index]
        pathscat = tp7data[:, int(colnums[3]), run_index]
        refl = pathscat + sfctot - thmsfcref
        emit = tot - refl

        return freq, refl, emit

    def load_srf(self, channel: str, version: str = "0-0-1"):
        srf_file = self.srf_path / f"libera_srf_{channel}_v{version}.csv"
        with srf_file.open('r') as f:
            return pd.read_csv(f, header=1, names=["Wavelength", "Response"])

    def get_interpolated_srf(self, channel: str,
                             interpolation_points=np.linspace(0.3, 100, 500),
                             version: str = "0-0-1"):
        base_srf_data = self.load_srf(channel, version)
        return np.interp(interpolation_points, base_srf_data.Wavelength, base_srf_data.Response)

    def _integrate_radiances(self):
        wavelengths = np.array(self.rads[0, 0, :])

        ssw_filter = self.get_interpolated_srf("ssw", interpolation_points=wavelengths)
        sw_filter = self.get_interpolated_srf("sw", interpolation_points=wavelengths)
        lw_filter = self.get_interpolated_srf("lw", interpolation_points=wavelengths)
        total_filter = self.get_interpolated_srf("total", interpolation_points=wavelengths)

        shortwave_unfiltered = [integrate.simpson(y=run[1], x=run[0]) for run in self.rads]
        longwave_unfiltered = [integrate.simpson(y=run[2], x=run[0]) for run in self.rads]

        ssw_filtered = [
            integrate.simpson(y=(run[1] * ssw_filter), x=run[0]) for run in self.rads
        ]
        sw_filtered = [
            integrate.simpson(y=(run[1] * sw_filter), x=run[0]) for run in self.rads
        ]
        lw_filtered = [
            integrate.simpson(y=(run[2] * lw_filter), x=run[0]) for run in self.rads
        ]
        total_filtered = [
            integrate.simpson(
                y=(run[1] * total_filter) + (run[2] * total_filter), x=run[0]
            ) for run in self.rads
        ]

        self.describer_df["Shortwave Unfiltered Rads (Integrated)"] = shortwave_unfiltered
        self.describer_df["Longwave Unfiltered Rads (Integrated)"] = longwave_unfiltered
        self.describer_df["Shortwave Filtered Rads (Integrated)"] = sw_filtered
        self.describer_df["Longwave Filtered Rads (Integrated)"] = lw_filtered
        self.describer_df["Split Shortwave Filtered Rads (Integrated)"] = ssw_filtered
        self.describer_df["Total Filtered Rads (Integrated)"] = total_filtered

        return np.array([np.array(shortwave_unfiltered), np.array(longwave_unfiltered),
                         np.array(sw_filtered), np.array(lw_filtered), np.array(ssw_filtered), np.array(total_filtered)])
