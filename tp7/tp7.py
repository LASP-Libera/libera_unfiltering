"""Tape7 class, turns tp7 data file into its own object"""
import numpy as np
import pandas as pd
import os
import scipy.integrate as integrate
import time

class Tape7:
    def __init__(self, filepath: str, srf_wvlns, sw_srf, lw_srf):
        self.filepath = filepath
        self.tests = []
        self.num_runs = 0
        self.header_data = None
        self.file_data = None
        self.rads = []
        self.describer_df = None
        self.title = None
        self.integrated_rads = None
        self._init_data(srf_wvlns, sw_srf, lw_srf)

    def _init_data(self, srf_wvlns, sw_srf, lw_srf):
        """
        Initialize the tp7 file and set variables
        """
        s1 = time.time()
        read_result = self._read_tp7(self.filepath)
        s2 = time.time()
        # print("Read result sec", s2 - s1)

        self.header_data, self.file_data = read_result[0], read_result[1]

        s3 = time.time()
        self.title = self._get_title(self.filepath)
        s4 = time.time()
        # print("Title time", s4 - s3)

        s5 = time.time()
        self.describer_df = self._build_scene_description()
        s6 = time.time()
        # print("Describer df sec", s6 - s5)

        s7 = time.time()
        self.rads = self._compute_radiences()
        s8 = time.time()
        # print("radiance time", s8 - s7)

        s9 = time.time()
        self.integrated_rads = self._integrate_radiances(srf_wvlns, sw_srf, lw_srf)
        s10 = time.time()
        # print("integration time", s10 - s9)
        

    @staticmethod
    def _parse_metadata(lines):
        """
        Parses metadata from the headers of the tp7 file
        :param lines: list of file lines
        :return: tuple containing header data and metadata
        """

        day_or_night = lines[0]
        size = 4000 if day_or_night[0] == 'F' else 4996

        # Determine number of columns based on the last line before data ends
        sample_row = lines[-2].split()
        num_cols = len(sample_row)
        num_runs = len([line for line in lines if line.strip() == lines[-1].strip()])

        # Define starting index based on the number of columns detected
        start_index = 13 if num_cols == 12 else 12

        return size, num_cols, num_runs, start_index

    @staticmethod
    def _get_title(filepath: str):
        """
        Given the filepath to the data file, returns the Scene type/title of the data
        :param filepath : str the filepath to the tp7 data file
        :return : scene type of the file
        """
        file_name = os.path.basename(filepath)
        scene_type = file_name.split('_')[0]

        file_to_title = {
            "lnd": "Land",
            "ocecld": "Cloudy Ocean",
            "oceclr": "Clear Ocean",
            "sno": "Snow",
            "dc": "Deep Convective Cloud"
        }
        title = file_to_title.get(scene_type, -1)

        if title == -1:
            raise ValueError("File not found")

        return title

    def _read_tp7(self, filepath: str):
        """
        Reads the file given and turns the data into a numpy array
        :param filepath : str the filepath to the tp7 data file
        :return: file_data : numpy array of the tp7 file data
        """

        with open(filepath, 'r') as file:
            lines = file.readlines()

        size, num_cols, num_runs, start_index = self._parse_metadata(lines)

        # Initialize an empty grid array with calculated dimensions
        grid = np.zeros((size, num_cols, num_runs))

        # Iterate through each run and populate the grid
        for i in range(num_runs):
            idxs = start_index + (i * ((size + 1) + start_index))
            run_lines = lines[idxs:idxs + size]
            grid[:, :, i] = np.genfromtxt(run_lines, dtype=float, usecols=None)

        # empty list for headers
        lst = [['#' for col in range(start_index)] for row in range(num_runs)]

        for i in np.arange(0, num_runs, 1):
            idxs = (i * ((size + 1) + start_index))
            idxl = idxs + start_index
            thing = lines[idxs:idxl]
            for j in range(start_index):
                lst[i][j] = thing[j]

        return [lst, grid]


    def _build_scene_description(self):
        """
        Given the data file and all the data create a dataframe that describes the data by run
        Describes by scene type
        :return describer_df : pandas dataframe containing the description of the data by run
        """
        num_runs = self.num_runs = np.shape(self.file_data)[2]
        header_data = self.header_data

        # Initialize descriptor arrays
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
        """
        Given all info about the date file calculate the correct data for each scene type
        :param i: index of run number
        :param title: scene type
        :param scene: scene type
        :param runnumber: the run number
        :param sza:
        :param vza:
        :param raz:
        :param headerdata: file header data
        :param cldtype:
        :param cldalt:
        :param cldthick:
        :param cldext:
        :param surfacetemp:
        :param albedomodel:
        :param met:
        :param windspd:
        :param atmmodel:
        :param season:
        :param aerosoltype:
        :return:
        """

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
            # deal with card 3 for vza
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
        """
        Creates the dataframe per scene type with the correct data
        :param title:
        :param scene:
        :param sza:
        :param vza:
        :param raz:
        :param cldtype:
        :param cldalt:
        :param cldthick:
        :param cldext:
        :param surfacetemp:
        :param albedomodel:
        :param met:
        :param windspd:
        :param atmmodel:
        :param season:
        :param aerosoltype:
        :param runnumber:
        :return:
        """

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
        """
        Computes the radiences of the datafile from the tp7 data
        :return: nparray of the radiences
        """
        num_runs = self.num_runs
        tp7data = self.file_data
        colnums = self._get_column_numbers(tp7data)

        # Create the radiance array
        self.rads = np.zeros((4000, 3, num_runs))

        for i in range(num_runs):
            freq, refl, emit = self._calculate_radiances_for_run(i, tp7data, colnums)
            # print(f"freq values: {freq[0]:.20f}, wv value: {tp7data[:, 0, i][0]:.20f}, scene: {self.title}")
            self.tests.append(freq[0])
            # if self.title in ["Land", "Deep Convective Cloud"]:
            # convert to wavelength and to microns
            lam = (1.0000000000000000000000000 / freq) * 1E4
            swrad = (freq ** 2.00) * refl
            lwrad = (freq ** 2.00) * emit

            self.rads[:, 0, i] = lam[::-1]
            self.rads[:, 1, i] = swrad[::-1]
            self.rads[:, 2, i] = lwrad[::-1]

            # else:
            #     self.rads[:, 0, i] = tp7data[:, 0, i][::-1]
            #     # self.rads[:, 0, i] = freq
            #     self.rads[:, 1, i] = refl[::-1]
            #     self.rads[:, 2, i] = emit[::-1]

        self.rads = np.transpose(self.rads, (2, 1, 0))

        return self.rads


    def _get_column_numbers(self, tp7data):
        """
        Determine column numbers based on the shape of tp7data.
        """
        if tp7data.shape[1] == 12:
            return [1, 6, 10, 5, 4]
        elif tp7data.shape[1] == 14:
            return [0, 7, 13, 5, 9]
        else:
            raise ValueError("Error with tp7data: Unexpected number of columns")

    def _calculate_radiances_for_run(self, run_index, tp7data, colnums):
        """
        Calculate frequencies and radiances for a single run.
        """
        freq = tp7data[:, int(colnums[0]), run_index]
        tot = tp7data[:, int(colnums[4]), run_index]
        sfctot = tp7data[:, int(colnums[1]), run_index]
        thmsfcref = tp7data[:, int(colnums[2]), run_index]
        pathscat = tp7data[:, int(colnums[3]), run_index]
        # print(pathscat[0], sfctot[0], thmsfcref[0], tot[0])
        refl = pathscat + sfctot - thmsfcref
        emit = tot - refl

        return freq, refl, emit

    def _integrate_radiances(self, srf_wvlns, sw_srf, lw_srf):
        """
        Integrate all calculated radiences, without the SRFS - Unfiltered Ground Truth y value
        :return: integrated radiences : array
        """

        interpolated_sw_srf = np.interp(x=self.rads[0, 0, :], xp=srf_wvlns, fp=sw_srf)
        interpolated_lw_srf = np.interp(x=self.rads[0, 0, :], xp=srf_wvlns, fp=lw_srf)

        # rads = (run_num, wv, measurement)
        shortwave_unfiltered = [integrate.simpson(y=run[1], x=run[0]) for run in self.rads]
        longwave_unfiltered = [integrate.simpson(y=run[2], x=run[0]) for run in self.rads]

        shortwave_filtered = [integrate.simpson(y=(run[1] * interpolated_sw_srf), x=run[0]) for run in self.rads]
        longwave_filtered = [integrate.simpson(y=(run[2] * interpolated_lw_srf), x=run[0]) for run in self.rads]

        self.describer_df["Shortwave Unfiltered Rads (Integrated)"] = shortwave_unfiltered
        self.describer_df["Longwave Unfiltered Rads (Integrated)"] = longwave_unfiltered

        self.describer_df["Shortwave Filtered Rads (Integrated)"] = shortwave_filtered
        self.describer_df["Longwave Filtered Rads (Integrated)"] = longwave_filtered

        # self.describer_df["Wavelength"] = self.rads[0, 0, :]


        return np.array([np.array(shortwave_unfiltered), np.array(longwave_unfiltered),
                         np.array(shortwave_filtered), np.array(longwave_filtered)])
