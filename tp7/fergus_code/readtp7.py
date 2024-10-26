# -*- coding: utf-8 -*-
"""
Created on Mon May 24 10:47:12 2021

@author: frdma
"""

import numpy as np
from datetime import datetime

#GOAL FOR THIS FILE IS TO READ ENTIRE DATA SETS: CHOOSE SPECIFIC RUNS, AVERAGE ALL RUNS
#1. Probe into file to find out how many lines, runs, columns (use matlab code)
#2. Build 2D array of all columns and all lines of data
#3. Build 3D array that stores all different runs for access
#4. Use loop to average all runs
#literal

#CLOUDY OCEAN DAY
#data has 337176 lines
#13 lines of junk, then 4000 lines of data with 12 columns
#after run, empty line, then repeats for 84 runs
#headers = np.genfromtxt(r"C:\Users\frdma\OneDrive\Documents\LASP stuff\ocecld_sz41L.tp7", dtype = (str), delimiter = "random", usecols = None, skip_header = 0, skip_footer = 333160)


#CLEAR OCEAN DAY
#data has 590058 lines
#13 lines of junk, then 4000 lines of data with 12 columns
#after run, empty line, then repeats for 147 runs
#headers0 = np.genfromtxt(r"C:\Users\frdma\OneDrive\Documents\LASP stuff\oceclr_sz41L.tp7", dtype = (str), delimiter = "random", usecols = None, skip_header = 0, skip_footer = 590045)


#SNOW DAY
#data has 1264410 lines
#13 lines of junk, then 4000 lines of data with 12 columns
#after run, empty line, then repeats for 315 runs
#headers1 = np.genfromtxt(r"C:\Users\frdma\OneDrive\Documents\LASP stuff\sno_sz41L.tp7", dtype = (str), delimiter = "random", usecols = None, skip_header = 0, skip_footer = 1264397)


#LAND DAY
#data has 2528820 lines
#12 lines of junk, then 4000 lines of data with 14 columns
#after run, empty line, then repeats for 630 runs
#headers2 = np.genfromtxt(r"C:\Users\frdma\OneDrive\Documents\LASP stuff\lnd_sz41L.tp7", dtype = (str), delimiter = "random", usecols = None, skip_header = 0, skip_footer = 2524177)

#adfa
#DCC DAY
#data has 252882 lines
#12 lines of junk, then 4000 lines of data with 14 columns
#after run, empty line, then repeats for 63 runs
#headers3 = np.genfromtxt(r"C:\Users\frdma\OneDrive\Documents\LASP stuff\dc_sz41.spectra.tp7", dtype = (str), delimiter = "random", usecols = None, skip_header = 0, skip_footer = 252807)


def readtp7(filename, verbose):
    starttime = datetime.now() #for computing runtime
    lines = open(r"/home/caleb/Masters/ceres_data_analysis/data/Modtran_Unfiltering_Tape7s_SZA41" + "/" + filename).readlines()
    
    #print data headers for referencing parameter columns
    if verbose:
        print('BEGINNING OF DATA FILE:')
        print('')
        for j in range(15):
            print(lines[j])
        print('')
        print('')
    
    #determine if file is night (wavenumer 10-10,000) 
    #or day (wavenumber 10-40,000)
    firstline = lines[0]
    if firstline[0] == 'F': #daytime SW file
        size = 4000
    if firstline[0] == 'T': #nighttime LW file
        size = 4996
    
    #calculate number of columns and number of runs
    samplerow = lines[-2]
    endofrun = lines[-1]
    numcols = 0
    numruns = 0
    for i in range(len(samplerow)):
        if samplerow[i] == '.':
            numcols = numcols + 1
    for i in range(len(lines)):
        if lines[i] == endofrun:
            numruns = numruns + 1
    
    #depending on filetype, different size headers
    if numcols == 12:
        start = 13
    if numcols == 14:
        start = 12
    
    #build empty array with calculated dimensions and populate
    grid = np.zeros((size, numcols, numruns))
    for i in np.arange(0,numruns,1):
        idxs = start + (i * ((size + 1) + start))
        idxl = idxs + size
        run = lines[idxs:idxl]
        #np.genfromtxt much more efficient than split function
        grid[:,:,i] = np.genfromtxt(run, dtype = (float), usecols = None)
        if verbose:
            print("run ", i, " processed")
    
    #compute runtime and print
    endtime = datetime.now()
    if verbose:
        print("Time taken to read data: ", (endtime - starttime))
    return grid
#    
#cldyoceandata = readtp7("\ocecld_sz41L.tp7")
#cldynightdata = readtp7("\ocecld_vz.tp7")

#clroceandata = readtp7("\oceclr_sz41L.tp7")

#snowdata = readtp7("\sno_sz41L.tp7")

#dccdata = readtp7("\dc_sz41.spectra.tp7")
#dccnightdata = readtp7("\dc_vz.spectra.tp7")

#landdata = readtp7("\lnd_sz41L.tp7")


# TO DO LIST
    #   Refine header data crunch for geometry cards 3 and 3A2
    #       Improve description for card3A2 X
    #       Figure out way to split card 3 path length
    #   Figure out geometry
    #       What do card 3 and 3A2 mean
    #       Visualize geometry on paper
    #   Try to correlate radiance with scene
    #       Visually match spectra with what should be based on geometry
    #       Plot unflitered integrated radiance against scene type
    #   Bruce task
    #       Try to compare integrated radiance with values in Leb paper
    #       Compare spectra with plots he sent
    #   Peter task
    #       Fit quadratic curve to filt vs unfilt scatter y = ax^2 + bx + c
    #       Update with rest of summer / future


