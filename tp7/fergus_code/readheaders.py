import numpy as np

#this function will isolate the headers only and store each header
#into a 2D grid - access header data with: [run][line][character]
def readheaders(filename):
    lines = open(r"/Users/calebkumar/Desktop/Desktop_Caleb_MacBook_Air/Masters/Modeling-Radiance/data/Modtran_Unfiltering_Tape7s_SZA00" + "/" + filename).readlines()
    
    
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
    
    #build empty list for headers
    lst = [['#' for col in range(start)] for row in range(numruns)]
    #populate list
    for i in np.arange(0,numruns,1):
        idxs = (i * ((size + 1) + start))
        idxl = idxs + start
        thing = lines[idxs:idxl]
        for j in range(start):
            lst[i][j] = thing[j]

    return lst

# clrocnheaders = readheaders('\oceclr_sz41L.tp7')
#dccheaders = readheaders('\dc_sz41.spectra.tp7')
#clrnightheaders = readheaders('\oceclr_vz.tp7')