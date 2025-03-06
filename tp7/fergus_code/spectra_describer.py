import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from readtp7 import readtp7
from readheaders import readheaders


#TP7 data
filename = "sno_sz85L.tp7"
tp7data = readtp7(filename, False)
headerdata = readheaders(filename)

# print("FUCK \n", tp7data)

#Figure out how many runs there are
numruns = np.shape(tp7data)[2]
# print(numruns)

ddf = None
#=====Step 1: build a scene type description dataframe=====#

#Initialize every descriptor item (these are only the variables that change)
scene = []
runnumber = []

sza = np.zeros(numruns)
vza = np.zeros(numruns)
raz = np.zeros(numruns)

met = np.zeros(numruns)
print("sza", sza)
cldtype = []
cldalt = np.zeros(numruns)
cldthick = np.zeros(numruns)
cldext = np.zeros(numruns)

surfacetemp = np.zeros(numruns)
albedomodel = np.zeros(numruns)

atmmodel = []
season = []
aerosoltype = []
windspd = np.zeros(numruns)


#Customize header descriptions based on filename
title = ""
if filename == "lnd_sz00L.tp7":
    title = "Land"
    for i in range(numruns):
        scene.append(title)
        runnumber.append(i)
        sza[i] = headerdata[i][10].split()[2]
        raz[i] = headerdata[i][10].split()[5]
        #deal with card 3 for vza
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
        
      
            
            
            
    vza = 180 - vza
    #Create as 630 long dataframe with Scenetype, SZA, VZA, RAZ, Atmmodel, Surfacetemp, Surfacealbedo, Season, Aerosoltype, Met range, Windspeed, Cld Type, Cld Alt, Cld Thick, Cld Ext, runnum
    d = {'Scene': scene, 'SZA': sza, 'VZA': vza, 'RAZ': raz, 'Atm model': atmmodel, 'Sfc Temp': surfacetemp, 'Albedo model': albedomodel, 'Season': season, 'Aerosol Type': aerosoltype, 'Met Range': met, 'Wind spd (ms-1)': windspd, 'Cld Type': cldtype, 'Cld Alt (km)': cldalt, "Cld Thick (km)": cldthick, 'Cld Ext Coef (km-1)': cldext, 'Run #': runnumber}
    ddf = pd.DataFrame(data=d)
    
elif filename == "ocecld_sz60L.tp7":
    title = "Cloudy Ocean"
    for i in range(numruns):
        scene.append(title)
        runnumber.append(i)
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
        
    #Create as 84 long dataframe with Scenetype, SZA, VZA, RAZ, Cld Type, Cld Alt, Cld Thick, Cld Ext, runnum
    d = {'Scene': scene, 'SZA': sza, 'VZA': vza, 'RAZ': raz, 'Cld Type': cldtype, 'Cld Alt (km)': cldalt, "Cld Thick (km)": cldthick, 'Cld Ext Coef (km-1)': cldext, 'Run #': runnumber}
    ddf = pd.DataFrame(data=d)
    
elif filename == "\oceclr_sz41L.tp7":
    title = "Clear Ocean"
    for i in range(numruns):
        scene.append(title)
        runnumber.append(i)
        sza[i] = headerdata[i][11].split()[0]
        vza[i] = headerdata[i][11].split()[1]
        raz[i] = headerdata[i][11].split()[2]
        met[i] = headerdata[i][1].split()[6]
        

    #Create as 147 long dataframe with Scenetype, SZA, VZA, RAZ, Met range, runnum
    d = {'Scene': scene, 'SZA': sza, 'VZA': vza, 'RAZ': raz, 'Met Range': met, 'Run #': runnumber}
    ddf = pd.DataFrame(data=d)
        
elif filename == "sno_sz85L.tp7":
    title = "Snow"
    for i in range(len(headerdata)):
        scene.append(title)
        runnumber.append(i)
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
        
    #Create as 315 long dataframe with Scenetype, SZA, VZA, RAZ, Surfacetemp, Surfacealbedo, Cld Type, Cld Alt, Cld Thick, Cld Ext, runnum
    d = {'Scene': scene, 'SZA': sza, 'VZA': vza, 'RAZ': raz, 'Sfc Temp': surfacetemp, 'Albedo model': albedomodel, 'Cld Type': cldtype, 'Cld Alt (km)': cldalt, "Cld Thick (km)": cldthick, 'Cld Ext Coef (km-1)': cldext, 'Run #': runnumber}
    ddf = pd.DataFrame(data=d)
    
    
elif filename == "dc_sz41.spectra.tp7":
    title = "Deep Convective Cloud"
    for i in range(numruns):
        scene.append(title)
        runnumber.append(i)
        sza[i] = headerdata[i][10].split()[2]
        raz[i] = headerdata[i][10].split()[5]
        cldalt[i] = headerdata[i][2].split()[1]
        #deal with card 3 for vza
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
        
    vza = 180 - vza
    #Create as 63 long dataframe with SZA, VZA, RAZ, Scenetype, Cloudalt
    d = {'Scene': scene, 'SZA': sza, 'VZA': vza, 'RAZ': raz, 'Cloud Alt (km)': cldalt, 'Run #': runnumber}
    ddf = pd.DataFrame(data=d)
    
    
else:
    print("Error with title")
    
#We now have a dataframe with a description for each run, customized by the filename
#describer_df will have a description of all of the variables that change across each run,
#but it doesn't tell the full story for each datafile
#Variables that do not change:
    #Land: basically everything changes
    #CloudyOcn: Tropical Model atm, spring/summer, maritime aerosols, VIS 100 km, surface albedo 0, temp 300 k 
    #ClrOcn: Tropical Model atm, spring/summer, no clouds, no wind, maritime aerosols
    #Snow: Subarctic Winter Model atm, fall/winter, tropospheric aerosols, VIS 300 km, 
    #Deepcloud: Tropical model atm, spring/summer, rural aerosols, VIS 20 km, no wind


#=====Step 2: Building a dataframe of spectra for each run=====#
#Will be numruns long, with 3 variables - lam (um), swrad (w/m2 sr-1 um-1), and lwrad (w/m2 sr-1 um-1)



#Select data columns based on filetype (number of columns changes)
colnums = np.zeros(5)
if np.shape(tp7data)[1] == 12:
    colnums[0] = 1
    colnums[1] = 6
    colnums[2] = 10
    colnums[3] = 5
    colnums[4] = 4
elif np.shape(tp7data)[1] == 14:
    colnums[0] = 0
    colnums[1] = 7
    colnums[2] = 13
    colnums[3] = 5
    colnums[4] = 9
else:
    print('Error with tp7data')
    
    
    
    
    
#Build np array of wavelength, swrad, lwrad, for each run
#create 3D array with (4000, 3, numruns)
rads= np.zeros((4000, 3, numruns))
for i in range(numruns):
    #=====READ IN DATA AND MAKE CONVERSIONS=====#
    freq = tp7data[:,int(colnums[0]),i] # units cm-1
    tot = tp7data[:,int(colnums[4]),i] # units w/cm2sr /cm-1 
    #tot is the total radiance
    
    sfctot = tp7data[:,int(colnums[1]),i]
    thmsfcref = tp7data[:,int(colnums[2]),i]
    pathscat = tp7data[:,int(colnums[3]),i]
    refl = pathscat + sfctot - thmsfcref #units w/cm2sr /cm-1
    #refl is the sw refelcted radiance
    
    emit = tot - refl #units w/cm2sr /cm-1
    #emit is the lw emmitted radiance (total - sw reflected)
    
    
    #conversions to radiance as a function of wavelength
    lam = (1/freq) * 1E4 #wavelength in microns 
    swrad = ((freq**2)*refl)  #SW radiance - units w/m2 sr-1 um-1
    lwrad = ((freq**2)*emit)  #LW radiance - units w/m2 sr-1 um-1

    #Populate grid
    rads[:,0,i] = lam
    rads[:,1,i] = swrad
    rads[:,2,i] = lwrad


#We now have an array rads which has the sw and lw radiance spectra from each run
#And a describer_df that has information on the scene for every


#=====PLOT FOR SANITY CHECK=====#
#access wavelength with rads[:,0,runnum], swrad with rads[:,1,runnum], lwrad with rads[:,2,runnum]
runnum = 0
# print('')
# print(describer_df)
# print(np.shape(rads))

# figure = plt.figure(2, figsize=(19,10))
# ax = plt.axes()
# ax.plot(rads[:,0,runnum], rads[:,1, runnum], 'r-', label='SW Radiance')
# #ax.plot(rads[:,0,runnum], rads[:,2, runnum], 'b-', label='LW Radiance')
# ax.legend(prop={"size":20})
# plt.title("%s Radiance Spectrum"%title, fontsize=40)
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.xlabel('Wavelength um', fontsize=30)
# plt.ylabel('Radiance $W m^{-2} um^{-1} sr^{-1}$', fontsize=30)
# plt.xlim(0,5)
# plt.ylim(0, 5E-4)

print(ddf)
print(np.shape(rads))

