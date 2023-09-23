# %%
import icartt
import pandas as pd
import numpy as np
import datetime
from math import pi
import os
import ast

# %%
#import csv files with information
file_info_file = "C:/Users/cphal/OneDrive/Desktop/Aerosols/Module A/Metadata/module_a_bins.csv"
binlocations = pd.read_csv(file_info_file)

# %%
#function purpose: get new file name
def get_new_name(filename, binlocations):
    #find associated row by looping through files
    campaign = ''
    instrument = ''
    i = 0
    campaign_found = False
    instrument_found = False
    while (i < len(binlocations) or (not campaign_found and not instrument_found)):
        if (not campaign_found and isinstance(binlocations.iloc[i]['Campaign'],str)):
            if(binlocations.iloc[i]['Campaign'] in filename):
                campaign = binlocations.iloc[i]['Campaign']
                campaign_found = True
        if (not instrument_found and isinstance(binlocations.iloc[i]['Filename/Instrument'],str)):
            if(binlocations.iloc[i]['Filename/Instrument'] in filename):
                instrument = binlocations.iloc[i]['Filename nickname']
                instrument_found = True
        i += 1

    if(not campaign_found or not instrument_found):
        print('FATAL ERROR: campaign/instrument not found for ' + filename)
        exit()
    
    #find new filename and get its index
    new_file_name = campaign + '_' + instrument + '_'

    return new_file_name

# %%
#create diameter dataframe
def createDiamDf():
    diameter_file = "C:/Users/cphal/OneDrive/Desktop/Aerosols/Module A/NASA diameters.csv"

    #diameters_info = pd.read_csv(diameter_file, delimiter=',', encoding='utf-8')
    file = open(diameter_file,'r')
    count = 0
    dfDiam = pd.DataFrame(columns = ['file_name', 'row', 'column_start', 'column_end', 'diameters'], index = range(0,23))

    #cycle through file, extract values, and put them where they're supposed to be
    for line in file:
        if count > 0:
            line.strip('/n')
            nextline = line.split('[')
            nextline[1] = '[' + nextline[1]
            firstpart = nextline[0].split(',')

            dfDiam.at[count-1, 'file_name'] = firstpart[0]
            dfDiam.at[count-1, 'row'] = firstpart[1]
            dfDiam.at[count-1, 'column_start'] = firstpart[2]
            dfDiam.at[count-1, 'column_end'] = firstpart[3]
            dfDiam.at[count-1, 'diameters'] = nextline[1]
        count += 1

    return dfDiam

# %%
#if not looping
filebeginning = 'C:/Users/cphal/OneDrive/Desktop/Aerosols/Module A/Input Tests/'
filename = filebeginning + 'DC3-LARGE-LAS-PSL_DC8_20120607_R2.ict'

# %%
#module a code
def module_a(filename, dfDiam):
        
    new_file_name = get_new_name(filename, binlocations)

    #create dfDiam
    ind = dfDiam.index[dfDiam['file_name'] == new_file_name].tolist()

    #make sure not empty
    if (len(ind) > 1):
        print('FATAL ERROR: more than 1 match for ' + new_file_name)
        exit(0)

    #format diameter input properly
    diameters = dfDiam.at[ind[0],'diameters']
    diameters = diameters.strip('][\n').split(',')
    diameters = [float(x) for x in diameters]
    size_dist_diameter_input = pd.Series(diameters)

    #load in ict file to python
    ict = icartt.Dataset(filename)
    df = ict.data[:]

    df = pd.DataFrame(df)

    varnames = [x for x in ict.variables]

    column_start = int(dfDiam.at[ind[0],'column_start'])
    column_end = int(dfDiam.at[ind[0],'column_end'])

    varnames = [x for x in ict.variables]
    binnames = varnames[column_start:column_end]

    size_dist_input_original = df[binnames]

    #transpose size_dist_input to get it to work with Module A
    size_dist_input = pd.DataFrame.transpose(size_dist_input_original)

    #create new vector to hold column names
    ser=pd.Series(range(np.size(size_dist_input,1)))
    #assign this to be the columns
    size_dist_input.rename(columns = ser, inplace = True)

    #replace all input values less than 0 with np.nan
    #FIXME: consider LLOD of instrument instead
    size_dist_input.mask(size_dist_input <= 0, np.nan , inplace=True )

    #create list to hold input values
    d_Nx_list = [10.5, 30]
    d_Vx_list = [25,40]
    bin_num = 3
    #get_inputs(d_Nx_list, d_Vx_list)
    #3 bins , d_Nx = [10.5, 30] , d_Vx = [25,40]

    #reformat date and time
    date_time = ict.times
    datetimedf = pd.DataFrame(date_time, columns= ['datetime'])
    date = str(date_time[0])
    date = datetime.datetime.strptime(date, '%Y-%m-%dT%H:%M:%S.%f000').strftime('%Y_%m_%d')

    #make sure properly formatted
    size_dist_diameter_input.reset_index(drop=True)
    size_dist_diameter_input = size_dist_diameter_input.astype(float)

    #calculate dVdlogdP using dNdlogdP
    diameter_power = np.power(size_dist_diameter_input, 3)
    diameter_power = diameter_power.to_numpy()
    size_dist_volume = pi/6*size_dist_input.mul(diameter_power,axis='index')

    #calculate dVdlogdP using dNdlogdP TEST PURPOSES ONLY
    #size_dist_volume_slow = pd.DataFrame().reindex_like(size_dist_input)
    #for m in range(np.size(size_dist_volume_slow,1)):
    #for n in range(len(size_dist_diameter_input)):
    #size_dist_volume_slow.values[n,m]=pi/6* np.power(size_dist_diameter_input, 3).values[n]*size_dist_input.values[n,m]

    #prepare for trapz calculation
    size_dist_input[np.isnan(size_dist_input)] = 0 
    size_dist_volume[np.isnan(size_dist_volume)] = 0 

    ln_size_dist_diameter_input=np.log(size_dist_diameter_input)

    #function purpose: find nearest value (d_N1, d_N2, d_V1, d_V2) in numpy array (instrument measured diameters)
    def find_nearest(array, value):
        n = [abs(i-value) for i in array]
        idx = n.index(min(n))
        return idx

    ##MODIFIED CODE
    #instead of having [num_bins] separate arrays with equal length, have [num_bins] columns in a dataframe
    #have a row for each data type needed to store an array in
    #multidimensional array! TODO: draw a picture of this

    #find nearest index in size_dist_diameter_input
    d_Nx_nearest = []
    d_Vx_nearest = []

    #for num_bins, find nearest index in size_dist_diameter input
    for i in range(len(d_Nx_list)):
        x = find_nearest(size_dist_diameter_input, d_Nx_list[i])
        d_Nx_nearest.append(x)

    for i in range(len(d_Vx_list)):
        x = find_nearest(size_dist_diameter_input, d_Vx_list[i])
        d_Vx_nearest.append(x)

    #create empty dataframes to hold num_bins: should be (num_bins) x (num_columns in size_dist)
    num_bins = len(d_Nx_list) + 1

    N_df = pd.DataFrame(index = range(np.size(size_dist_input,1)), columns = range(num_bins))
    F_N_df = pd.DataFrame(index = range(np.size(size_dist_input,1)), columns = range(num_bins))
    V_df = pd.DataFrame(index = range(np.size(size_dist_input,1)), columns = range(num_bins))
    F_V_df = pd.DataFrame(index = range(np.size(size_dist_input,1)), columns = range(num_bins))

    #add beginning and ending indexes for the divisions
    d_Nx_nearest.insert(0,0)
    d_Nx_nearest.append(len(size_dist_diameter_input))

    d_Vx_nearest.insert(0,0)
    d_Vx_nearest.append(len(size_dist_diameter_input))

    #for each bin, make a smaller dataframe with the diameter bins desired. Then apply trapz calculation by row
    #and add result to dataframe in adjacent bin
    for i in range(num_bins):
        small_df = size_dist_input.iloc[(d_Nx_nearest[i]):(d_Nx_nearest[i+1]),:] 
        N_df.iloc[:,i] = np.trapz(small_df, x=ln_size_dist_diameter_input[(d_Nx_nearest[i]):(d_Nx_nearest[i+1])], axis=0)

    for i in range(num_bins):
        small_df = size_dist_input.iloc[(d_Vx_nearest[i]):(d_Vx_nearest[i+1]),:] 
        V_df.iloc[:,i] = np.trapz(small_df, x=ln_size_dist_diameter_input[(d_Vx_nearest[i]):(d_Vx_nearest[i+1])], axis=0)

    #calculate areaXY_number and volumeXY_number
    areaXY_number = pd.Series(np.trapz(size_dist_input, x=ln_size_dist_diameter_input, axis=0))
    volumeXY_number = pd.Series(np.trapz(size_dist_volume, x=ln_size_dist_diameter_input, axis=0))

    #FIXME: question: can we replace 0 with np.nan?
    areaXY_number.replace(0, np.nan, inplace=True)
    volumeXY_number.replace(0, np.nan, inplace=True)

    #divide N by areaXY for f_N dataframe
    for i in range(num_bins):
        F_N_df.iloc[:,i] = N_df.iloc[:,i] / areaXY_number

    for i in range(num_bins):
        F_V_df.iloc[:,i] = V_df.iloc[:,i] / volumeXY_number

    #put in big dataframe!
    file_M_1 = pd.DataFrame() 
    file_M_1['datetime'] = datetimedf
    for i in range(num_bins):
        file_M_1['N'+str(i)] = N_df.iloc[:,i]
        file_M_1['N_Total'] = areaXY_number
        file_M_1['F_N'+str(i)] = F_N_df.iloc[:,i]
        file_M_1['V'+str(i)] = V_df.iloc[:,i]
    file_M_1['V_Total'] = volumeXY_number
    file_M_1['F_V'+str(i)] = F_V_df.iloc[:,i]

    filepath = 'C:/Users/cphal/OneDrive/Desktop/Aerosols/Module A/Module_A_out/'
    export_csv = file_M_1.to_csv(filepath+new_file_name+date+'.csv', index = None, header=True) 

# %%
#MAIN METHOD
#set directory
fileDirectory = str('C:/Users/cphal/OneDrive/Desktop/Aerosols/Module A/Input Tests/')

dfDiam = createDiamDf()

#get filenames in folder
path = os.chdir(fileDirectory)
with os.scandir(path) as entries:
    for entry in entries:
        filename = fileDirectory + entry.name
        print('Processing ' + entry.name)
        module_a(filename, dfDiam)


