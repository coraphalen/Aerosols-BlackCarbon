import icartt
import pandas as pd
import numpy as np
import datetime
from math import pi
import os

# Initialize a translation table to remove non-numeric characters
translation_table = str.maketrans('','','abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-/:;<=>?@[\\]^_`{|}~\t')

#import csv files with information
file_info_file = "C:/Users/cphal/OneDrive/Desktop/Aerosols/Module A/Metadata/module_a_bins.csv"
binlocations = pd.read_csv(file_info_file)

bin_diameters = "C:/Users/cphal/OneDrive/Desktop/Aerosols/Module A/Metadata/bin_diameters.csv"
bin_diameters_df = pd.read_csv(bin_diameters)


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


#function purpose: get bin starting string
def get_bin_loc(new_file_name,binlocations):
    ind_list = binlocations.index[binlocations['New File Name'] == new_file_name].tolist()

    if(len(ind_list) != 1):
        print('FATAL ERROR: Number of matches found not equal to 1 for ' + filename)
        exit()

    bin_loc = binlocations.iloc[ind_list[0]]['Bin Location']

    return bin_loc


#get units from file
def get_unit_info(varnames,ict):
    varDf = pd.DataFrame(columns=['var_name','unit','descr','descr_long'])

    #split rows of data and put into df
    for x in varnames:
        var = str(ict.variables[x])
        varAr = var.split(',')
        while len(varAr) < 4:
            varAr.append('')
        varDf.loc[len(varDf)] = varAr
    
    return varDf


#extract list of numeric values in string and then split it
def split_numeric(nextline):
    #then get list of diameters
    diameters = nextline.split(',')
    firstline = diameters[0].split()
    first_diam = firstline[len(firstline) - 1]
    diameters[0] = first_diam

    #delete nextline character
    str = diameters[len(diameters) - 1].replace('\n','')
    diameters[len(diameters) - 1] = str

    diameters = [float(x) for x in diameters]

    return diameters


#alternative method of getting diameters; searches lines for leadstr
#lead_str : from spreadsheet
def get_diameters_listed(filename, lead_str):
    #keep track of the number of asterisk strings! once we hit 2 then exit and throw error
    asterisks = 0

    #read lines until line contains the string we want
    f = open(filename)
    nextline = f.readline()
    while (lead_str not in nextline and asterisks < 2):
        nextline = f.readline()
        if('**************************' in nextline):
            asterisks += 1
    
    if(asterisks == 2):
        print('FATAL ERROR: Bins not found! for ' + filename)
        exit()

    #then get list of diameters
    diameters = split_numeric(nextline)

    #if need to find average because two lines
    if('lower' in lead_str.lower()):
        nextline = f.readline()
        while(not 'upper' in nextline.lower() and asterisks < 2):
            nextline = f.readline()
            if('**************************' in nextline):
                asterisks += 1
        diameters2 = split_numeric(nextline)
        diameters = (diameters + diameters2)/2

    if(asterisks == 2):
        print('FATAL ERROR: Bins not found! for ' + filename)
        exit()

    return diameters


#create new dataframe to store info about variables
#parameter col: if bin names in long description in variable header (bin_loc == 'IN_VAR'), col = 3
#parameter col: if bin names in variable names (bin_loc == 'IN_NAME'), col = 0
def get_diameters_in_header(varnames, ict,col):
    varDf = get_unit_info(varnames,ict)

    #get numeric values
    diamAr = []
    for row in range(len(varDf)):
        long_desc = varDf.iloc[row][col]
        diam = ''
        #print(long_desc)
        # Use str.translate() with the translation table to remove non-numeric characters
        numeric_string = long_desc.translate(translation_table)
        numeric_string.replace(' ','')
        diamAr.append(numeric_string)

    #strip whitespace
    diamAr = [i.strip() for i in diamAr]

    #add to dataframe
    varDf['Diameters'] = diamAr

    #find which rows contain 'bin'
    mask = varDf['descr_long'].apply(lambda x: 'dNdlogD' not in x)

    #filtered_var_df = varDf.filter(mask)

    diameters = varDf['Diameters'].mask(mask)
    diameters = diameters.dropna()

    return diameters


#function purpose: get list of dN1, dN2, dV1, dV2
#FIXME: look up what these are actually supposed to be?
def get_inputs(dNx, dVx, bin_num):
    bin_num = input('How many bins?')
    bin_num = int(bin_num)

    #loop through dividers
    for i in (range(bin_num-1)):
        #get value for #FIXME dN
        dNi = input('Value for dN' + str(i+1))
        dNi = float(dNi)
        dNx.append(dNi)

        #get value for #FIXME dN
        dVi = input('Value for dV' + str(i+1))
        dVi = float(dVi)
        dVx.append(dVi)

def module_a(filename):

    #get new file name and bin locations
    new_file_name = get_new_name(filename, binlocations)
    bin_loc = get_bin_loc(new_file_name,binlocations)

    #load in ict file to python
    ict = icartt.Dataset(filename)
    df = ict.data[:]

    #get variable names and create dataframe based on this
    varnames = [x for x in ict.variables]
    df = pd.DataFrame(df, columns = varnames)
    varDf = get_unit_info(varnames,ict)

    diam = []
    if bin_loc == 'IN_VAR':
        diam = get_diameters_in_header(varnames,ict,3)
    elif bin_loc == 'IN_NAME':
        diam = get_diameters_in_header(varnames,ict,0)
    elif 'lower' in bin_loc.lower():
        lower = diam = get_diameters_listed(filename,bin_loc)    
    else:
        diam = get_diameters_listed(filename,bin_loc)

    if len(diam) < 1:
        print('FATAL ERROR: No diameters found! for' + filename)
        exit()

    #get variable info
    varDf = get_unit_info(varnames,ict)

    #get variable names that have bins
    mask = varDf['descr_long'].apply(lambda x: 'dNdlogD' in x)

    #if that doesn't work, try looking in other places
    if mask.sum() == 0:
        print('FATAL ERROR: No bin variables found for ' + filename)
        exit()

    #put them into a dataframe
    binnames = varDf['var_name'].loc[mask]
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
    date = datetime.datetime.strptime(date, '%Y-%m-%dT%H:%M:%S.000000000').strftime('%Y_%m_%d')

    #save size_dist_diameter_input to df
    bin_diameters_df[new_file_name+date] = diam

    #make sure properly formatted
    size_dist_diameter_input = pd.Series(diam)
    size_dist_diameter_input.reset_index(drop=True)
    size_dist_diameter_input = size_dist_diameter_input.astype(float)

    #FIXME: Modify for multiple columns
    # If the selected size classes are broader than the measured aerosol sizes 
    #print (d_Nx_list, d_Vx_list,size_dist_diameter_input.iloc[len(size_dist_diameter_input)-1])
    if (d_Nx_list[0] < size_dist_diameter_input.iloc[0]):  #only large particles is available
        N1=0; F_N1=0  
        print("violation of: d_N1 < size_dist_diameter_input[0]")
    if (d_Nx_list[len(d_Nx_list)-1] > size_dist_diameter_input.iloc[len(size_dist_diameter_input)-1]):  #only small particles is available
        N3=0; F_N3=0
        print("violation of: d_N2 > size_dist_diameter_input[len(size_dist_diameter_input)-1]")
    if (d_Vx_list[0] < size_dist_diameter_input.iloc[0]):
        V1=0; F_V1=0
        print("violation of: d_V1 < size_dist_diameter_input[0]")
    if (d_Vx_list[len(d_Vx_list)-1] > size_dist_diameter_input.iloc[len(size_dist_diameter_input)-1]):
        V3=0; F_V=0
        print("violation of: d_V2 > size_dist_diameter_input[len(size_dist_diameter_input)-1]")

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
    export_csv = file_M_1.to_csv(filepath+new_file_name+'.csv', index = None, header=True) 
    bin_diameters_df.to_csv(bin_diameters,index = None, header=True)


#MAIN METHOD
#set directory
fileDirectory = str('C:/Users/cphal/OneDrive/Desktop/Aerosols/Module A/Input Tests/')

#get filenames in folder
path = os.chdir(fileDirectory)
with os.scandir(path) as entries:
    for entry in entries:
        filename = fileDirectory + entry.name
        print('Processing ' + entry.name)
        module_a(filename)