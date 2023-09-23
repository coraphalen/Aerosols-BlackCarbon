# %%
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

# %%
filebeginning = 'C:/Users/cphal/OneDrive/Desktop/Aerosols/Module A/Success/'
filename = filebeginning + 'DC3-LARGE-LAS-PSL_DC8_20120607_R2.ict'

# %%
##for one file (filename) at top
#get new file name and bin locations
new_file_name = get_new_name(filename, binlocations)
#bin_loc = get_bin_loc(new_file_name,binlocations)

#load in ict file to python
ict = icartt.Dataset(filename)
df = ict.data[:]

#get variable names and create dataframe based on this
varnames = [x for x in ict.variables]
df = pd.DataFrame(df, columns = varnames)
varDf = get_unit_info(varnames,ict)

varDf

# %%
#get diameters
row = 'var_name'
start_column = 5
end_column = len(varDf)
diam_list = varDf[row][start_column:end_column]

diams = [x.translate(translation_table) for x in diam_list]
diams = [float(x) for x in diams]

#append to diameter txt file
diameter_file = "C:/Users/cphal/OneDrive/Desktop/Aerosols/Module A/NASA diameters.csv"
# Append-adds at last
file1 = open(diameter_file, "a")  # append mode
file1.write(new_file_name + "," + row + "," + str(start_column) +","+ str(end_column) + "," + str(diams) + "\n")
file1.close()


