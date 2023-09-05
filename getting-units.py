# %%
import icartt
import pandas as pd
import numpy as np
import datetime
from math import pi

filename = 'C:/Users/cphal/OneDrive/Desktop/MAC prediction model/FIREXAQ-LARGE-CDP_DC8_20190722_R0.ict'

 # Initialize a translation table to remove non-numeric characters
translation_table = str.maketrans('','','abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-/:;<=>?@[\\]^_`{|}~\t')

# %%
#load in ict file to python
ict = icartt.Dataset(filename)
df = ict.data[:]

#get variable names and create dataframe based on this
varnames = [x for x in ict.variables]
df = pd.DataFrame(df, columns = varnames)

# %%
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
#alternative method of getting diameters

def get_diameters_listed(filename):

    #read lines until line contains 'diameters'
    f = open(filename)
    nextline = f.readline()
    while ('diameter' not in nextline):
        nextline = f.readline()
    #then get list of diameters
    diameters = nextline.split(',')
    firstline = diameters[0].split()
    first_diam = firstline[len(firstline) - 1]
    diameters[0] = first_diam
    del diameters['\n']

    return diameters

# %%
#create new dataframe to store info about variables
def get_diameters_in_header(varnames, ict):
    varDf = get_unit_info(varnames,ict)

    #get numeric values
    diamAr = []
    for row in range(len(varDf)):
        long_desc = varDf.iloc[row][3]
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
    mask = varDf['descr_long'].apply(lambda x: 'bin' not in x)

    #filtered_var_df = varDf.filter(mask)

    diameters = varDf['Diameters'].mask(mask)
    diameters = diameters.dropna()

    return diameters


# %%
#function purpose: get list of dN1, dN2, dV1, dV2
#FIXME: look up what these are actually supposed to be?

def get_inputs(dNx, dVx):
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

# %%
#get variable info
varDf = get_unit_info(varnames,ict)

#get variable names that have bins
mask = varDf['var_name'].apply(lambda x: 'bin' in x.lower())

#put them into a dataframe
binnames = varDf['var_name'].loc[mask]
size_dist_input_original = df[binnames]

#transpose size_dist_input to get it to work with Module A
size_dist_input = pd.DataFrame.transpose(size_dist_input_original)

#create new vector to hold column names
ser=pd.Series(range(np.size(size_dist_input,1)))
#assign this to be the columns
size_dist_input.rename(columns = ser)

# %%
#create list to hold input values
d_Nx_list = []
d_Vx_list = []
get_inputs(d_Nx_list, d_Vx_list)

# %%
d_Vx_list

# %%
date_time = ict.times
datetimedf = pd.DataFrame(date_time, columns= ['datetime'])
num1 = str(date_time[0])
dateparse = lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.000000000')

# %%
#FIXME: check either method based on file beginning; create system for this
size_dist_diameter_input = pd.Series(get_diameters_in_header(varnames,ict))
size_dist_diameter_input.reset_index(drop=True)
size_dist_diameter_input = size_dist_diameter_input.astype(float)

# %%
# If the selected size classes are broader than the measured aerosol sizes 
print (d_Nx_list, d_Vx_list,size_dist_diameter_input.iloc[len(size_dist_diameter_input)-1])
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

# %%
#calculate dVdlogdP using dNdlogdP
diameter_power = np.power(size_dist_diameter_input, 3)
diameter_power = diameter_power.to_numpy()
size_dist_volume = pi/6*size_dist_input.mul(diameter_power,axis='index')


