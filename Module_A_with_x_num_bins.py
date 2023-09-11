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
#3 bins , d_Nx = [10.5, 30] , d_Vx = [25,40]

# %%
d_Nx_list

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

# %%
size_dist_volume.head()

# %%
#calculate dVdlogdP using dNdlogdP TEST PURPOSES ONLY
#size_dist_volume_slow = pd.DataFrame().reindex_like(size_dist_input)
#for m in range(np.size(size_dist_volume_slow,1)):
    #for n in range(len(size_dist_diameter_input)):
        #size_dist_volume_slow.values[n,m]=pi/6* np.power(size_dist_diameter_input, 3).values[n]*size_dist_input.values[n,m]

# %%
#prepare for trapz calculation
size_dist_input[np.isnan(size_dist_input)] = 0 
size_dist_volume[np.isnan(size_dist_volume)] = 0 

ln_size_dist_diameter_input=np.log(size_dist_diameter_input)

# %%
#function purpose: find nearest value (d_N1, d_N2, d_V1, d_V2) in numpy array (instrument measured diameters)
def find_nearest(array, value):
    n = [abs(i-value) for i in array]
    idx = n.index(min(n))
    return idx

# %%
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

# %%
#create empty dataframes to hold num_bins: should be (num_bins) x (num_columns in size_dist)
num_bins = len(d_Nx_list) + 1

N_df = pd.DataFrame(index = range(np.size(size_dist_input,1)), columns = range(num_bins))
F_N_df = pd.DataFrame(index = range(np.size(size_dist_input,1)), columns = range(num_bins))
V_df = pd.DataFrame(index = range(np.size(size_dist_input,1)), columns = range(num_bins))
F_V_df = pd.DataFrame(index = range(np.size(size_dist_input,1)), columns = range(num_bins))

# %%
#add beginning and ending indexes for the divisions
d_Nx_nearest.insert(0,0)
d_Nx_nearest.append(len(size_dist_diameter_input))

d_Vx_nearest.insert(0,0)
d_Vx_nearest.append(len(size_dist_diameter_input))

# %%
#for each bin, make a smaller dataframe with the diameter bins desired. Then apply trapz calculation by row
#and add result to dataframe in adjacent bin
for i in range(num_bins):
    small_df = size_dist_input.iloc[(d_Nx_nearest[i]):(d_Nx_nearest[i+1]),:] 
    N_df.iloc[:,i] = np.trapz(small_df, x=ln_size_dist_diameter_input[(d_Nx_nearest[i]):(d_Nx_nearest[i+1])], axis=0)

for i in range(num_bins):
    small_df = size_dist_input.iloc[(d_Vx_nearest[i]):(d_Vx_nearest[i+1]),:] 
    V_df.iloc[:,i] = np.trapz(small_df, x=ln_size_dist_diameter_input[(d_Vx_nearest[i]):(d_Vx_nearest[i+1])], axis=0)

# %%
V_df

# %%
#calculate areaXY_number and volumeXY_number
areaXY_number = pd.Series(np.trapz(size_dist_input, x=ln_size_dist_diameter_input, axis=0))
volumeXY_number = pd.Series(np.trapz(size_dist_volume, x=ln_size_dist_diameter_input, axis=0))

# %%
#FIXME: question: can we replace 0 with np.nan?
areaXY_number.replace(0, np.nan, inplace=True)
volumeXY_number.replace(0, np.nan, inplace=True)

# %%
#divide N by areaXY for f_N dataframe
for i in range(num_bins):
    F_N_df.iloc[:,i] = N_df.iloc[:,i] / areaXY_number

for i in range(num_bins):
    F_V_df.iloc[:,i] = V_df.iloc[:,i] / volumeXY_number

# %%
#put in big dataframe!
file_M_1 = pd.DataFrame() 
file_M_1['datetime'] = datetime
for i in range(num_bins):
    file_M_1['N'+str(i)] = N_df.iloc[:,i]
    file_M_1['N_Total'] = areaXY_number
    file_M_1['F_N'+str(i)] = F_N_df.iloc[:,i]
    file_M_1['V'+str(i)] = V_df.iloc[:,i]
    file_M_1['V_Total'] = volumeXY_number
    file_M_1['F_V'+str(i)] = F_V_df.iloc[:,i]

export_csv = file_M_1.to_csv ('Output_Module_1.csv', index = None, header=True) 

# %%
##ORIGINAL CODE
#loops through columns (all 20000+!)
#each column = 1 measurement for the bins in the instrument
N1 = np.zeros(np.size(size_dist_input,1)); N2 = np.zeros(np.size(size_dist_input,1)); N3 = np.zeros(np.size(size_dist_input,1))
F_N1 = np.zeros(np.size(size_dist_input,1)); F_N2 = np.zeros(np.size(size_dist_input,1)); F_N3 = np.zeros(np.size(size_dist_input,1))   
areaXY_number = np.zeros(np.size(size_dist_input,1))  
V1 = np.zeros(np.size(size_dist_input,1)); V2 = np.zeros(np.size(size_dist_input,1)); V3 = np.zeros(np.size(size_dist_input,1))
F_V1 = np.zeros(np.size(size_dist_input,1)); F_V2 = np.zeros(np.size(size_dist_input,1)); F_V3 = np.zeros(np.size(size_dist_input,1))   
areaXY_volume = np.zeros(np.size(size_dist_input,1)) 

for m in range(np.size(size_dist_input,1)):

        #for num_bins, find nearest index in size_dist_diameter input -> this would be the same across columns
        d_N1_nearest = find_nearest(size_dist_diameter_input,10.5)
        d_N2_nearest = find_nearest(size_dist_diameter_input,30)
        d_V1_nearest = find_nearest(size_dist_diameter_input,25)
        d_V2_nearest = find_nearest(size_dist_diameter_input,40)

        #perform composite trapezoidal calculation on every element in a given column
        areaXY_number[m] = np.trapz(size_dist_input.values[:,m], x=ln_size_dist_diameter_input)
        #perform composite trapezoidal calculation on every row from smallest to diameter closest to d_N1 in a given column
        N1[m] = np.trapz(size_dist_input.values[0:d_N1_nearest,m], x=ln_size_dist_diameter_input[0:d_N1_nearest])
        #perform composite trapezoidal calculation on every row from diameter closest to dN1 to dN2 in a given column
        N2[m] = np.trapz(size_dist_input.values[d_N1_nearest+1:d_N2_nearest,m], x=ln_size_dist_diameter_input[d_N1_nearest+1:d_N2_nearest])    
        #perform composite trapezoidal calculation on every row from diameter closest to dN2 to end in a given column
        N3[m] = np.trapz(size_dist_input.values[d_N2_nearest+1:,m], x=ln_size_dist_diameter_input[d_N2_nearest+1:])

        #for num_bins, divide each column by its corresponding area_XY number
        F_N1[m] = N1[m]/areaXY_number[m]
        F_N2[m] = N2[m]/areaXY_number[m]       
        F_N3[m] = N3[m]/areaXY_number[m]


        #do the same for volume
        areaXY_volume[m] = np.trapz(size_dist_volume.values[:,m], x=ln_size_dist_diameter_input)
        V1[m] = np.trapz(size_dist_volume.values[0:d_V1_nearest,m], x=ln_size_dist_diameter_input[0:d_V1_nearest])
        V2[m] = np.trapz(size_dist_volume.values[d_V1_nearest+1:d_V2_nearest,m], x=ln_size_dist_diameter_input[d_V1_nearest+1:d_V2_nearest])    
        V3[m] = np.trapz(size_dist_volume.values[d_V2_nearest+1:,m], x=ln_size_dist_diameter_input[d_V2_nearest+1:])
        
        F_V1[m] = V1[m]/areaXY_volume[m]
        F_V2[m] = V2[m]/areaXY_volume[m]       
        F_V3[m] = V3[m]/areaXY_volume[m]

# %%
file_M_1 = pd.DataFrame() 
file_M_1['datetime'] = datetimedf
file_M_1['N1'] = N1; file_M_1['N2'] = N2 ; file_M_1['N3'] = N3 ; file_M_1['N_total'] = areaXY_number 
file_M_1['F_N1'] = F_N1; file_M_1['F_N2'] = F_N2 ; file_M_1['F_N3'] = F_N3 
file_M_1['V1'] = V1; file_M_1['V2'] = V2 ; file_M_1['V3'] = V3 ; file_M_1['V_total'] = areaXY_volume 
file_M_1['F_V1'] = F_V1; file_M_1['F_V2'] = F_V2 ; file_M_1['F_V3'] = F_V3 


