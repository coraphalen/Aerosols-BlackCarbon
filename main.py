from sqlalchemy import create_engine
import pandas as pd
import icartt
import pathlib
import os

fileDirectory = str('c:/Users/cphal/Desktop/NOAA_DATA')
path = os.chdir(fileDirectory)

# load a new dataset from an existing file
wd = pathlib.Path(__file__).parent
ict = icartt.Dataset("c:/Users/cphal/Desktop/" + "FIREXAQ-LARGE-AerosolCloudConc_DC8_20190716_R0.ict")

# list variable names
varnames = [x for x in ict.variables]
#print(varnames)

#get date
date = ict.dateOfCollection
year = date[0]
month = date[1]
day = date[2]


# get all data (NumPy array):
df = ict.data[:]
#print(df)
#print to pandas dataframe
dfpandas = pd.DataFrame(df, columns = varnames)
print(dfpandas)

#define terms of Postgres database
username = 'postgres'
password = 'blackcarbon'
database = 'test_database_2'
host = 'localhost'
port = str(5432)

#create engine to connect to Postgres
engine = create_engine('postgresql://'+username+':'+password+'@'+host+':'+port+'/'+database)
print(engine)

#actually import file
dfpandas.to_sql("FIREXAQ-LARGE-AerosolCloudConc_DC8_20190716_R0.ict", engine)