import pandas as pd
import numpy as np
import icartt

#function: checking to see if a value has already been saved to a file
#else prompt user to enter new abbreviation
#parameters: table_file 
def check_table(table_file, given_column, given_object, given_value, checking_object, checking_column):
    #open table file
    table = pd.read_csv(table_file)

    #initialize a variable to be the new value
    new_val = ''

    #check to see if file beginning = current abbreviation
    file_contains = table[given_column] == given_value
   
   
    #if there are no matches (only True)
    if(pd.Series(file_contains).any() == False):
        new_val = input(given_object + ' ' + given_value + ' not found. Please enter a new ' + checking_object + ' for ' + given_object + " " + given_value + ' :')
        #check to see if that abbreviation already exists
        unique = table[checking_column] == new_val
        #if so, prompt to enter a new abbreviation until a unique one has been found
        while(pd.Series(unique).any() == True):
            print('Sorry, that ' + checking_object + ' already exists.')
            new_val = input('Please enter a ' + checking_object + 'for ' + given_object + ' ' + given_value + ' :')
            unique = table[checking_column] == new_val
        #add new row to abbreviations and export abbreviations
        table.loc[(len(table.index))] = [given_value, new_val]
        table.to_csv(table_file)
    #if so (one False), file abbreviation is the one from the csv file
    elif(pd.Series(file_contains).any() == True):
        true_index_list = list(np.where(file_contains)[0])
        index = int(true_index_list[0])
        new_val = table.iloc[index, 2]

    return new_val

#function purpose: load file into dataframe given directory, campaign, and file name
def load_file_into_df(filename):
    #load in ict file to python
    ict = icartt.Dataset(filename)

    # get all data (NumPy array):
    df = ict.data[:]

    # list variable names
    varnames = [x for x in ict.variables]

    #print to pandas dataframe
    dfpandas = pd.DataFrame(df, columns = varnames)

    #return dataframe
    return dfpandas