
# Standard Library Imports

try :
    import os
    import sys
    import pandas as pd
    from timeit import default_timer as timer
    import datetime
except Exception as e:
    print('Some Modules are Missing')
    

# Function

def xl2df(filename, sheetname, index_col= None):
    
# This function receives an xl filename and sheetname.  
# It returns a dataframe

    print()
    print('reading file: {} , sheet: {}, index_col: {}'.format(filename,sheetname,index_col))
    start     = timer()
    df        = pd.read_excel(filename,sheet_name=sheetname, index_col=index_col)
    end       = timer()
    seconds = end-start
    print('loaded File {} in {:,.0f} seconds'.format(filename,seconds))
    print('rows: {}, cols: {}, cells: {}'.format(df.shape[0],df.shape[1],df.shape[0]*df.shape[1]))
    print()

    return df
