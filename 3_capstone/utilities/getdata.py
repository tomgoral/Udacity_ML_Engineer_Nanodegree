
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

def getdata(filename, sheetname, index_col= None):
    
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
    
    
    
    ##  chunk 1
    
    categorical_features=['class', 'sub', 'assy', 'head', 'drive', 'thread',
                      'nom', 'point', 'heat', 'lock', 'plate'] 
            
numeric_features       = ["qty","mm"]               # floats or integers
ordinal_features       = []                         # unspecific scale high, medium, low
response_label         = "cost"

feature_columns        = categorical_features + numeric_features + ordinal_features
feature_columns.append(response_label)

features               = df[feature_columns]

print("                original features shape:",features.shape)
features               = df[feature_columns].dropna()                # eliminate rows with any empty features
print("eliminated empty records features shape:",features.shape)


  ##  chunk 2
    
    ranges = {'qty':[10000,100000000],'mm':[0,150], 'cost': [0, 1.0]}

records2delete = []
for key in ranges:
    minValue = ranges[key][0]
    maxValue = ranges[key][1]
    a = (features[features[key] <minValue].index).tolist()
    b = (features[features[key] >maxValue].index).tolist()
    c = a+b
    records2delete.extend(c)
    
features.drop(index=records2delete, inplace = True)
print("    Eliminate Outside Range Features shape:",features.shape)



  ## chunk 3
    
    response = features[response_label]
features.drop([response_label],axis=1, inplace =True)


ohe_columns            = categorical_features + ordinal_features
features               = pd.get_dummies(features, columns=ohe_columns)   # OHE categorical features
print("                     OHE features shape:",features.shape)



  ## chunk4
    
    features.to_excel("features.xlsx",sheet_name='features')
response.to_excel("response.xlsx",sheet_name='response')

    return df
