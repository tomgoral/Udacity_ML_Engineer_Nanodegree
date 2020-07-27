'''
Functions in this Library:

segment(dataframe)
column_metrics(column, column_name)
hist_plot(data, num_bins, labels, xy_max_min)
mse(actual, preds)
mae(actual, preds)
print_metrics(y_true, preds, model_name=None)
r2(actual, preds)

'''

# Standard Library Imports

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Functions

def column_metrics(column, column_name):
    print()
    print(column_name, 'METRICS')
    print('==================')
    print('MAXIMUM : ', column.max())
    print('MEDIAN  : ', column.median())
    print('MEAN    : ', column.mean())
    print('MIN     : ', column.min())
    print('BLANKS  : ', column.isnull().sum())
    return

def hist_plot(data, num_bins, labels, xy_max_min):
    # INPUT: data = input series, num_bins = data segements
    # OUTPUT :  plot the data

    import statistics
    mu    = statistics.mean(data)
    sigma = statistics.stdev(data)


    ymin = xy_max_min['ymin']
    ymax = xy_max_min['ymax']
    xmin = xy_max_min['xmin']
    xmax = xy_max_min['xmax']

    ylabel = labels['ylabel']
    xlabel = labels['xlabel']
    title  = labels['title']


    chart,axes = plt.subplots(1,1)
    chart.set_figheight(8)
    chart.set_figwidth(16)

    axes.hist(data, num_bins, color='olive', alpha =0.75, edgecolor='k',
              linestyle='dashed', linewidth=2, density=False)
    axes.set_xlim(xmin, xmax)
    axes.set_ylim(ymin, ymax)
    axes.set_ylabel(ylabel,fontsize=24)
    axes.set_xlabel(xlabel,fontsize=24)
    axes.set_title(title,fontsize=32, fontweight='bold')
    return

def mae(actual, preds):
    '''
    INPUT:
    actual - numpy array or pd series of actual y values
    preds - numpy array or pd series of predicted y values
    OUTPUT:
    returns the mean absolute error as a float
    '''
    return np.sum(np.abs(actual-preds))/len(actual)

def mse(actual, preds):
    '''
    INPUT:
    actual - numpy array or pd series of actual y values
    preds - numpy array or pd series of predicted y values
    OUTPUT:
    returns the mean squared error as a float
    '''
    return np.sum((actual-preds)**2)/len(actual)

def print_metrics(y_true, preds, model_name=None):
    '''
    INPUT:
    y_true - the y values that are actually true in the dataset (numpy array or pandas series)
    preds - the predictions for those values from some model (numpy array or pandas series)
    model_name - (str - optional) a name associated with the model if you would like to add it to the print statements
    OUTPUT:
    None - prints the mse, mae, r2
    '''
    if model_name == None:
        print('Mean Squared Error: ', format(mean_squared_error(y_true, preds)))
        print('Mean Absolute Error: ', format(mean_absolute_error(y_true, preds)))
        print('R2 Score: ', format(r2_score(y_true, preds)))
        print('\n\n')

    else:
        print('Mean Squared Error ' + model_name + ' :' , format(mean_squared_error(y_true, preds)))
        print('Mean Absolute Error ' + model_name + ' :', format(mean_absolute_error(y_true, preds)))
        print('R2 Score ' + model_name + ' :', format(r2_score(y_true, preds)))
        print('\n\n')
    return

def r2(actual, preds):
    '''
    INPUT:
    actual - numpy array or pd series of actual y values
    preds - numpy array or pd series of predicted y values
    OUTPUT:
    returns the r-squared score as a float
    '''
    sse = np.sum((actual-preds)**2)
    sst = np.sum((actual-np.mean(actual))**2)
    return 1 - sse/sst

def segment(dataframe):

    print ('\n\nBOOLEAN COLUMNS : ', end ='')
    for each in  dataframe.columns:
        if str(dataframe[each].dtype) == 'bool':
            print (each,', ',end='')

    print ('\n\nFEATURE COLUMNS : ', end ='')
    for each in  dataframe.columns:
        if str(dataframe[each].dtype) == 'object':
            print (each,', ',end='')

    print ('\n\nFLOAT  COLUMNS : ', end ='')
    for each in  dataframe.columns:
        if str(dataframe[each].dtype) == 'float64':
            print (each,', ',end='')

    print ('\n\nINTEGER COLUMNS : ', end ='')
    for each in  dataframe.columns:
        if str(dataframe[each].dtype) == 'int64':
            print (each,', ',end='')

    return

def xl2df(filename, sheetname, index_col= None):
    import pandas as pd
    from timeit import default_timer as timer
    import datetime


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
