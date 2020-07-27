# Standard Library Imports

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns



def mae(actual, preds):
    
    #INPUT:
    #actual - numpy array or pd series of actual y values
    #preds - numpy array or pd series of predicted y values
    #OUTPUT:
    #returns the mean absolute error as a float
    
    return np.sum(np.abs(actual-preds))/len(actual)

def mse(actual, preds):
    
    #INPUT:
    #actual - numpy array or pd series of actual y values
    #preds  - numpy array or pd series of predicted y values
    #OUTPUT:
    #returns the mean squared error as a float
    
    return np.sum((actual-preds)**2)/len(actual)

def print_metrics(y_true, preds, model_name=None):
    
    #INPUT:
    #y_true - the y values that are actually true in the dataset (numpy array or pandas series)
    #preds  - the predictions for those values from some model (numpy array or pandas series)
    #model_name - (str - optional) a name associated with the model if you would like to add it to the print statements
    #OUTPUT:
    #None - prints the mse, mae, r2
    
    if model_name == None:
        print(mse(y_true, preds))
        print(mae(y_true, preds))
        print(r2(y_true, preds))
        print('\n\n')

    else:
        print(mse(y_true, preds))
        print(mae(y_true, preds))
        print(r2(y_true, preds))
        print('\n\n')

def r2(actual, preds):
    '''
    INPUT:
    actual - numpy array or pd series of actual y values
    preds  - numpy array or pd series of predicted y values
    OUTPUT:
    returns the r-squared score as a float
    '''
    sse = np.sum((actual-preds)**2)
    sst = np.sum((actual-np.mean(actual))**2)
    return 1 - sse/sst