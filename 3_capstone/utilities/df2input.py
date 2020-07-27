

def df2input(df,categories=[],numerical=[],ordinal=[],response=None):
    import os
    import sys
    import numpy as np
    import pandas as pd

    #  INPUT:
    #  df          : the input dataframe
    #  categories  : categorical features of df
    #  numerical   : numerical float or int features of df
    #  ordinal     : ordinal features (unspecific scale: high, medium, low)
    #  response    : response


    #  OUTPUT:
    #  df_features    : df prepped for ML
    #  array_features : numpy array version of df_features
    #  series_response: pandas series of the response


    feature_columns = categories + numerical + ordinal
    feature_columns.append(response)
    df_features     = df[feature_columns]
    print("                original features shape:",df_features.shape)
    df_features     = df[feature_columns].dropna()
    print("eliminated empty records features shape:",df_features.shape)

    series_response = df_features[response]
    df_features.drop([response],axis=1, inplace =True)
    ohe_columns     = categories + ordinal
    df_features     = pd.get_dummies(df_features, columns=ohe_columns)
    print("                     OHE features shape:",df_features.shape)
    array_features  = df_features.to_numpy()


    return df_features, array_features, series_response
