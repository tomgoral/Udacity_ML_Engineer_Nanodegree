def dfNoHdr(df):
    # INPUT:  df with a header
    # OUTPUT: dfNH without a header
    
    dict={}
    for column in df.columns:
        dict[column] = df.columns.get_loc(column)
        
    df.rename(columns = dict, inplace = True)
    return df