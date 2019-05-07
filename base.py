import pandas as pd
import numpy as np

def check_col_exist(df,col_name):
    '''
    Check is col_name is a name of df's columns
    :param df: DataFrame
    :param col_name: str
    :return: None
    '''
    if col_name in df.columns:
        return None
    else:
        raise KeyError("Data doesn't contain column named %s" % col_name)

def check_dataframe(df):
    '''
    Check df is the pd.DataFrame
    :param df:DataFrame
    :return:None
    '''
    if isinstance(df,pd.DataFrame):
        return None
    else:
        pd_type=type(df)
        raise TypeError('Df should be DataFrame instead of %s' % pd_type)


