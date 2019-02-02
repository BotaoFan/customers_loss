#-*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import os
from datetime import datetime

import tools

def load_data(data_dir_path):
    '''
    Load raw data and all data files should be in data_dir_path
    We need .csv file below in the data_dir_path:
        cust_info.csv
        yyb_area.csv
        cust_trade_kc_columns.csv
        cust_trade_kc_1.csv
        cust_trade_kc_2.csv
        cust_trade_kc_3.csv
        cust_trade_kc_4.csv
        cust_trade_kc_5.csv
        cust_trade_rzq_1.csv
        cust_trade_rzq_2.csv
        cust_trade_rzq_3.csv
    :param data_dir_path:String
    :return:DataFrame,DataFrame
    '''
    #Load cust_info
    cust_info=pd.read_csv(data_dir_path+'cust_info.csv',encoding='GBK')
    yyb_area=pd.read_csv(data_dir_path+'yyb_area.csv',encoding='GBK')
    cust_info=pd.merge(cust_info,yyb_area,left_on='yyb',right_on='yyb',how='left')
    del yyb_area
    #Load cust_trade_kc
    cust_trade_columns=pd.read_csv(data_dir_path+'cust_trade_kc_columns.csv')
    cust_trade_columns_dict=dict()
    for i in cust_trade_columns.index:
        cust_trade_columns_dict[cust_trade_columns.loc[i,'colid']-1]=cust_trade_columns.loc[i,'name']
    del cust_trade_columns
    cust_trade_1=pd.read_csv(data_dir_path+'cust_trade_kc_1.csv',header=None,index_col=None)
    cust_trade_1.rename(columns=cust_trade_columns_dict,inplace=True)
    cust_trade_2=pd.read_csv(data_dir_path+'cust_trade_kc_2.csv',header=None,index_col=None)
    cust_trade_2.rename(columns=cust_trade_columns_dict,inplace=True)
    cust_trade_1=pd.concat([cust_trade_1,cust_trade_2])
    del cust_trade_2
    cust_trade_3=pd.read_csv(data_dir_path+'cust_trade_kc_3.csv',header=None,index_col=None)
    cust_trade_3.rename(columns=cust_trade_columns_dict,inplace=True)
    cust_trade_1=pd.concat([cust_trade_1,cust_trade_3])
    del cust_trade_3
    cust_trade_4=pd.read_csv(data_dir_path+'cust_trade_kc_4.csv',header=None,index_col=None)
    cust_trade_4.rename(columns=cust_trade_columns_dict,inplace=True)
    cust_trade_1=pd.concat([cust_trade_1,cust_trade_4])
    del cust_trade_4
    cust_trade_5=pd.read_csv(data_dir_path+'cust_trade_kc_5.csv',header=None,index_col=None)
    cust_trade_5.rename(columns=cust_trade_columns_dict,inplace=True)
    cust_trade_1=pd.concat([cust_trade_1,cust_trade_5])
    del cust_trade_5
    #Load cust_trade_rzq
    cust_trade_rzq_1=pd.read_csv(data_dir_path+'cust_trade_rzq_1.csv',header=None,index_col=None)
    cust_trade_rzq_1.rename(columns=cust_trade_columns_dict,inplace=True)
    cust_trade_1=pd.concat([cust_trade_1,cust_trade_rzq_1])
    del cust_trade_rzq_1
    cust_trade_rzq_2=pd.read_csv(data_dir_path+'cust_trade_rzq_2.csv',header=None,index_col=None)
    cust_trade_rzq_2.rename(columns=cust_trade_columns_dict,inplace=True)
    cust_trade_1=pd.concat([cust_trade_1,cust_trade_rzq_2])
    del cust_trade_rzq_2
    cust_trade_rzq_3=pd.read_csv(data_dir_path+'cust_trade_rzq_3.csv',header=None,index_col=None)
    cust_trade_rzq_3.rename(columns=cust_trade_columns_dict,inplace=True)
    #Merge cust_trade_rzq and cust_trade_kc into cust_trade
    cust_trade_1=pd.concat([cust_trade_1,cust_trade_rzq_3])
    del cust_trade_rzq_3
    cust_trade=cust_trade_1
    del cust_trade_1
    cust_trade.reset_index(drop=True,inplace=True)
    return cust_info,cust_trade





if __name__=='__main__':
    pass