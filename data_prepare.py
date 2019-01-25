#-*- coding:utf-8 -*-
import os
import warnings

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


import tools

pd.set_option('display.width',1000)
pd.set_option('display.max_columns',1000)
warnings.filterwarnings('ignore')



if __name__=='__main__':
    #Load raw data
    data_dir_name='raw_data_customers_loss'
    data_dir_path=os.getcwd()+'/../'+data_dir_name+'/'
    cust_info=pd.read_csv(data_dir_path+'cust_info.csv',encoding='GBK')
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
    cust_trade_1=pd.concat([cust_trade_1,cust_trade_rzq_3])
    del cust_trade_rzq_3

    cust_trade=cust_trade_1
    del cust_trade_1

    #Data clean
    cust_info.loc[cust_info['khrq'].astype(np.int32).astype(str).apply(lambda x: len(x) < 8), 'khrq']='18000101'
    cust_info['khrq_date']=pd.to_datetime(cust_info['khrq'].astype(np.int32).astype(str),format='%Y%m%d')
    cust_info.loc[cust_info['khrq_date']==datetime(1800,01,01),'khrq_date']=np.nan
    cust_info.loc[cust_info['khrq']=='18000101','khrq']=np.nan

    cust_info.loc[cust_info['birthday'].isnull(),'birthday']='18000101'
    cust_info.loc[cust_info['birthday'].astype(np.int32).astype(str).apply(lambda x: len(x) < 8),'birthday']='18000101'
    cust_info['birthday']=cust_info['birthday'].astype(np.int32).astype(str)
    cust_info.loc[~(cust_info['birthday'].apply(lambda x: tools.check_date_str_right(x))),'birthday']='18000101'
    cust_info['birthday_datetime']=pd.to_datetime(cust_info['birthday'].astype(np.int32).astype(str),format='%Y%m%d')
    cust_info.loc[cust_info['birthday_datetime']==datetime(1800,01,01),'birthday_datetime']=np.nan
    cust_info.loc[cust_info['birthday']=='18000101','birthday']=np.nan
    cust_info['age']=2018-cust_info['birthday_datetime'].apply(lambda x:x.year)


    #Convert data type
    dc=tools.DtypeConvert()
    trade_to_int32_list=['yyb','sno','operdate','cleardate','bizdate','moneytype','orgid','brhid','orderdate','ordertime','matchtimes','matchtime']
    trade_to_float32_list=['fundeffect','fundbal','orderprice','matchamt','matchprice','fee_jsxf','fee_sxf']
    cust_trade=dc.to_int32_group(cust_trade,trade_to_int32_list)
    cust_trade=dc.to_float32_group(cust_trade,trade_to_float32_list)
    del trade_to_int32_list,trade_to_float32_list

    #Detect missing data
    dd=tools.DataDetect()
    dd.count_df_na(cust_info)

    #Detect start_jyzc
    (cust_info['start_jyzc']/1e4).describe().astype(int)




