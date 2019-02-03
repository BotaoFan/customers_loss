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


def clean_data(cust_info,cust_trade):
    #cust_info Data Clean
    #convert khrq from number to datetime
    cust_info.loc[cust_info['khrq'].astype(np.int32).astype(str).apply(lambda x: len(x) < 8), 'khrq']='18000101'
    cust_info['khrq_date']=pd.to_datetime(cust_info['khrq'].astype(np.int32).astype(str),format='%Y%m%d')
    cust_info.loc[cust_info['khrq_date']==datetime(1800,01,01),'khrq_date']=np.nan
    cust_info.loc[cust_info['khrq']=='18000101','khrq']=np.nan
    #convert birthday from number to datetime
    cust_info.loc[cust_info['birthday'].isnull(),'birthday']='18000101'
    cust_info.loc[cust_info['birthday'].astype(np.int32).astype(str).apply(lambda x: len(x) < 8),'birthday']='18000101'
    cust_info['birthday']=cust_info['birthday'].astype(np.int32).astype(str)
    cust_info.loc[~(cust_info['birthday'].apply(lambda x: tools.check_date_str_right(x))),'birthday']='18000101'
    cust_info['birthday_datetime']=pd.to_datetime(cust_info['birthday'].astype(np.int32).astype(str),format='%Y%m%d')
    cust_info.loc[cust_info['birthday_datetime']==datetime(1800,01,01),'birthday_datetime']=np.nan
    cust_info.loc[cust_info['birthday']=='18000101','birthday']=np.nan
    #Add some attributes to cust_info
    cust_info['start_jyzc_ln']=np.log1p(cust_info['start_jyzc'])
    cust_info['end_jyzc_ln']=np.log1p(cust_info['end_jyzc'])
    cust_info['age']=2018-cust_info['birthday_datetime'].apply(lambda x:x.year)
    cust_info['kh_age']=2018-cust_info['khrq_date'].apply(lambda x: x.year)
    ##Clear cust_trade date
    dc=tools.DtypeConvert()
    trade_to_int32_list=['yyb','sno','operdate','cleardate','bizdate','moneytype','orgid','brhid','orderdate','ordertime','matchtimes','matchtime']
    trade_to_float32_list=['fundeffect','fundbal','orderprice','matchamt','matchprice','fee_jsxf','fee_sxf']
    cust_trade=dc.to_int32_group(cust_trade,trade_to_int32_list)
    cust_trade=dc.to_float32_group(cust_trade,trade_to_float32_list)
    del trade_to_int32_list,trade_to_float32_list

    cust_trade['bizdate_date']=pd.to_datetime(cust_trade['bizdate'].astype(str))
    cust_trade['order_minute']=(cust_trade['ordertime'].astype(np.int32).astype(str).str[-6:-4]).apply(lambda x: 0 if x=='' else int(x))
    cust_trade['order_hour']=(cust_trade['ordertime'].astype(np.int32).astype(str).str[:-6]).apply(lambda x: 0 if x=='' else int(x))
    cust_trade['match_minute']=(cust_trade['matchtime'].astype(np.int32).astype(str).str[-6:-4]).apply(lambda x: 0 if x=='' else int(x))
    cust_trade['match_hour']=(cust_trade['matchtime'].astype(np.int32).astype(str).str[:-6]).apply(lambda x: 0 if x=='' else int(x))

    cust_trade.loc[cust_trade['ordertime']==0,'ordertime']=np.nan
    cust_trade.loc[cust_trade['matchtime']==0,'matchtime']=np.nan
    cust_trade.loc[cust_trade['ordertime'].isnull(),['order_hour','order_minute']]=np.nan
    cust_trade.loc[cust_trade['matchtime'].isnull(),['match_hour','match_minute']]=np.nan
    cust_trade.drop(cust_trade.loc[cust_trade['matchamt']<0].index,inplace=True)
    #Add biz_weekday
    cust_trade['biz_weekday']=cust_trade['bizdate_date'].apply(lambda x: x.weekday())
    cust_trade['biz_week']=cust_trade['bizdate_date'].apply(lambda x: x.week)
    #If matchamt>0 then customer sell stocks(buy==0) else customer buy stocks(buy==1)
    cust_trade['buy']=1
    cust_trade.loc[cust_trade['fundeffect']>0,'buy']=0
    cust_trade['stktype_cate']='other'
    cust_trade.loc[cust_trade['stktype']==' ','stktype_cate']='fund'
    cust_trade.loc[cust_trade['stktype']=='0','stktype_cate']='stock'
    cust_trade.loc[cust_trade['stktype']=='G','stktype_cate']='G'
    cust_trade.loc[cust_trade['stktype']=='L','stktype_cate']='L'
    cust_trade.loc[cust_trade['stktype']=='E','stktype_cate']='E'

def generate_features(cust_info,cust_trade):
    #cust_info features
    cust_features=cust_info.loc[:,['khh','kh_age','age','start_jyzc','chg_rate','start_jyzc_ln','sec','area']]
    cust_features['y']=0
    cust_features.loc[cust_features['chg_rate']<=-0.5,'y']=1
    cust_features.set_index('khh',drop=False,inplace=True)
    all_features=cust_features.copy()
    #Trade features
    #Buy and sell every week every customers(all stock type)
    trade_feats_group=cust_trade.groupby(['custid','biz_week','buy'])
    trade_feats=trade_feats_group['sno'].count()
    trade_feats_unstack=(trade_feats.unstack()).unstack()
    trade_feats_unstack.fillna(0,inplace=True)
    all_features=pd.merge(all_features,trade_feats_unstack,left_index=True,right_index=True,suffixes=('','_trade_count'),how='left')
    #Buy and sell every week every customers(all stock type) matchamt
    trade_feats=trade_feats_group['matchamt'].sum()
    trade_feats_unstack=(trade_feats.unstack()).unstack()
    trade_feats_unstack.fillna(0,inplace=True)
    all_features=pd.merge(all_features,trade_feats_unstack,left_index=True,right_index=True,suffixes=('','_trade_amt'),how='left')
    #['fund','stock','G','L','E','other'] trade amount count
    trade_feats_group=cust_trade.groupby(['custid','buy','stktype_cate'])
    trade_feats=trade_feats_group['sno'].count()
    trade_feats_unstack=(trade_feats.unstack()).unstack()
    trade_feats_unstack.fillna(0,inplace=True)
    all_features=pd.merge(all_features,trade_feats_unstack,left_index=True,right_index=True,suffixes=('','_trade_count'),how='left')
    #['fund','stock','G','L','E','other'] trade amount matchamt
    trade_feats=trade_feats_group['matchamt'].count()
    trade_feats_unstack=(trade_feats.unstack()).unstack()
    trade_feats_unstack.fillna(0,inplace=True)
    all_features=pd.merge(all_features,trade_feats_unstack,left_index=True,right_index=True,suffixes=('','trade_amt'),how='left')
    #fund out count trend
    stock_trade=cust_trade.loc[(cust_trade['stktype_cate']=='fund') | (cust_trade['buy']==1),:]
    stock_trade_group=stock_trade.groupby(['custid','biz_week'])
    trade_feats=stock_trade_group['sno'].count()
    trade_feats_unstack=trade_feats.unstack()
    trade_feats_unstack.fillna(0,inplace=True)
    reg_x=range(1,15)
    reg_result=trade_feats_unstack.apply(lambda y: np.polyfit(reg_x,y.values,3),axis=1)
    reg_beta_3=reg_result.apply(lambda x:x[0])
    reg_beta_2=reg_result.apply(lambda x:x[1])
    reg_beta_1=reg_result.apply(lambda x:x[2])
    reg_beta_0=reg_result.apply(lambda x:x[3])
    reg_result_beta=pd.concat([reg_beta_3,reg_beta_2,reg_beta_1,reg_beta_0],axis=1)
    reg_result_beta.columns=['fund_out_trend_3','fund_out_trend_2','fund_out_trend_1','fund_out_trend_0']
    all_features=pd.merge(all_features,reg_result_beta,left_index=True,right_index=True,how='left')
    # fund in count trend
    stock_trade = cust_trade.loc[(cust_trade['stktype_cate'] == 'fund') | (cust_trade['buy'] == 0), :]
    stock_trade_group = stock_trade.groupby(['custid', 'biz_week'])
    trade_feats=stock_trade_group['sno'].count()
    trade_feats_unstack = trade_feats.unstack()
    trade_feats_unstack.fillna(0, inplace=True)
    reg_x = range(1, 15)
    reg_result = trade_feats_unstack.apply(lambda y: np.polyfit(reg_x, y.values, 3), axis=1)
    reg_beta_3 = reg_result.apply(lambda x: x[0])
    reg_beta_2 = reg_result.apply(lambda x: x[1])
    reg_beta_1 = reg_result.apply(lambda x: x[2])
    reg_beta_0 = reg_result.apply(lambda x: x[3])
    reg_result_beta = pd.concat([reg_beta_3, reg_beta_2, reg_beta_1, reg_beta_0], axis=1)
    reg_result_beta.columns = ['fund_in_trend_3','fund_in_trend_2','fund_in_trend_1','fund_in_trend_0']
    all_features = pd.merge(all_features, reg_result_beta, left_index=True, right_index=True, how='left')
    # stock buy count trend
    stock_trade = cust_trade.loc[(cust_trade['stktype_cate'] == 'stock') | (cust_trade['buy'] == 1), :]
    stock_trade_group = stock_trade.groupby(['custid', 'biz_week'])
    trade_feats=stock_trade_group['sno'].count()
    trade_feats_unstack = trade_feats.unstack()
    trade_feats_unstack.fillna(0, inplace=True)
    reg_x = range(1, 15)
    reg_result = trade_feats_unstack.apply(lambda y: np.polyfit(reg_x, y.values, 3), axis=1)
    reg_beta_3 = reg_result.apply(lambda x: x[0])
    reg_beta_2 = reg_result.apply(lambda x: x[1])
    reg_beta_1 = reg_result.apply(lambda x: x[2])
    reg_beta_0 = reg_result.apply(lambda x: x[3])
    reg_result_beta = pd.concat([reg_beta_3, reg_beta_2, reg_beta_1, reg_beta_0], axis=1)
    reg_result_beta.columns = ['stock_buy_trend_3', 'stock_buy_trend_2', 'stock_buy_trend_1', 'stock_buy_trend_0']
    all_features = pd.merge(all_features, reg_result_beta, left_index=True, right_index=True, how='left')
    # stock sell count trend
    stock_trade = cust_trade.loc[(cust_trade['stktype_cate'] == 'stock') | (cust_trade['buy'] == 0), :]
    stock_trade_group = stock_trade.groupby(['custid', 'biz_week'])
    trade_feats=stock_trade_group['sno'].count()
    trade_feats_unstack = trade_feats.unstack()
    trade_feats_unstack.fillna(0, inplace=True)
    reg_x = range(1, 15)
    reg_result = trade_feats_unstack.apply(lambda y: np.polyfit(reg_x, y.values, 3), axis=1)
    reg_beta_3 = reg_result.apply(lambda x: x[0])
    reg_beta_2 = reg_result.apply(lambda x: x[1])
    reg_beta_1 = reg_result.apply(lambda x: x[2])
    reg_beta_0 = reg_result.apply(lambda x: x[3])
    reg_result_beta = pd.concat([reg_beta_3, reg_beta_2, reg_beta_1, reg_beta_0], axis=1)
    reg_result_beta.columns = ['stock_sell_trend_3', 'stock_sell_trend_2', 'stock_selltrend_1', 'stock_sell_trend_0']
    all_features = pd.merge(all_features, reg_result_beta, left_index=True, right_index=True, how='left')
    #Convert trend to positive or negative
    pos_neg_list=['fund_out_trend_3','fund_out_trend_2','fund_out_trend_1','fund_out_trend_0',
                  'fund_in_trend_3', 'fund_in_trend_2', 'fund_in_trend_1', 'fund_in_trend_0',
                  'stock_buy_trend_3', 'stock_buy_trend_2', 'stock_buy_trend_1', 'stock_buy_trend_0',
                  'stock_sell_trend_3', 'stock_sell_trend_2', 'stock_selltrend_1', 'stock_sell_trend_0'
                  ]
    for l in pos_neg_list:
        all_features.loc[all_features[l]>0,l]=1
        all_features.loc[all_features[l]<0,l]=-1
    return all_features





if __name__=='__main__':
    pass