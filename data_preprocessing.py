#-*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import os
from datetime import datetime

import tools
import features_generate as fg

def load_data(data_dir_path):
    '''
    Load raw data and all data files should be in data_dir_path
    We need .csv file below in the data_dir_path:
        cust_info.csv
        yyb_area.csv
        cust_trade_columns.csv
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
    cust_trade_columns=pd.read_csv(data_dir_path+'cust_trade_columns.csv')
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

def clean_str_date(df,col):
    #Translate all illegal date(in string format) to '18000101'
    df.loc[df[col].isnull(),col]='18000101'
    df.loc[df[col].astype(np.int32).astype(str).apply(lambda x: len(x) < 8), col]='18000101'

def covert_str_date(df,col):
    #Add a new col which is the date format of col and set col which is '18000101' as np.nan
    col_date=col+'_date'
    df[col_date]=pd.to_datetime(df[col].astype(np.int32).astype(str),format='%Y%m%d',errors='coerce')
    df.loc[df[col_date]==datetime(1800,01,01),col_date]=np.nan

def covert_int_str(df,col):
    #Add a new col which is the date format of col and set col which is '18000101' as np.nan
    df[col]=df[col].astype(np.int32).astype(str)


def set_ln_col(df,col):
    col_ln=col+'_ln'
    df[col_ln]=np.log1p(df[col])

def convert_to_minutes_hours(df,col):
    #col should be 'hhmmssss/hmmssss'
    df[col+'_minute']=(df[col].astype(np.int32).astype(str).str[-6:-4]).apply(lambda x: 0 if x=='' else int(x))
    df[col+'_hour']=(df[col].astype(np.int32).astype(str).str[:-6]).apply(lambda x: 0 if x=='' else int(x))
    df.loc[df[col].astype(np.int32).astype(str).apply(lambda x:len(x)<7),col]=np.nan
    df.loc[df[col].isnull(),[col+'_hour',col+'_minute']]=np.nan

def get_date_week_info(df,col):
    #col should be datetime dtype
    df[col+'_weekday']=df[col].apply(lambda x: x.weekday())
    df[col+'_week']=df[col].apply(lambda x: x.week)

def set_category(df,col,cate,default=''):
    #cate should be dict
    df[col+'_cate']=default
    for k in cate.keys():
        df.loc[df[col] == k, col+'_cate'] = cate[k]
def date_seperate(df,col,cycle=5):
    date_list=df[col].unique()
    date_list.sort()
    len_date_list=len(date_list)
    date_df=pd.DataFrame({'cycle_date':date_list,'order':range(len_date_list)})
    date_df[col+'_cycle']=date_df['order']//cycle+1
    date_df['cycle_date']=pd.to_datetime(date_df['cycle_date'])
    df=pd.merge(df,date_df.loc[:,['cycle_date',col+'_cycle']],left_on=col,right_on='cycle_date',how='left')
    del df['cycle_date']
    return df


def clean_data_process(cust_info,cust_trade):
    #Change some cols' dtype to save memory
    dc=tools.DtypeConvert()
    trade_to_int32_list=['yyb','sno','operdate','cleardate','bizdate','moneytype','orgid','brhid','orderdate','ordertime','matchtimes','matchtime']
    trade_to_float32_list=['fundeffect','fundbal','orderprice','matchamt','matchprice','fee_jsxf','fee_sxf']
    cust_trade=dc.to_int32_group(cust_trade,trade_to_int32_list)
    cust_trade=dc.to_float32_group(cust_trade,trade_to_float32_list)
    del trade_to_int32_list,trade_to_float32_list
    #Clean datetime columns and convert them into datetime format
    clean_str_date(cust_info,'khrq')
    covert_str_date(cust_info,'khrq')
    clean_str_date(cust_info,'birthday')
    covert_str_date(cust_info,'birthday')
    clean_str_date(cust_trade,'bizdate')
    covert_str_date(cust_trade,'bizdate')
    get_date_week_info(cust_trade, 'bizdate_date')
    #Convert time into hours and minutes
    convert_to_minutes_hours(cust_trade,'ordertime')
    convert_to_minutes_hours(cust_trade,'matchtime')
    #Change some variables into ln(variables)
    set_ln_col(cust_info,'start_jyzc')
    set_ln_col(cust_info,'end_jyzc')
    #Get some variables
    cust_info['age']=2018-cust_info['birthday_date'].apply(lambda x:x.year)
    cust_info['kh_age']=2018-cust_info['khrq_date'].apply(lambda x: x.year)
    #Set buy or sell
    cust_trade['buy']=1
    cust_trade.loc[cust_trade['fundeffect']>0,'buy']=0
    #Set stktype category
    cate={' ':'fund','0':'stock','G':'G','L':'L','E':'E'}
    set_category(cust_trade,'stktype',cate,'other')


def generate_feats_process(cust_info,cust_trade):
    #cust_info features
    cust_features=cust_info.loc[:,['khh','kh_age','age','start_jyzc','chg_rate','start_jyzc_ln','sec','area']]
    cust_features['y']=0
    cust_features.loc[cust_features['chg_rate']<=-0.5,'y']=1
    cust_features.set_index('khh',drop=False,inplace=True)
    all_features=cust_features.copy()

    f_buy_sell_weekly=fg.FeatsBuySellWeekly(all_features,cust_trade)
    all_features=f_buy_sell_weekly.generate()
    f_buy_sell_stock_cate=fg.FeatBuySellStockCate(all_features,cust_trade)
    all_features=f_buy_sell_stock_cate.generate()
    f_fund_out_trend=fg.FeatFundOutTrend(all_features,cust_trade)
    all_features=f_fund_out_trend.generate()
    f_fund_in_trend=fg.FeatFundInTrend(all_features,cust_trade)
    all_features=f_fund_in_trend.generate()



def generate_features(cust_info,cust_trade):
    #cust_info features
    cust_features=cust_info.loc[:,['khh','kh_age','age','start_jyzc','chg_rate','start_jyzc_ln','sec','area']]
    cust_features['y']=0
    cust_features.loc[cust_features['chg_rate']<=-0.5,'y']=1
    cust_features.set_index('khh',drop=False,inplace=True)
    all_features=cust_features.copy()
    #Trade features
    #Buy and sell every week every customers(all stock type)
    trade_feats_group=cust_trade.groupby(['custid','bizdate_date_week','buy'])
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
    stock_trade_group=stock_trade.groupby(['custid','bizdate_date_week'])
    trade_feats=stock_trade_group['sno'].count()
    trade_feats_unstack=trade_feats.unstack()
    trade_feats_unstack.fillna(0,inplace=True)
    week_max=len(cust_trade.bizdate_date_week.unique())
    reg_x=range(1,week_max+1)
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
    stock_trade_group = stock_trade.groupby(['custid', 'bizdate_date_week'])
    trade_feats=stock_trade_group['sno'].count()
    trade_feats_unstack = trade_feats.unstack()
    trade_feats_unstack.fillna(0, inplace=True)
    reg_x=range(1,week_max+1)
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
    stock_trade_group = stock_trade.groupby(['custid', 'bizdate_date_week'])
    trade_feats=stock_trade_group['sno'].count()
    trade_feats_unstack = trade_feats.unstack()
    trade_feats_unstack.fillna(0, inplace=True)
    reg_x=range(1,week_max+1)
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
    stock_trade_group = stock_trade.groupby(['custid', 'bizdate_date_week'])
    trade_feats=stock_trade_group['sno'].count()
    trade_feats_unstack = trade_feats.unstack()
    trade_feats_unstack.fillna(0, inplace=True)
    reg_x=range(1,week_max+1)
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