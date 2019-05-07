#-*- coding:utf-8 -*-
import os
import warnings

import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

import tools
import data_preprocessing as dp
import model_train

pd.set_option('display.width',1000)
pd.set_option('display.max_columns',1000)
warnings.filterwarnings('ignore')

def insert_sort(nums):
    if nums is None:
        return nums
    l=len(nums)
    for i in range(1,l):
        temp=nums[i]
        j=i
        while j>0 and nums[j-1]>=temp:
            nums[j]=nums[j-1]
            j-=1
        nums[j]=temp
    return nums

def quick_sort(nums,l,h):
    if nums is None or l>=h:
        return None
    mid=nums[h]
    less=l-1
    more=l-1
    i=l
    while i<=h:
        if nums[i]<=mid:
            less+=1
            temp=nums[i]
            nums[i]=nums[less]
            nums[less]=temp
        more+=1
        i+=1
    quick_sort(nums,l,less-1)
    quick_sort(nums,less+1,h)

def merge_sort(nums,p,r):
    if p>=r:
        return None
    q=(p+r)/2
    merge_sort(nums,p,q)
    merge_sort(nums,q+1,r)
    merge(nums,p,q,r)

def merge(nums,p,q,r):
    arr_l=nums[p:q+1]
    arr_r=nums[q+1:r+1]
    arr_l.append(np.float('inf'))
    arr_r.append(np.float('inf'))
    i=p
    c1=0
    c2=0
    while i<=r:
        if arr_l[c1]>arr_r[c2]:
            nums[i]=arr_r[c2]
            c2+=1
        else:
            nums[i]=arr_l[c1]
            c1+=1
        i+=1

def generate_train_and_test(all_features,test_set_perc=0.3):
    #Seperate train and test dataset
    len_all_features=len(all_features)
    test_all_features=all_features.iloc[:int(len_all_features*test_set_perc),]
    train_all_features=all_features.iloc[int(len_all_features*test_set_perc):,]
    train_x=train_all_features.drop(columns=['y']).values
    train_y=pd.DataFrame(train_all_features['y']).values
    test_x=test_all_features.drop(columns=['y']).values
    test_y=pd.DataFrame(test_all_features['y']).values
    all_features.drop(columns=['y'],inplace=True)
    return all_features.iloc[:10,:],train_x,train_y,test_x,test_y





def generate_dataset_model_1(data_dir_name='raw_data_customers_loss_17-5_17-8',cycle_number=12):
    #feats:stock trade time series
    import os
    import warnings
    import pandas as pd
    import numpy as np
    from datetime import datetime
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import f1_score
    import xgboost as xgb
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import roc_auc_score
    import tools
    import data_preprocessing as dp
    import model_train

    #cust_info features
    data_dir_path=os.getcwd()+'/../'+data_dir_name+'/'
    (cust_info,cust_trade)=dp.load_data(data_dir_path)
    dp.clean_data_process(cust_info,cust_trade)
    cust_trade=dp.date_seperate(cust_trade,'bizdate_date')
    cust_trade.drop(index=cust_trade[cust_trade['bizdate_date_cycle'] > cycle_number].index, inplace=True)

    cust_features=cust_info.loc[:,['khh','kh_age','age','start_jyzc','chg_rate','start_jyzc_ln','sec','area']]
    cust_features['y']=0
    cust_features.loc[cust_features['chg_rate']<=-0.5,'y']=1
    cust_features.set_index('khh',drop=False,inplace=True)
    all_features=cust_features.copy()
    #get cust_trade_stock
    cust_trade_stock=cust_trade.loc[cust_trade['stktype_cate']=='stock',:]
    #cust_trade_stock clean
    cust_trade_stock.drop(index=cust_trade_stock[cust_trade_stock['matchamt'] < 0].index, inplace=True)
    #stock_trade_times
    stock_trade_group=cust_trade_stock.groupby(['custid','buy','bizdate_date_cycle'])
    stock_trade_count=stock_trade_group['sno'].count()
    stock_trade_count=(stock_trade_count.unstack()).unstack()
    stock_trade_count.fillna(0,inplace=True)
    columns=[]
    for i in stock_trade_count.columns.levels[0]:
        for j in stock_trade_count.columns.levels[1]:
            columns.append('count_cycle-'+str(i)+'_'+'buy-'+str(j))
    stock_trade_count.columns=columns
    all_features=pd.merge(all_features,stock_trade_count,left_index=True,right_index=True,how='left')
    #stock trade amt
    stock_trade_amt=stock_trade_group['matchamt'].sum()
    stock_trade_amt=(stock_trade_amt.unstack()).unstack()
    stock_trade_amt.fillna(0,inplace=True)
    columns=[]
    for i in stock_trade_amt.columns.levels[0]:
        for j in stock_trade_amt.columns.levels[1]:
            columns.append('amt_cycle-'+str(i)+'_'+'buy-'+str(j))
    stock_trade_amt.columns=columns
    all_features=pd.merge(all_features,stock_trade_amt,left_index=True,right_index=True,how='left')
    all_features.drop(columns=['khh', 'chg_rate', 'start_jyzc','sec','area'],inplace=True)
    return all_features

def generate_dataset_model_2(data_dir_name='raw_data_customers_loss_17-5_17-8',cycle_number=12):
    #feats:fund time series
    import os
    import warnings
    import pandas as pd
    import numpy as np
    from datetime import datetime
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import f1_score
    import xgboost as xgb
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import roc_auc_score
    import tools
    import data_preprocessing as dp
    import model_train

    #cust_info features
    data_dir_path=os.getcwd()+'/../'+data_dir_name+'/'
    (cust_info,cust_trade)=dp.load_data(data_dir_path)
    dp.clean_data_process(cust_info,cust_trade)
    cust_trade=dp.date_seperate(cust_trade,'bizdate_date')
    cust_trade.drop(index=cust_trade[cust_trade['bizdate_date_cycle'] > cycle_number].index, inplace=True)

    cust_features=cust_info.loc[:,['khh','kh_age','age','start_jyzc','chg_rate','start_jyzc_ln','sec','area']]
    cust_features['y']=0
    cust_features.loc[cust_features['chg_rate']<=-0.5,'y']=1
    cust_features.set_index('khh',drop=False,inplace=True)
    all_features=cust_features.copy()
    #get cust_trade_stock
    cust_trade_stock=cust_trade.loc[cust_trade['stktype_cate']=='fund',:]
    #cust_trade_stock clean
    cust_trade_stock.drop(index=cust_trade_stock[cust_trade_stock['matchamt'] < 0].index, inplace=True)
    #stock_trade_times
    stock_trade_group=cust_trade_stock.groupby(['custid','buy','bizdate_date_cycle'])
    stock_trade_count=stock_trade_group['sno'].count()
    stock_trade_count=(stock_trade_count.unstack()).unstack()
    stock_trade_count.fillna(0,inplace=True)
    columns=[]
    for i in stock_trade_count.columns.levels[0]:
        for j in stock_trade_count.columns.levels[1]:
            columns.append('count_cycle-'+str(i)+'_'+'buy-'+str(j))
    stock_trade_count.columns=columns
    all_features=pd.merge(all_features,stock_trade_count,left_index=True,right_index=True,how='left')
    #stock trade amt
    stock_trade_amt=stock_trade_group['matchamt'].sum()
    stock_trade_amt=(stock_trade_amt.unstack()).unstack()
    stock_trade_amt.fillna(0,inplace=True)
    columns=[]
    for i in stock_trade_amt.columns.levels[0]:
        for j in stock_trade_amt.columns.levels[1]:
            columns.append('amt_cycle-'+str(i)+'_'+'buy-'+str(j))
    stock_trade_amt.columns=columns
    all_features=pd.merge(all_features,stock_trade_amt,left_index=True,right_index=True,how='left')
    all_features.drop(columns=['khh', 'chg_rate', 'start_jyzc','sec','area'],inplace=True)
    return all_features

def generate_dataset_model_3(data_dir_name='raw_data_customers_loss_17-5_17-8',cycle_number=12):
    #feats:stock trade time series features
    import os
    import warnings
    import pandas as pd
    import numpy as np
    from datetime import datetime
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import f1_score
    import xgboost as xgb
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import roc_auc_score
    import tools
    import data_preprocessing as dp
    import model_train

    #cust_info features
    data_dir_path=os.getcwd()+'/../'+data_dir_name+'/'
    (cust_info,cust_trade)=dp.load_data(data_dir_path)
    dp.clean_data_process(cust_info,cust_trade)
    cust_trade=dp.date_seperate(cust_trade,'bizdate_date')
    cust_trade.drop(index=cust_trade[cust_trade['bizdate_date_cycle'] > cycle_number].index, inplace=True)

    #get cust_trade_stock
    cust_trade_stock=cust_trade.loc[cust_trade['stktype_cate']=='stock',:]
    #cust_trade_stock clean
    cust_trade_stock.drop(index=cust_trade_stock[cust_trade_stock['matchamt'] < 0].index, inplace=True)
    cust_trade_cycle=cust_trade_stock.groupby(['custid','bizdate_date_cycle','buy'])['matchamt'].agg(['count','sum'])
    cust_trade_cycle.reset_index(inplace=True)
    cust_trade_cycle.rename(columns={'sum':'matchamt'},inplace=True)

    stock_trade_group=cust_trade_cycle.groupby(['custid','buy'])
    #Get Trade amount features
    #Get min,max,count,sum,mean,std
    stock_trade_amt=stock_trade_group['matchamt'].agg(['min','max','count','sum','mean','std'])
    #Get median
    stock_trade_amt_feat=pd.DataFrame(stock_trade_group['matchamt'].apply(lambda x:np.median(x)))
    stock_trade_amt_feat.columns=['median']
    stock_trade_amt=pd.merge(stock_trade_amt,stock_trade_amt_feat,left_index=True,right_index=True,how='left')
    #Get kurtosis
    stock_trade_amt_feat=pd.DataFrame(stock_trade_group['matchamt'].apply(lambda x:stats.kurtosis(x)))
    stock_trade_amt_feat.columns=['kurt']
    stock_trade_amt=pd.merge(stock_trade_amt,stock_trade_amt_feat,left_index=True,right_index=True,how='left')
    #Get skewness
    stock_trade_amt_feat=pd.DataFrame(stock_trade_group['matchamt'].apply(lambda x:stats.skew(x)))
    stock_trade_amt_feat.columns=['skew']
    stock_trade_amt=pd.merge(stock_trade_amt,stock_trade_amt_feat,left_index=True,right_index=True,how='left')
    stock_trade_amt=stock_trade_amt.unstack()
    columns=[]
    for i in stock_trade_amt.columns.levels[0]:
        for j in stock_trade_amt.columns.levels[1]:
            columns.append('amt_'+str(i)+'_'+'buy-'+str(j))
    stock_trade_amt.columns=columns

    #Get Trade count features
    #Get min,max,count,sum,mean,std
    stock_trade_count=stock_trade_group['count'].agg(['min','max','count','sum','mean','std'])
    #Get median
    stock_trade_count_feat=pd.DataFrame(stock_trade_group['count'].apply(lambda x:np.median(x)))
    stock_trade_count_feat.columns=['median']
    stock_trade_count=pd.merge(stock_trade_count,stock_trade_count_feat,left_index=True,right_index=True,how='left')
    #Get kurtosis
    stock_trade_count_feat=pd.DataFrame(stock_trade_group['count'].apply(lambda x:stats.kurtosis(x)))
    stock_trade_count_feat.columns=['kurt']
    stock_trade_count=pd.merge(stock_trade_count,stock_trade_count_feat,left_index=True,right_index=True,how='left')
    #Get skewness
    stock_trade_count_feat=pd.DataFrame(stock_trade_group['count'].apply(lambda x:stats.skew(x)))
    stock_trade_count_feat.columns=['skew']
    stock_trade_count=pd.merge(stock_trade_count,stock_trade_count_feat,left_index=True,right_index=True,how='left')
    stock_trade_count=stock_trade_count.unstack()
    columns=[]
    for i in stock_trade_count.columns.levels[0]:
        for j in stock_trade_count.columns.levels[1]:
            columns.append('count_'+str(i)+'_'+'buy-'+str(j))
    stock_trade_count.columns=columns

    #Get buy count trend
    stock_trade_group = cust_trade_cycle.loc[cust_trade_cycle['buy']==1,:].groupby(['custid', 'bizdate_date_cycle'])
    trade_feats=stock_trade_group['count'].sum()
    trade_feats_unstack = trade_feats.unstack()
    trade_feats_unstack.fillna(0, inplace=True)
    cycle_max=len(trade_feats_unstack.columns)
    reg_x=range(1,cycle_max+1)
    reg_result = trade_feats_unstack.apply(lambda y: np.polyfit(reg_x, y.values, 3), axis=1)
    reg_beta_3 = reg_result.apply(lambda x: x[0])
    reg_beta_2 = reg_result.apply(lambda x: x[1])
    reg_beta_1 = reg_result.apply(lambda x: x[2])
    reg_beta_0 = reg_result.apply(lambda x: x[3])
    reg_result_beta = pd.concat([reg_beta_3, reg_beta_2, reg_beta_1, reg_beta_0], axis=1)
    reg_result_beta.columns = ['stock_buy_trend_3', 'stock_buy_trend_2', 'stock_buy_trend_1', 'stock_buy_trend_0']

    #Get sell count trend
    stock_trade_group = cust_trade_cycle.loc[cust_trade_cycle['buy']==0,:].groupby(['custid', 'bizdate_date_cycle'])
    trade_feats=stock_trade_group['count'].sum()
    trade_feats_unstack = trade_feats.unstack()
    trade_feats_unstack.fillna(0, inplace=True)
    cycle_max=len(trade_feats_unstack.columns)
    reg_x=range(1,cycle_max+1)
    reg_result = trade_feats_unstack.apply(lambda y: np.polyfit(reg_x, y.values, 3), axis=1)
    reg_beta_3 = reg_result.apply(lambda x: x[0])
    reg_beta_2 = reg_result.apply(lambda x: x[1])
    reg_beta_1 = reg_result.apply(lambda x: x[2])
    reg_beta_0 = reg_result.apply(lambda x: x[3])
    reg_result_beta_sell = pd.concat([reg_beta_3, reg_beta_2, reg_beta_1, reg_beta_0], axis=1)
    reg_result_beta_sell.columns = ['stock_sell_trend_3', 'stock_sell_trend_2', 'stock_sell_trend_1', 'stock_sell_trend_0']

    #generate cust feas
    cust_features=cust_info.loc[:,['khh','kh_age','age','start_jyzc','chg_rate','start_jyzc_ln','sec','area']]
    cust_features['y']=0
    cust_features.loc[cust_features['chg_rate']<=-0.5,'y']=1
    cust_features.set_index('khh',drop=False,inplace=True)
    all_features=cust_features.copy()

    #all_features merge
    all_features= pd.merge(all_features, stock_trade_amt, left_index=True, right_index=True, how='left')
    all_features= pd.merge(all_features, stock_trade_count, left_index=True, right_index=True, how='left')
    all_features = pd.merge(all_features, reg_result_beta, left_index=True, right_index=True, how='left')
    all_features = pd.merge(all_features, reg_result_beta_sell, left_index=True, right_index=True, how='left')
    all_features.fillna(0,inplace=True)

    all_features.drop(columns=['khh', 'chg_rate', 'start_jyzc','sec','area'],inplace=True)
    return all_features



if __name__=='__main__':
    trade_feat_model1=generate_dataset_model_1(data_dir_name='raw_data_customers_loss_17-5_17-8',cycle_number=12)
    trade_feat_model3=generate_dataset_model_3(data_dir_name='raw_data_customers_loss_17-5_17-8',cycle_number=12)

    feats = pd.merge(trade_feat_model1, trade_feat_model3.iloc[:, 4:], how='left', left_index=True,right_index=True)
    feat_head, train_x, train_y, test_x, test_y=generate_train_and_test(feats, test_set_perc=0.3)

    paras = {'n_estimators':0.1,'n_estimators':100,'min_child_weight':1,'max_depth':4,
              'seed': 0,'gamma':4,'subsample':0.8,'colsample_bytree':0.8}
    model = xgb.XGBClassifier(**paras)
    model.fit(train_x,train_y)

    #======Training Test Score======
    train_pred_y=model.predict_proba(train_x)
    train_pred_y = train_pred_y[:, 1]
    roc_auc_score(train_y,train_pred_y)
    test_pred_y=model.predict_proba(test_x)
    test_pred_y = test_pred_y[:, 1]
    roc_auc_score(test_y,test_pred_y)
    a,b,c=model_train.test_result(model,test_x,test_y,bins=20)

    #======Training Test Score======
    test_trade_feat_model1=generate_dataset_model_1(data_dir_name='raw_data_customers_loss_18-1_18-4',cycle_number=12)
    test_trade_feat_model3=generate_dataset_model_3(data_dir_name='raw_data_customers_loss_18-1_18-4',cycle_number=12)
    test_feats = pd.merge(test_trade_feat_model1, test_trade_feat_model3.iloc[:, 4:], how='left', left_index=True,right_index=True)
    test_feat_head, test_train_x, test_train_y, test_test_x, test_test_y=generate_train_and_test(test_feats, test_set_perc=0)


    test_train_pred_y=model.predict_proba(test_train_x)
    test_train_pred_y = test_train_pred_y[:, 1]
    roc_auc_score(test_train_y,test_train_pred_y)
    a,b,c=model_train.test_result(model,test_train_x,test_train_y,bins=20)

    #2019-03-28
    temp_trade_stock_buy=cust_trade[(cust_trade['buy']==1) & (cust_trade['stktype_cate']=='stock')]
    cust_stock_buy_group=temp_trade_stock_buy.groupby(['custid','bizdate_date_week'])
    cust_stock_buy_amt=cust_stock_buy_group['matchamt'].sum()
    cust_stock_buy_amt=cust_stock_buy_amt.unstack(fill_value=0)
    mean_stock_buy_amt=cust_stock_buy_amt.mean(axis=1)
    std_stock_buy_amt=cust_stock_buy_amt.std(axis=1)
    std_stock_buy_amt[std_stock_buy_amt==0]=np.nan
    cust_stock_buy_amt_zscore=cust_stock_buy_amt.sub(mean_stock_buy_amt,0).div(std_stock_buy_amt,0)





