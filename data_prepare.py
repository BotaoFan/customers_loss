#-*- coding:utf-8 -*-
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

pd.set_option('display.width',1000)
pd.set_option('display.max_columns',1000)
warnings.filterwarnings('ignore')






if __name__=='__main__':
    #Set data dir path
    data_dir_name='raw_data_customers_loss'
    data_dir_path=os.getcwd()+'/../'+data_dir_name+'/'
    #=====Load Raw Data=====
    (cust_info,cust_trade)=dp.load_data(data_dir_path)
    #=====Data Clean=====
    dp.clean_data(cust_info,cust_trade)
    #=====Generate Features======
    all_features=dp.generate_features(cust_info,cust_trade)
    #=====Prepare for model======
    #Convert y to binary variables
    all_features['y']=0
    all_features.loc[all_features['chg_rate']<=-0.5,'y']=1
    y_df=pd.DataFrame(all_features['y'])
    #Drop columns which is not features
    all_features.drop(columns=['khh', 'chg_rate', 'start_jyzc','sec','area'],inplace=True)
    #Seperate train data and test data(7:3)
    test_all_features=all_features.iloc[:23668,]
    train_all_features=all_features.iloc[23668:,]
    train_x=train_all_features.drop(columns=['y']).values
    train_y=pd.DataFrame(train_all_features['y']).values
    test_x=test_all_features.drop(columns=['y']).values
    test_y=pd.DataFrame(test_all_features['y']).values



    #Parameters selection
    cv_params = {'n_estimators': [400, 500, 600, 700, 800]}
    other_params = {'learning_rate': 0.1, 'n_estimators': 500, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0}
    xgboost_paras_select(train_x,train_y,cv_params,other_params)


    other_params = {'learning_rate': 0.1, 'n_estimators': 500, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0}

    import numpy as np
    from sklearn.metrics import roc_auc_score
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    roc_auc_score(true_y, pred_y)


    params = {'learning_rate': 0.1, 'n_estimators': 600, 'max_depth': 5, 'min_child_weight': 1,
              'seed': 0,'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0}
    model_500=xgb.XGBClassifier(**params)
    t_start=datetime.now()
    model_500.fit(train_x,train_y)
    t_end=datetime.now()
    #test score

    test_pred_y=model.predict_proba(test_x)
    test_pred_y = test_pred_y[:, 1]
    roc_auc_score(test_y,test_pred_y)
    test_pred_y_df=pd.DataFrame(test_pred_y,columns=['prob'])
    test_y_df=pd.DataFrame(test_y,columns=['true'])
    test_result_df=pd.concat([test_pred_y_df,test_y_df],axis=1)
    test_result_df.sort_values('prob',inplace=True,ascending=False)


    other_paras = {'learning_rate': 0.1, 'max_depth': 5, 'min_child_weight': 1,
              'seed': 0,'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0}
    cv_paras={'n_estimators':[50,100,200,300,400]}
    model['n_estimators_300'],time_consume['n_estimators_300']=model_train.train_model(paras, train_x, train_y)
    a_700,b_700,c_700=model_train.test_result(model_700,test_x,test_y,bins=20)

    model_train.xgboost_paras_select(train_x,train_y,cv_paras,other_paras,cv=5,verbose=3)










