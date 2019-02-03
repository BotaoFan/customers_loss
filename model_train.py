#-*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from datetime import datetime
import matplotlib.pyplot as plt

def train_model(paras,train_x,train_y):
    model=xgb.XGBClassifier(**paras)
    t_start=datetime.now()
    model.fit(train_x,train_y)
    t_end=datetime.now()
    time_consume=t_end-t_start
    return model,time_consume

def test_result(model,x,y,bins=20,paras_label=''):
    pred_y=model.predict_proba(x)
    pred_y = pred_y[:, 1]
    rocau_score=roc_auc_score(y,pred_y)
    pred_y_df=pd.DataFrame(pred_y,columns=['prob'])
    y_df=pd.DataFrame(y,columns=['true'])
    result_df=pd.concat([pred_y_df,y_df],axis=1)
    result_df.sort_values('prob',inplace=True,ascending=False)
    n=result_df.shape[0]
    step=n/bins
    next_step=step
    number=[]
    correct_rate=[]
    while True:
        if next_step>=n:
            correct_rate.append(result_df.iloc[:n,]['true'].sum()/(n+0.0000))
            number.append(n)
            break
        else:
            correct_rate.append(result_df.iloc[:next_step,]['true'].sum()/(next_step+0.0000))
            number.append(next_step)
            next_step+=step
    #correct_rate_df=pd.DataFrame([number,correct_rate],columns=['number','correct_rate'])
    plt.plot(number,correct_rate,'o-',label=paras_label)
    plt.legend()

    return result_df,rocau_score,correct_rate,


def xgboost_paras_select(train_x,train_y,cv_paras,other_paras,cv=5,verbose=3):
    model=xgb.XGBClassifier(**other_paras)
    optimized_GBM=GridSearchCV(estimator=model,param_grid=cv_paras,scoring='roc_auc',cv=cv,verbose=verbose,n_jobs=4)
    optimized_GBM.fit(train_x, train_y)
    evalute_result = optimized_GBM.grid_scores_
    print('每轮迭代运行结果:{0}'.format(evalute_result))
    print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
    print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))