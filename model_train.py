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
    for x,y in zip(number,correct_rate):
        plt.text(x,y,"%.1f %%"%(y*100),ha='center',va='bottom',fontsize=10)
    plt.xticks(number,rotation=60)
    plt.legend()

    return result_df,rocau_score,correct_rate,


def xgboost_paras_select(train_x,train_y,cv_paras,other_paras,cv=5,verbose=3,n_jobs=3):
    model=xgb.XGBClassifier(**other_paras)
    optimized_GBM=GridSearchCV(estimator=model,param_grid=cv_paras,scoring='roc_auc',cv=cv,verbose=verbose,n_jobs=n_jobs)
    optimized_GBM.fit(train_x, train_y)
    evalute_result = optimized_GBM.grid_scores_
    print('每轮迭代运行结果:{0}'.format(evalute_result))
    print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
    print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))

class XGBoostTrainer(object):
    def __init__(self,x,y,cv=5,verbose=3,n_jobs=3):
        self.x=x
        self.y=y
        self.cv=cv
        self.verbose=verbose
        self.n_jobs=n_jobs

    def xgboost_single_paras_select(self,cv_paras,other_paras):
        x=self.x.copy()
        y=self.y.copy()
        cv=self.cv
        verbose=self.verbose
        n_jobs=self.n_jobs

        model = xgb.XGBClassifier(**other_paras)
        optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_paras, scoring='roc_auc', cv=cv, verbose=verbose,
                                     n_jobs=n_jobs)
        optimized_GBM.fit(x, y)
        return [optimized_GBM.best_params_,optimized_GBM.best_score_]

    def paras_train(self, best_paras, test_paras, selected_para_names):
        other_paras={}
        cv_paras={}
        for k in best_paras.keys():
            if k not in selected_para_names:
                other_paras[k]=best_paras[k]
            else:
                cv_paras[k]=test_paras[k]
        (paras,score)=self.xgboost_single_paras_select(cv_paras,other_paras)
        for k in selected_para_names:
            best_paras[k]=paras[k]
        print 'Training in ',selected_para_names
        print score


    def model_train(self,init_paras,test_paras):
        '''
        paras is dict contain:{'n_estimators',
        'max_depth','min_child_weight',
        'gamma',
        'subsample','colsample_bytree',
        'learning_rate'
        }
        :param paras:
        :return:
        '''
        best_paras=init_paras.copy()
        #Estimate n_estimators
        selected_para_names=['n_estimators']
        self.paras_train(best_paras,test_paras,selected_para_names)
        #Estimate max_depth and min_child_weight
        selected_para_names=['max_depth','min_child_weight']
        self.paras_train(best_paras,test_paras,selected_para_names)
        #Estimate gamma
        selected_para_names=['gamma']
        self.paras_train(best_paras,test_paras,selected_para_names)
        #Estimate subsample and colsample_bytree
        selected_para_names=['subsample','colsample_bytree']
        self.paras_train(best_paras,test_paras,selected_para_names)
        #Estimate learning_rate
        selected_para_names=['learning_rate']
        self.paras_train(best_paras,test_paras,selected_para_names)
        self.trained_paras=best_paras
        return self.trained_paras






















