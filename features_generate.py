#-*- coding:utf-8 -*-
import pandas as pd
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class FeatsBase(object):
    def __init__(self,features,raw_data):
        self.features=features
        self.raw_data=raw_data

    def generate(self):
        pass
class FeatsBuySellWeekly(FeatsBase):
    def generate(self):
        raw_data=self.raw_data
        features=self.features
        #Generate weekly buy and sell count
        trade_feats_group = raw_data.groupby(['custid', 'bizdate_date_week', 'buy'])
        trade_feats = trade_feats_group['sno'].count()
        trade_feats_unstack = (trade_feats.unstack()).unstack()
        trade_feats_unstack.fillna(0, inplace=True)
        features = pd.merge(features, trade_feats_unstack, left_index=True, right_index=True,
                                suffixes=('', '_trade_count'), how='left')

        trade_feats = trade_feats_group['matchamt'].sum()
        trade_feats_unstack = (trade_feats.unstack()).unstack()
        trade_feats_unstack.fillna(0, inplace=True)
        features = pd.merge(features, trade_feats_unstack, left_index=True, right_index=True,
                                suffixes=('', '_trade_amount'), how='left')
        return features
class FeatBuySellStockCate(FeatsBase):
    def generate(self):
        raw_data=self.raw_data
        features=self.features
        #Generate weekly buy and sell count
        trade_feats_group = raw_data.groupby(['custid', 'buy','stktype_cate'])
        trade_feats = trade_feats_group['sno'].count()
        trade_feats_unstack = (trade_feats.unstack()).unstack()
        trade_feats_unstack.fillna(0, inplace=True)
        features = pd.merge(features, trade_feats_unstack, left_index=True, right_index=True,
                                suffixes=('', '_trade_count'), how='left')

        trade_feats = trade_feats_group['matchamt'].sum()
        trade_feats_unstack = (trade_feats.unstack()).unstack()
        trade_feats_unstack.fillna(0, inplace=True)
        features = pd.merge(features, trade_feats_unstack, left_index=True, right_index=True,
                                suffixes=('', '_trade_amount'), how='left')
        return features

class FeatFundOutTrend(FeatsBase):
    def generate(self):
        raw_data=self.raw_data
        features=self.features
        stock_trade = raw_data.loc[(raw_data['stktype_cate'] == 'fund') | (raw_data['buy'] == 1), :]
        stock_trade_group = stock_trade.groupby(['custid', 'bizdate_date_week'])
        trade_feats = stock_trade_group['sno'].count()
        trade_feats_unstack = trade_feats.unstack()
        trade_feats_unstack.fillna(0, inplace=True)
        week_max = len(raw_data['bizdate_date_week'].unique())
        reg_x = range(1, week_max + 1)
        reg_result = trade_feats_unstack.apply(lambda y: np.polyfit(reg_x, y.values, 3), axis=1)
        reg_beta_3 = reg_result.apply(lambda x: x[0])
        reg_beta_2 = reg_result.apply(lambda x: x[1])
        reg_beta_1 = reg_result.apply(lambda x: x[2])
        reg_beta_0 = reg_result.apply(lambda x: x[3])
        reg_result_beta = pd.concat([reg_beta_3, reg_beta_2, reg_beta_1, reg_beta_0], axis=1)
        reg_result_beta.columns = ['fund_out_trend_3', 'fund_out_trend_2', 'fund_out_trend_1', 'fund_out_trend_0']
        features = pd.merge(features, reg_result_beta, left_index=True, right_index=True, how='left')
        return features

class FeatFundInTrend(FeatsBase):
    def generate(self):
        raw_data=self.raw_data
        features=self.features
        stock_trade = raw_data.loc[(raw_data['stktype_cate'] == 'fund') | (raw_data['buy'] == 0), :]
        stock_trade_group = stock_trade.groupby(['custid', 'bizdate_date_week'])
        trade_feats = stock_trade_group['sno'].count()
        trade_feats_unstack = trade_feats.unstack()
        trade_feats_unstack.fillna(0, inplace=True)
        week_max = len(raw_data['bizdate_date_week'].unique())
        reg_x = range(1, week_max + 1)
        reg_result = trade_feats_unstack.apply(lambda y: np.polyfit(reg_x, y.values, 3), axis=1)
        reg_beta_3 = reg_result.apply(lambda x: x[0])
        reg_beta_2 = reg_result.apply(lambda x: x[1])
        reg_beta_1 = reg_result.apply(lambda x: x[2])
        reg_beta_0 = reg_result.apply(lambda x: x[3])
        reg_result_beta = pd.concat([reg_beta_3, reg_beta_2, reg_beta_1, reg_beta_0], axis=1)
        reg_result_beta.columns = ['fund_in_trend_3', 'fund_in_trend_2', 'fund_in_trend_1', 'fund_in_trend_0']
        features = pd.merge(features, reg_result_beta, left_index=True, right_index=True, how='left')
        return features

class FeatStockAmtDiff(FeatsBase):
    def generate(self):
        raw_data=self.raw_data
        features=self.features
        mean_raw_data = raw_data.mean(axis=1)
        std_raw_data = raw_data.std(axis=1)
        std_raw_data[std_raw_data == 0] = np.nan
        raw_data_zscore = mean_raw_data.sub(mean_raw_data, 0).div(std_raw_data, 0)
        columns_count=len(raw_data.columns)
        for i in range(1,columns_count):
            features[i-1]=raw_data_zscore.iloc[:,i]-raw_data_zscore.iloc[:,i-1]
        return features



if __name__=='__main__':


    pass