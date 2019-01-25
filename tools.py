#-*- coding=utf8 -*-
import pandas as pd
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_csv_in_dir(path):
    '''
    Load all csv files from the given directory
    :param path:String
    :return:dict
    '''
    raw_data={}
    files_list=[]
    for root,dirs,files in os.walk(path):
        files_list=files
    for file in files_list:
        if file.find('.csv')==-1:
            continue
        file_name=file.replace('.csv','')
        try:
            raw_data[file_name]=pd.read_csv(path+file_name+'.csv',index_col=0,)
        except:
            pass
    return raw_data

def get_size(data):
    print 'The size is %f KB' % (sys.getsizeof(data)/(1024.0))
    print 'The size is %f MB' % (sys.getsizeof(data)/(1024.0**2))
    print 'The size is %f GB' % (sys.getsizeof(data)/(1024.0**3))

def check_date_str_right(date_str):
    '''
    Check the date(which is string) is right
    :param date_str: String
    :return: Boolean
    '''
    if len(date_str)!=8:
        return False

    year=int(date_str[:4])
    month=int(date_str[4:6])
    day=int(date_str[6:])

    if month>12 or month<1:
        return False

    days_31_month=[1,3,5,7,8,10,12]
    days_31=31
    days_30_month=[4,6,9,11]
    days_30=30
    days_28_month=[2]
    days_28=28

    if year%4==0:
        days_28+=1

    if month in days_31_month:
        max_day=days_31
    elif month in days_30_month:
        max_day=days_30
    else:
        max_day=days_28

    if day<1 or day>max_day:
        return False

    return True

class DtypeConvert(object):
    def __init__(self):
        pass
    def to_int32(self,data,col):
        '''
        Convert data[col] to np.int32
        :param data:DataFrame
        :param col: String
        :return: Series
        '''
        return data[col].astype(np.int32)
    def to_category(self,data,col):
        '''
        Convert data[col] to category
        :param data:DataFrame
        :param col: String
        :return: Series
        '''
        return data[col].astype('category')
    def to_float32(self,data,col):
        '''
        Convert data[col] to float32
        :param data:DataFrame
        :param col: String
        :return: Series
        '''
        return data[col].astype(np.float32)
    def auto_convert(self,data):
        type_series=data.dtypes
        for i in type_series.index:
            if type_series[i]==float:
                data[i]=self.to_float32(data,i)
            if type_series[i]==int:
                data[i]=self.to_int32(data,i)
        return data
    def to_int32_group(self,data,col_list):
        for col in col_list:
            data[col]=self.to_int32(data,col)
        return data
    def to_float32_group(self,data,col_list):
        for col in col_list:
            data[col]=self.to_float32(data,col)
        return data

class DataDetect(object):
    def __init__(self):
        pass
    def show_kdeplot(self,data,col,y_col=None):
        plt.figure(figsize=(12,6))
        if y_col is None:
            sns.kdeplot(data[col])
            plt.xlabel(col)
            plt.ylabel('Density')
            plt.title('Density of %s'%col)
        else:
            for y in y_col:
                sns.kdeplot(data.loc[y_col==y,col],label=y)
            plt.legend()
    def count_col_na(self,data,col):
        '''
        Detect the number of missing data in single column
        :param data:DataFrame
        :param col:string
        :return:int
        '''
        return np.sum(data[col].isnull())
    def count_df_na(self,data):
        '''
        Detect the number of missing data in all dataframe
        :param data:
        :return:
        '''
        missing_count=data.isnull().sum()
        missing_count.rename(columns=['count'],inplace=True)
        missing_perc=data.isnull().sum()/len(data)*1e2
        missing_perc.rename(columns=['percent'],inplace=True)
        missing_des=pd.concat([missing_count,missing_perc],axis=1)
        missing_des.rename(columns={0:'Count',1:'Percent(%)'},inplace=True)
        return missing_des
    def numeric_describle(self,data,col):
        detect_data=data[col]
        plt.figure(figsize=(14,10))
        plt.subplot(311)
        sns.kdeplot(detect_data)
        plt.title('Desity of %s' %col)
        plt.subplot(312)
        sns.boxplot(detect_data)
        plt.title('Boxplot of %s' %col)
        plt.subplot(313)
        plt.hist(detect_data)
        plt.xlabel(col)
        plt.title('Histogram of %s' %col)
        print detect_data.describe()


if __name__=='__main__':
    pass
