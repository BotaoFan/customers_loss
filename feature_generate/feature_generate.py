import pandas as pd
import numpy as np

from data_preprocess import base as pp




class Feature(object):
    def __init__(self):
        pass
    def generate(self):
        pass

#Base Feature
class FeatureExtractColumn(Feature):
    def __init__(self,data):
        '''
        :param data:DataFrame which index should be the index of feature DataFrame
        '''
        self.data=data
        self.data_cols_name=data.columns


    def generate(self,col_name,feature_name=None):
        '''
        :param col_name:str
        :param feature_name:str
        :return:Series named feature_name
        '''
        if feature_name is None:
            feature_name=col_name

#Single Feature
class FeatureAge(Feature):
    def __init__(self):
        pass
    def generate_from_birthday(self,birthday):
        '''
        :param birthday:
        :return: Series named age
        '''


