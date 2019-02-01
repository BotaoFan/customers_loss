#-*- coding:utf-8 -*-
'''
Script for getting market data
'''

import tushare as ts
import pandas as pd
import numpy as np
from datetime import datetime


pro=ts.pro_api('2ad42b0cc3eb93d83e6d1e4b3f39300968f888cd5810f44974a1cd83')
stock_info=pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')

#Get stock price between start_date and end_date
start_date='20170501'
end_date='20170731'
date_list=pd.date_range(start=start_date,end=end_date)
#取股票每日前复权行情
stock_close=pd.DataFrame()
stock_list=stock_info['ts_code']
for s in stock_list:
    try:
        s_price=ts.pro_bar(pro_api=pro, ts_code=s, adj='qfq', start_date=start_date, end_date=end_date)
        s_close = s_price['close']
        s_close.rename(s)
        if len(stock_close) == 0:
            stock_close = s_close
        else:
            stock_close = pd.concat([stock_close, s_close], axis=1, )
    except:
        print s

del s_close,s_price,stock_list