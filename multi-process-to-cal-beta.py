#!/Tsan/bin/python
# -*- coding: utf-8 -*-

# Libraries To Use
from __future__ import division
import numpy as np
import pandas as pd
#import statsmodels.api as sm
import os
from sklearn import linear_model
from datetime import datetime,time,date
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Pool, cpu_count
import time

# Import My own library for factor testing
from SingleFactorTest import factorFilterFunctions as ff
#from config import *

path = ff.data_path # path

startTime =  datetime.strptime('20131201', '%Y%m%d')
endTime = datetime.strptime('20170429', '%Y%m%d')
filenameBENCH = 'LZ_GPA_INDXQUOTE_CLOSE.csv'
filenamePrice = 'Own_Factor_AdjustedPriceForward-1d.csv'
HS300Index ='000300.SH'

benchmarkReturn = pd.read_csv(path+filenameBENCH,infer_datetime_format=True,parse_dates=[0],index_col=0)[HS300Index].loc[startTime:endTime].pct_change()
stkreturn = pd.read_csv(path+filenamePrice,infer_datetime_format=True,parse_dates=[0],index_col=0).loc[startTime:endTime].pct_change()

def cal_beta_df(returndf,benchmark,riskFreeRate = 0 ):
    return returndf.apply(lambda x: ((x - riskFreeRate).cov(benchmark)) / (benchmark.var())).values

def process_fun(args):
    return cal_beta_df(*args)

if __name__ == '__main__':
    fargs = []
    start = time.time()
    for i in range(99, len(stkreturn)):
        stkSlice = stkreturn  .iloc[i-99:i+1]
        benchmarkSlice = benchmarkReturn.iloc[i-99:i+1]
        fargs.append((stkSlice,benchmarkSlice))
    #print '%s to complete iteration' %(time.time() - start)
    #print cal_beta_df(fargs[-1][0],fargs[-1][1])


    pool = Pool(cpu_count()-1)
    #results = map(lambda x: pool.apply_async(process_fun, x), fargs)   # apply_async 不报错！
    #pool.close()
    #pool.join()
    #values = map(lambda x: x.get() if x.successful() else [np.nan]*len(stkreturn.columns), results)
    results = pool.map(process_fun, fargs)
    #dff = reduce(lambda x,y: pd.concat([x,y],axis=1),results)
    #values = map(lambda x: x.values, results)
    #dff.columns = stkreturn.iloc[49:60]
    df = pd.DataFrame(index = stkreturn.index[99:len(stkreturn)], columns=stkreturn.columns, data = results)
    print '%s seconds elapsed' % (time.time() - start)
    print (df)
