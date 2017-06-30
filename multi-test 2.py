#!/Tsan/bin/python
# -*- coding: utf-8 -*-

from multiprocessing import Pool, cpu_count
import time
import pandas as pd
import numpy as np
# Import My own library for factor testing
from SingleFactorTest import factorFilterFunctions as ff
#from config import *
path = ff.data_path

filenameUQMassIndex = 'Uqer_factor_MassIndex.csv'  # 失去单调性，无超额收益
sololist = [filenameUQMassIndex ]

def f(x):
    return x*x

def wrapCor(args):
    return calCor(*args)

def calCor(a,b):
    return a.corr(b)

def process_fun(*args):
    return calCor(*args)

def multi_run_wrapper(args):
   return add(*args)

#def add(x,y):
#    return x+y
def add(args):
    (x,y) = args
    return x+y

if __name__ == '__main__':
    '''start = time.time()
    pool = Pool(processes=4)              # start 4 worker processes
    result = pool.apply_async(f, [100])    # evaluate "f(10)" asynchronously
    print result.get(timeout=1)           # prints "100" unless your computer is *very* slow
    print pool.map(f, range(10))          # prints "[0, 1, 4,..., 81]"
    print time.time() -start'''
    sparedata = pd.read_csv(path + sololist[0], infer_datetime_format=True, parse_dates=[0], index_col=0)
    a = sparedata.iloc[-5]
    b = sparedata.iloc[-200:]
    c = []

    start = time.time()
    for i in range(len(b)):
        c.append(b.iloc[i])

    pool = Pool(cpu_count() - 1)
    fargs = zip(c, [a] * len(b))

    results = map(lambda x: pool.apply_async(process_fun, x), fargs)  # apply_async 不报错！

    pool.close()
    pool.join()
    values = map(lambda x: x.get() if x.successful() else np.nan, results)
    print values ,'\n', 'The elapsed time of multi process is %s'%(time.time()- start)

    d = []
    start2 = time.time()
    for i in range(len(b)):
        d.append(calCor(b.iloc[i],a))
    print d
    print ' The elapsed time of single process is %s'%(time.time()-start2)
    '''pool = Pool(4)
    results = pool.map(add, [[1, 2], [2, 3], [3, 4]])

    sparedata = pd.read_csv(path + sololist[0], infer_datetime_format=True, parse_dates=[0], index_col=0)
    a = sparedata.iloc[-5]
    b = sparedata.iloc[-200:]
    c = []

    start = time.time()
    for i in range(len(b)):
        c.append(b.iloc[i])

    pool = Pool(cpu_count() - 1)
    fargs = zip(c, [a] * len(b))
    results = pool.map(wrapCor, fargs)
    #pool.close()
    print results ,'\n',time.time() - start'''

