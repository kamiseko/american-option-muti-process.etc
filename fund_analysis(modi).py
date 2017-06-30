#!/Tsan/bin/python
# -*- coding: utf-8 -*-

# Libraries to use
from __future__ import division
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
#import matplotlib.pyplot as plt
#import seaborn as sns
from datetime import datetime
import mysql.connector

import time



from multiprocessing import Pool, cpu_count

tableName ='fund_nv_standard_w'  # table to query
indexID = 'hs300'  # benchmark
riskFreeRate = 0.02
weeksIn1M = 4  # constants
weeksIn3M = 12
weeksIn6M = 24
weeksIn1Y = 50
varThreshold = 0.05


# headers
timeCol = ['period']
cumReturnCol = ['cumulative_return','monthly_return_cum','quarterly_return_cum','annual_return_cum']
annualizedReturnCol = ['weekly_return_average_annualized','weekly_return_m1_annualized','weekly_return_m3_annualized','weekly_return_m6_annualized',\
                      'weekly_return_y1_annualized']
annualizedStdCol = ['total_weekly_std_annualized','weekly_std_m1_annualized','weekly_std_m3_annualized','weekly_std_m6_annualized',\
                    'weekly_std_y1_annualized']
oddsCol = ['total_weekly_odds','weekly_odds_m1','weekly_odds_m3','weekly_odds_m6','weekly_odds_y1']
downsideriskCol = ['total_weekly_downsideRisk_annualized','weekly_downsideRisk_m1_annualized','weekly_downsideRisk_m3_annualized',\
                  'weekly_downsideRisk_m6_annualized','weekly_downsideRisk_y1_annualized']
maxdrawdownCol = ['total_max_dd','max_dd_start_date','max_dd_end_date']
skewnessCol = ['total_weekly_skewness_annualized','weekly_skewness_m1_annualized','weekly_skewness_m3_annualized','weekly_skewness_m6_annualized',\
              'weekly_skewness_y1_annualized']
kurtosisCol = ['total_weekly_kurtosis_annualized','weekly_kurtosis_m1_annualized','weekly_kurtosis_m3_annualized','weekly_kurtosis_m6_annualized',\
              'weekly_kurtosis_y1_annualized']
betaCol = ['total_beta','beta_m1','beta_m3','beta_m6','beta_y1']
activeriskCol = ['total_active_risk_annualized','active_risk_m1_annualized','active_risk_m3_annualized','active_risk_m6_annualized',\
                 'active_risk_y1_annualized']
sharpeCol = ['total_sharpe_ratio','sharpe_ratio_m1','sharpe_ratio_m3','sharpe_ratio_m6','sharpe_ratio_y1']
calmarCol =['total_calmar_ratio','calmar_ratio_m1','calmar_ratio_m3','calmar_ratio_m6','calmar_ratio_y1']
sortinoCol =['total_sortino_ratio','sortino_ratio_m1','sortino_ratio_m3','sortino_ratio_m6','sortino_ratio_y1']
jensenCol =['total_jensen_ratio','jensen_ratio_m1','jensen_ratio_m3','jensen_ratio_m6','jensen_ratio_y1']
treynorCol =['total_treynor_ratio','treynor_ratio_m1','treynor_ratio_m3','treynor_ratio_m6','treynor_ratio_y1']
informationCol =['total_information_ratio','information_ratio_m1','information_ratio_m3','information_ratio_m6','information_ratio_y1']
varCol = ['total_var','var_m1','var_m3','var_m6','var_y1']

totalCol = timeCol + cumReturnCol + annualizedReturnCol + annualizedStdCol + oddsCol + downsideriskCol + maxdrawdownCol + skewnessCol + kurtosisCol \
           + betaCol + activeriskCol + sharpeCol + calmarCol + sortinoCol + jensenCol + treynorCol + informationCol + varCol
# attribute of the  Indicator class
attrList = ['period','cum_return','annualized_return','annualized_std','odds','dowside_risk','max_drawdown','skewness','kurtosis','beta',\
            'annualized_active_risk','sharpe_ratio','calmar_ratio','sortino_ratio','jensen_ratio','treynor_ratio','information_ratio','value_at_risk']

'''
cnx = mysql.connector.connect(user='tai', password='tai2015',
                                  host='192.168.77.161', port=3311,
                                  database='PrivateEquityFund', charset='utf8')
'''

cnx = mysql.connector.connect(user='tai', password='tai2015',
                               host='119.254.153.20',port = 13311,
                               database='PrivateEquityFund',charset = 'utf8')

def get_fund_data(fundID, tableName=tableName):   # 获取基金数据
    try:
        # sql_query='select id,name from student where  age > %s'
        cursor = cnx.cursor()
        sql = "select fund_id,statistic_date,swanav from %s where fund_id = '%s'" % (tableName, fundID)
        cursor.execute(sql)
        result = cursor.fetchall()
    except:
        print "No %s data!" %fundID
        pass
        #cnx.close()
    pdResult = pd.DataFrame(result, dtype=float)
    pdResult.columns = ['fund_id', 'date', 'net_worth']
    pdResult = pdResult.drop_duplicates().set_index('date')
    pdResult = pdResult.dropna(axis=0)
    pdResult = pdResult.fillna(method='ffill')
    pdResult['weekly_return'] = pdResult['net_worth'].pct_change()
    return pdResult

def get_benchmark(indexID,tableName = 'market_index'):   # 获取基准数据 默认HS300
    # cnx = mysql.connector.connect(user='tai', password='tai2015',
    #                               host='119.254.153.20',port = 13311,
    #                               database='PrivateEquityFund',charset = 'utf8')

    try:
        cursor = cnx.cursor()
        sql = "select %s,statistic_date from %s " % (indexID,tableName)
        cursor.execute(sql)
        result = cursor.fetchall()
    finally:
        pass
        #cnx.close()
    pdResult = pd.DataFrame(result,dtype =float)
    pdResult.columns = ['indexID','date']
    pdResult = pdResult.dropna(axis=0)
    pdResult = pdResult.drop_duplicates().set_index('date')
    pdResult = pdResult.fillna(method = 'ffill')
    return pdResult

def get_fundID_List(tableName = 'fund_info'):   # 获取基金ID列表
    # cnx = mysql.connector.connect(user='tai', password='tai2015',
    #                               host='119.254.153.20',port = 13311,
    #                               database='PrivateEquityFund',charset = 'utf8')

    try:
        cursor = cnx.cursor()
        sql = "select fund_id from %s " % (tableName)
        cursor.execute(sql)
        result = cursor.fetchall()
    finally:
        pass
        #cnx.close()
    return list(set(result))

# 指标计算类
class Indicators:


    def __init__(self, pdResult,
                 benchmarkreturn):  # pdResult is the outcome of the previous code and benchmarkreturn is a pd.Series
        self.fundID = pdResult['fund_id'].iloc[0]
        self.return_series = pdResult['weekly_return']
        self.networth_series = pdResult['net_worth']

        self.return_series_m1 = pdResult['weekly_return'].iloc[-weeksIn1M:]
        self.return_series_m3 = pdResult['weekly_return'].iloc[-weeksIn3M:]
        self.return_series_m6 = pdResult['weekly_return'].iloc[-weeksIn6M:]
        self.return_series_y1 = pdResult['weekly_return'].iloc[-weeksIn1Y:] if len(pdResult['weekly_return']) >= 50 else \
        pdResult['weekly_return']

        self.applyList = [self.return_series, self.return_series_m1, self.return_series_m3, self.return_series_m6,
                          self.return_series_y1]
        self.applyNWList = [self.networth_series, self.networth_series.iloc[-weeksIn1M:],
                            self.networth_series.iloc[-weeksIn3M:] \
            , self.networth_series.iloc[-weeksIn6M:], self.networth_series.iloc[-weeksIn1Y:]]
        #  benchmark

        self.benchmarkreturn = benchmarkreturn['indexID']
        self.benchList = [self.benchmarkreturn, self.benchmarkreturn.iloc[-weeksIn1M:],
                          self.benchmarkreturn.iloc[-weeksIn3M:], \
                          self.benchmarkreturn.iloc[-weeksIn6M:], self.benchmarkreturn.iloc[-weeksIn1Y:]]

        #  cumulative return
        self.period = np.array([pdResult.shape[0]])

    @staticmethod
    def cal_average_return(returnSeries):
        return ((1 + returnSeries.mean()) ** weeksIn1Y - 1)

    @staticmethod
    def cal_std(returnSeries):
        return returnSeries.std() * np.sqrt(weeksIn1Y)

    @staticmethod
    def cal_skewness(returnSeries):
        return returnSeries.skew() * np.sqrt(weeksIn1Y)

    @staticmethod
    def cal_kurtosis(returnSeries):
        return returnSeries.kurtosis() * weeksIn1Y

    @staticmethod
    def cal_downside_risk(returnSeries):
        rs = returnSeries.copy()
        rs[rs > rs.mean()] = 0
        return rs.std(skipna=True) * np.sqrt(weeksIn1Y)

    @staticmethod
    def cal_max_dd(networthSeries):
        maxdd = pd.DataFrame(index=networthSeries.index, data=None, columns=['max_dd', 'max_dd_period'], dtype=float)
        maxdd.iloc[0] = 0
        for date in networthSeries.index[1:]:
            maxdd['max_dd'].loc[date] = 1 - networthSeries.loc[date] / networthSeries.loc[:date].max()
            maxdd['max_dd_period'].loc[date] = (networthSeries.loc[:date].idxmax(), date)
        return maxdd['max_dd'].max(), maxdd.loc[maxdd['max_dd'].idxmax]['max_dd_period'][0], \
               maxdd.loc[maxdd['max_dd'].idxmax]['max_dd_period'][1]

    @staticmethod
    def cal_beta(returnSeries, benchmark):
        benchmarkSlice = benchmark.loc[returnSeries.iloc[1:].index] - riskFreeRate
        return ((returnSeries.iloc[1:] - riskFreeRate).cov(benchmarkSlice)) / (benchmarkSlice.std() ** 2)

    @staticmethod
    def cal_var(returnSeries, alpha=varThreshold):
        return returnSeries.quantile(alpha)

    @staticmethod
    def cal_active_return(returnSeries, benchmark):
        benchmarkSlice = benchmark.loc[returnSeries.index]
        return ((1 + (returnSeries - benchmarkSlice).mean()) ** weeksIn1Y - 1)

    @staticmethod
    def cal_active_risk(returnSeries, benchmark):
        benchmarkSlice = benchmark.loc[returnSeries.index]
        return (returnSeries - benchmarkSlice).std() * np.sqrt(weeksIn1Y)

    @staticmethod
    def cal_odds(returnSeries):
        return len(returnSeries[returnSeries > 0]) / len(returnSeries)

    def run(self):
        if len(self.return_series) > weeksIn1Y:
            self.weekly_return_cum = self.return_series.iloc[-1]
            self.monthly_return_cum = (self.networth_series.iloc[-1] - self.networth_series.iloc[-weeksIn1M]) / \
                                      self.networth_series.iloc[-weeksIn1M]
            self.quarterly_return_cum = (self.networth_series.iloc[-1] - self.networth_series.iloc[-weeksIn3M]) / \
                                        self.networth_series.iloc[-weeksIn3M]
            self.annual_return_cum = (
            (self.networth_series.iloc[-1] - self.networth_series.iloc[-weeksIn1Y]) / self.networth_series.iloc[-weeksIn1Y])

            self.cum_return = np.array([self.weekly_return_cum, self.monthly_return_cum, self.quarterly_return_cum,
                                        self.annual_return_cum])  # 累积年华收益率
            self.annualized_return = np.array(map(self.cal_average_return, self.applyList))  # 年化收益率
            self.annualized_market_rate = np.array(map(self.cal_average_return, self.benchList))  # 基准年化收益率
            self.odds = np.array(map(self.cal_odds, self.applyList))
            self.annualized_std = np.array(map(self.cal_std, self.applyList))  # 年化标准差
            self.dowside_risk = np.array(map(self.cal_downside_risk, self.applyList))  # 年化下行波动率
            self.max_drawdown = np.array(map(self.cal_max_dd, [self.applyNWList[0]])).flatten()  # 最大回撤
            self.skewness = np.array(map(self.cal_skewness, self.applyList))  # 年化偏度
            self.kurtosis = np.array(map(self.cal_kurtosis, self.applyList))  # 年化峰度
            self.beta = np.array([self.cal_beta(i, self.benchmarkreturn) for i in self.applyList])  # beta系数
            self.annualized_active_risk = np.array(
                [self.cal_active_risk(i, self.benchmarkreturn) for i in self.applyList])  # 年化主动风险
            self.sortino_ratio = (self.annualized_return - riskFreeRate) / self.dowside_risk  # sortino比率
            self.sharpe_ratio = (self.annualized_return - riskFreeRate) / self.annualized_std  # sharpe比率
            self.calmar_ratio = (self.annualized_return - riskFreeRate) / np.array(self.max_drawdown[0])  # calmar比率
            self.information_ratio = (
                                     self.annualized_return - self.annualized_market_rate) / self.annualized_active_risk  # 信息系数
            self.jensen_ratio = (self.annualized_return - riskFreeRate) - self.beta * (
            self.annualized_return - self.annualized_market_rate)  # jenson比率
            self.treynor_ratio = (self.annualized_return - riskFreeRate) / self.beta  # treynor比率
            self.value_at_risk = np.array(map(self.cal_var, self.applyList))   # VAR
        else:
            print 'Indexing error! data length of %s is not enough!' %(self.fundID)





def cal_fund_indicators(fundID, benchMarkDatafull):  # 计算单只基金的各个指标
    pdResult = get_fund_data(fundID, tableName=tableName)
    benchMarkData = benchMarkDatafull.loc[pdResult.index].fillna(method='ffill').fillna(
        method='bfill').pct_change()  # 对齐基准数据
    ind = Indicators(pdResult, benchMarkData)
    ind.run()

    dataDic = {}
    dataList = []
    for att in attrList:
        dataDic[att] = getattr(ind, att)
        dataList.append(getattr(ind, att))
    dataList = map(lambda x: x.tolist(), dataList)
    dataList = [item for sublist in dataList for item in sublist]
    return dataList



def process_fun(*args):
    return cal_fund_indicators(*args)



if __name__ == '__main__':
    start = time.time()
    finaldf = pd.DataFrame()
    benchMarkDatafull = get_benchmark(indexID)
    #fundList = ['JR00000' + str(i) for i in range(1, 10)]  # test
    fundListR = get_fundID_List()   # 服务器上跑的时候用这个
    fundList = sorted(map(lambda x: x[0], fundListR))  # 服务器上跑的时候用这个
    fundList = fundList[15:30]
    print fundList
    pool = Pool(cpu_count()-1)
    fargs = zip(fundList, [benchMarkDatafull] * len(fundList))
    results = map(lambda x: pool.apply_async(process_fun, x), fargs)   # apply_async 不报错！
    pool.close()
    pool.join()
    # map(lambda x: x.wait(), results)
    #values = map(lambda x: x.get(timeout=60), results)
    #for i in results:
    #    print i.wait(timeout=60)
    values = map(lambda x: x.get() if x.successful() else [np.nan]*len(totalCol), results)
    df = pd.DataFrame(index = fundList, columns = totalCol, data = np.array(values))
    df = df.round(5)
    path = 'C:\Users\DIY\Desktop\\roy'
    print '%s time elapsed' %(time.time() - start)
    print df
    #df.to_csv(os.path.join(path, 'fund_indicators.csv'), na_rep='NaN',date_format='%Y%m%d')
