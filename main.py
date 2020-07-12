import pandas as pd
import numpy as np
import datetime as dt
from scipy import stats                      
from dateutil.relativedelta import *
from pandas.tseries.offsets import *
import scipy as scipy
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso

from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn import linear_model
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score

from multiprocessing import Pool

import pdb

def initializing_df(start_date, end_date): #tested, all good
    """
    getting start with the dataframe
    start_date, end_date format example '1/1/1990', '12/31/1992'
    """
    x_raw = ['bm', 'mom12m', 'me', 'agr', 'op','hxz_abr', 'hxz_sue', 'hxz_re', 'ep', 'cfp', 'sp', 'ni', 'acc', 'roe', 'seas1a', 'adm', 'rdm', 'svar', 'beta', 'mom1m']
    x_rank = ['rank_bm','rank_mom12m', 'rank_me', 'rank_agr', 'rank_op', 'rank_hxz_abr', 'rank_hxz_sue', 'rank_hxz_re', 'rank_ep', 'rank_cfp', 'rank_sp', 'rank_ni', 'rank_acc', 'rank_roe', 'rank_seas1a', 'rank_adm', 'rank_rdm', 'rank_svar', 'rank_beta', 'rank_mom1m']
    x = x_raw + x_rank
    x_raw_fac = ['bm', 'mom12m', 'me', 'agr', 'op']
    x_rank_fac = ['rank_bm', 'rank_mom12m', 'rank_me', 'rank_agr', 'rank_op']
    x_all_fac = ['bm', 'mom12m', 'me', 'agr', 'op', 'rank_bm', 'rank_mom12m', 'rank_me', 'rank_agr', 'rank_op']
    
    df1 = pd.read_csv('eqchars_sp1500_15jun_1990s.csv', usecols = x+ ['permno', 'gvkey', 'public_date'])
    df2 = pd.read_csv('eqchars_sp1500_15jun_2000s.csv', usecols = x+ ['permno', 'gvkey', 'public_date'])
    df3 = pd.read_csv('eqchars_sp1500_15jun_2010s.csv', usecols = x+ ['permno', 'gvkey', 'public_date'])
    
    df = pd.concat([df1, df2, df3])
    df.public_date = pd.to_datetime(df.public_date)
    df.rename(columns = {'public_date': 'jdate'}, inplace = True)
    mask = (df['jdate'] >= start_date) & (df['jdate'] <= end_date)
    df = df.loc[mask]
    df = df[(df['sp400_sort'] == 1)|(df['sp500_sort'] == 1)|(df['sp600_sort'] == 1)]
    df['me'] = np.log(df['me'])

    return df, x_raw, x_rank, x_raw_fac, x_rank_fac


class Regression:
    
    """trying to define a class which allows to
    1/ perform different regression
    2/ store different results as according to regression types (as attribute)
    3/ perfrom prediction as accoding to the previous regression result store
    remark, I am rather new to design patterns, so still learning and increasing the database of design proptotype"""

    def __init__(self):
        self.type = None#initilizing a contro gate for control which prediciton to perform
        self.r2s = []
        self.tvalues = pd.DataFrame()
        self.mseS = []
        self.lambdaS = []
        self.results = []
        
    def normal(self, x, y, x_col): # normal regression
        x = sm.add_constant(x)
        self.model = sm.OLS(y,x).fit()
        self.r2 = self.model.rsquared
        self.tvalue = pd.DataFrame(self.model.tvalues[1:]).T
        self.tvalue.columns = x_col
        
        self.type = 'normal'
        
    def lasso(self, x, y): # lasso regression
        lasso = Lasso()
        parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10 ,20]}
        lasso_regressor = GridSearchCV(lasso, parameters, scoring = 'neg_mean_squared_error', cv =5)   
        lasso_regressor.fit(x, y)
        self.model = lasso_regressor
        self.lambda_ = lasso_regressor.best_params_
        self.mse = lasso_regressor.best_score_
        
        self.type = 'lasso'
        
    def kPcr(self, x, y, x_col, k): # k number of princiapl components regression
        self.pca = PCA(n_components = k, random_state = 10)
        self.stscaler = StandardScaler()
        xstd = self.stscaler.fit_transform(x)
        x_reduced = self.pca.fit_transform(xstd)
        x_reduced = sm.add_constant(x_reduced)
        self.model = sm.OLS(y, x_reduced).fit()
        self.r2 = self.model.rsquared
        self.tvalue = pd.DataFrame(self.model.tvalues[1:]).T
#         self.tvalue.columns = x_col # only 3/5 tvalues
        
        self.type = 'kPcr'
        
    def predict(self, xpred):
        if self.type == 'normal':
            xpred = sm.add_constant(xpred)
            self.yhats = self.model.predict(xpred)
        
        elif self.type == 'lasso':
            self.yhats = self.model.predict(xpred)
        
        elif self.type == 'kPcr':
            xpred = self.stscaler.transform(xpred)
            xpred = self.pca.transform(xpred)
            xpred = sm.add_constant(xpred)
            self.yhats = self.model.predict(xpred)
    def store_performance_metrics(self, today, y):
        if self.type == 'normal' or self.type =='kPcr' :
            self.r2s.append({'r2':self.r2, 'jdate':today, 'char':y})
            self.tvalue['date'] = today
            self.tvalue['char'] = y
            self.tvalues = self.tvalues.append(self.tvalue)
        
        elif self.type == 'lasso':
            self.mseS.append({'mse':self.mse, 'jdate':today, 'char':y})
            self.lambdaS.append({'mse':self.lambda_, 'jdate':today, 'char':y})
        
    def store_yhats(self, permnos, today, y):
        assert len(self.yhats) == len(permnos)
        for yhat, permno in zip(self.yhats, permnos):
            self.results.append({y: yhat, 'permno': permno, 'jdate': today}) #seems previously used fitdate is not correct 

class Regression_Chars_Generator:
    """try to use class structure to be dynamically generating the regression_chars
    this class will wrap another function"""
    # fucntion return ytd_df, today_df, pred_df
    # abstract steps inside the for loop
    def __init__(self, df, x_raw, x_rank, x_raw_fac, x_rank_fac, start_date, end_date):
        self.df = df
        self.x_raw = x_raw
        self.x_rank = x_rank
        self.x_raw_fac = x_raw_fac
        self.x_rank_fac = x_rank_fac
        self.x_all_fac = x_raw_fac + x_rank_fac
        self.x = self.x_raw + self.x_rank
        self.regression = Regression()
        self.dates = sorted(self.df.jdate.unique())
        self.start_date = start_date
        self.end_date = end_date
        
        self.results = pd.DataFrame()
        self.r2s = pd.DataFrame()
        self.tvalues = pd.DataFrame()
        self.mseS = pd.DataFrame()
        self.lambdaS = pd.DataFrame()
        
    def df_spliter(self, i, xfit_lags, xpred_lags):
        self.i = i
        self.xpred_lags = xpred_lags
        ytd = self.dates[i-xfit_lags : i+self.x_months-xfit_lags]
        self.ytd_df = self.df[(self.df.jdate >= ytd[0]) & (self.df.jdate <= ytd[-1])]

        today1 = self.dates[i:i+self.x_months]
        self.today = today1[-1]
        self.today_df = self.df[(self.df.jdate >= today1[0]) & (self.df.jdate <= today1[-1])]
        self.today_dates = sorted(self.today_df.jdate.unique())
        
        pred_date = self.dates[i+self.x_months-1-xpred_lags]
        self.pred_df = self.df[self.df.jdate == pred_date]
    
    def same_permnos_df_cleaner(self):
#         today_dates = sorted(self.today_df.jdate.unique())
        for k ,d in enumerate(self.today_dates, 1):
            permnos = list(set(self.ytd_df.groupby('jdate').get_group(self.dates[self.i-1+k-self.xfit_lags]).permno) 
                       & set(self.today_df.groupby('jdate').get_group(self.dates[self.i-1+k]).permno))
            
            self.permnos = sorted(permnos)
            
            ytd_df1 = self.ytd_df[self.ytd_df['jdate'] == self.dates[self.i-1+k-self.xfit_lags]][self.ytd_df.permno.isin(self.permnos)]
            today_df1 = self.today_df[self.today_df['jdate'] == self.dates[self.i-1+k]][self.today_df.permno.isin(self.permnos)]

            self.ytd_df2 = self.ytd_df2.append(ytd_df1)
            self.today_df2 = self.today_df2.append(today_df1)
            self.ytd_df2 = self.ytd_df2.sort_values(['jdate', 'permno'])
            self.today_df2 = self.today_df2.sort_values(['jdate', 'permno'])
        
#         breakpoint()
        self.permnos1 = list(set(self.pred_df.permno)& set(self.permnos))
        self.permnos1 = sorted(self.permnos1)
        self.pred_df = self.pred_df[self.pred_df.permno.isin(self.permnos1)]
    def chars_regression_loop(self, j, char):
        """funaction to use the class regression to perform regression"""
        
        self.y = char
        non_y = self.x_raw[:j] + self.x_raw[j+1:] + self.x_rank[:j] + self.x_rank[j+1:]
#   
        train_X = self.ytd_df2[non_y].values
        train_Y = self.today_df2[self.y].values
#             self.regression = Regression()
        if self.regress_type == 'normal':
            self.regression.normal(train_X, train_Y, non_y)

        elif self.regress_type == 'lasso':
            self.regression.lasso(train_X, train_Y)

        elif self.regress_type == '3pcr':
            self.regression.kPcr(train_X, train_Y, non_y, 3)

        elif self.regress_type == '5pcr':
            self.regression.kPcr(train_X, train_Y, non_y, 5)
            
        self.regression.store_performance_metrics(self.today, self.y)

        xpred = self.pred_df[non_y].values
        self.regression.predict(xpred)
        self.regression.store_yhats(self.permnos1, self.today, self.y)
        
        results1 = pd.DataFrame(self.regression.results)
        if self.regression.type == 'normal' or self.regression.type =='kPcr' :
            r2s1 = pd.DataFrame(self.regression.r2s)
            return results1, r2s1, self.regression.tvalues
        elif self.regression.type == 'lasso':
            results1 = pd.DataFrame(self.regression.results)
            mseS1 = pd.DataFrame(self.regression.mseS)
            lambdaS1 = pd.DataFrame(self.regression.lambdaS)
            return results1, mseS1, lambdaS1
        
    def chars_regression_helper(self, args):
        a = args[0]
        b = args[1]
        return self.chars_regression_loop(a,b)
    
    def chars_regression_mp(self, chars):
        ls_arg = [(a,b) for (a,b) in enumerate(chars)]
        print(ls_arg)
        
        with Pool() as pool:
            chars_regression_results = pool.map(self.chars_regression_helper, ls_arg)
            
        results1, metric1, metric2 = zip(*chars_regression_results)
        results1 = pd.concat(results1, axis = 1)
        
        self.results = pd.concat([self.results, results1], axis = 0)
        print(10, self.results)
        
        if self.regression.type == 'normal' or self.regression.type =='kPcr' :
            metric1 = pd.concat(metric1, axis = 1)
            self.r2s = pd.concat([self.r2s, metric1], axis = 0)
            self.tvalues = pd.concat([self.tvalues, metric2], axis = 0)
        elif self.regression.type == 'lasso':
            metric1 = pd.concat(metric1, axis = 1)
            metric2 = pd.concat(metric2, axis = 1)
            self.mseS = pd.concat([self.mseS, metric1], axis = 0)
            self.lambdaS = pd.concat([self.lambdaS, metric2], axis = 0)
            
    def char_generate(self, x_months, xfit_lags, xpred_lags, regress_type):
        """function that do the main execution"""
        self.x_months = x_months
        self.regress_type = regress_type
        self.xfit_lags = xfit_lags
        for i, today in enumerate(self.dates[1:-x_months], 1): #didnt use months+1 here
#             self.today = today
            self.ytd_df2 = pd.DataFrame()
            self.today_df2 = pd.DataFrame()
        
            self.df_spliter(i, xfit_lags, xpred_lags)
            self.same_permnos_df_cleaner()
            self.chars_regression_mp(self.x_all_fac)

#         breakpoint()
        self.results = self.results.loc[:,~self.results.columns.duplicated()]
        print (1, self.results)
        self.results = self.results[self.x_all_fac + ['permno', 'jdate']]
        print (2, self.results)
        self.results = self.results.groupby(['permno', 'jdate']).aggregate(sum)
        print (3, self.results)
        self.xhat = [s + "_hat" for s in self.x_all_fac]
        self.results.columns = self.xhat
        self.results.sort_values(['jdate'])
        print (4, self.results)
        return self.results, self.regression, self.xhat
    
    def merge_chars(self):
        # merge with the original chars
        self.results = self.results.reset_index()
        self.results['count'] = self.results.groupby(['permno']).cumcount()
#         breakpoint()
        print (6, self.results)
        self.results[self.xhat] = self.results.groupby(['permno'])[self.xhat].shift(1)
        print (7, self.results)
        self.results = self.results[self.results['count']!=0].reset_index()
        print (5, self.results)
        
        self.df['count'] = self.df.groupby(['permno']).cumcount()
        self.df[self.x] = self.df.groupby(['permno'])[self.x].shift(1)
        self.df = self.df[self.df['count']!=0]
   
        head = ['jdate', 'permno', 'shrcd','exchcd', 'retadj', 'me', 'wt', 'cumretx', 'date']
        head2 = ['jdate', 'permno', 'shrcd','exchcd', 'retadj', 'me2', 'wt', 'cumretx', 'date']
        df2 = pd.read_csv('crsp3.csv', usecols= head)
        df2['jdate'] = pd.to_datetime(df2['jdate'])
        mask = (df2['jdate'] >= self.start_date) & (df2['jdate'] <= self.end_date)
        df2 = df2.loc[mask]
        df2.rename(columns = {'me' : 'me2'}, inplace = True)
        # df3 = pd.merge(df2, results, how = 'outer', on = ['jdate', 'permno'], indicator = True)
        # many of(~50%) the company in df2 is excluded during the merge, manbe those company are very small
        df3 = pd.merge(df2, self.results, how = 'inner', on = ['jdate', 'permno'])
        self.df4 = pd.merge(df3, self.df, how = 'inner', on = ['jdate', 'permno'])
        self.df4 = self.df4.sort_values(['jdate'])

        self.xall = self.x_all_fac + self.xhat
        self.xall_ret = [s + '_ret' for s in self.xall]

        self.df4['count'] = self.df4.groupby(['permno']).cumcount()
        self.nyse = self.df4[(self.df4['exchcd'] == 1) & (self.df4['me2']>0) & (self.df4['count']>1) & ((self.df4['shrcd'] ==10) | (self.df4['shrcd'] == 11))]
        mask2 = (self.df4['retadj'] > -1) 
        self.df4 = self.df4[mask2]
        
        self.fac_dates = sorted(self.df4.jdate.unique())
        print('self.df4',self.df4)
        return self.df4


def for_loop_form_factor(i, char):
    head = ['jdate', 'permno', 'shrcd','exchcd', 'retadj', 'me', 'wt', 'cumretx', 'date']
    head2 = ['jdate', 'permno', 'shrcd','exchcd', 'retadj', 'me2', 'wt', 'cumretx', 'date']
    nyse_break = pd.DataFrame()
    sort_record = pd.DataFrame()
    port_return = pd.DataFrame()
    xdf = pd.DataFrame()

    nyse_break = generator.nyse.groupby(['jdate'])[char].describe(percentiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]).reset_index()
    nyse_break = nyse_break[['jdate', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%']]

    #cannot use nyse, cause nyse is just nyse universe
    xdf = pd.merge(chars[head2 + [char]], nyse_break, how = 'left', on = ['jdate'])
    
    def bucket(row):
        if row[char] <= row['10%']:
            value = -1
        elif row[char] <= row['20%']:
            value = 0
        elif row[char] <= row['30%']:
            value = 3
        elif row[char] <= row['40%']:
            value = 4
        elif row[char] <= row['50%']:
            value = 5
        elif row[char] <= row['60%']:
            value = 6
        elif row[char] <= row['70%']:
            value = 7
        elif row[char] <= row['80%']:
            value = 8
        elif row[char] <= row['90%']:
            value = 9
        elif row[char] > row['90%']:
            value = 1
        else:
            value = ''
        return value

    xdf[char + '_sort'] = xdf.apply(bucket, axis = 1)

    xdf2 = xdf[(xdf['wt'] > 0) & ((xdf['shrcd'] == 10) | (xdf['shrcd'] == 11))]


    if i == 0:
        sort_record = xdf2[['permno', 'jdate'] + [char + '_sort']]

    if i > 0:
        sort_record = xdf2[char + '_sort']

    ################
    # Form Factors #
    ################
    def wavg(group, avg_name, weight_name):
        d = group[avg_name]
        w = group[weight_name]

        try:
            return (d*w).sum()/w.sum()
        except ZeroDivisionError:
            return np.nan

    vwret = xdf2.groupby(['jdate'] + [char + '_sort']).apply(wavg, 'retadj', 'wt').to_frame().reset_index().rename(columns = {0: 'vwret'})

    vwret_n = xdf2.groupby(['jdate'] + [char + '_sort'])['retadj'].count().reset_index().rename(columns = {'retadj': 'n_firms'})

    # transpose
    factors = vwret.pivot(index = 'jdate', columns = char + '_sort', values = 'vwret').reset_index()
    nfirms = vwret_n.pivot(index = 'jdate', columns = char + '_sort', values = 'n_firms').reset_index()

    port_return[char + '_ret'] = factors[1] - factors[-1]

    return  port_return, sort_record

def form_factor_helper(args):
    a = args[0]
    b = args[1]
    return for_loop_form_factor(a,b)

def form_factor_mp():

    ls_arg = [(a,b) for (a,b) in enumerate(generator.x_all_fac)]
    with Pool() as pool:
        results = pool.map(form_factor_helper, ls_arg)

    port_return1, sort_record1 = zip(*results)
    port_return2 = pd.concat(port_return1, axis = 1)
    sort_record2 = pd.concat(sort_record1, axis = 1)

    port_return2['jdate'] = generator.fac_dates
    port_return2['jdate'] = pd.to_datetime(port_return2['jdate'])
    sort_record2 = pd.merge(sort_record2, generator.df[['permno', 'gvkey','jdate']], how='left', on=['permno','jdate'])

    return port_return2, sort_record2

def write_csv(df, metric, abc, date):
    csv_name = f'{str(generator.regress_type)}_{metric}_{abc}_{date}.csv'
    df.to_csv(csv_name)

def write_the_csvs(date):
    abc = abc_detector()
    date = date
    write_csv(port_return, 'port_return',abc ,date)
    write_csv(sort_record, 'sort_record',abc ,date)
    
    if generator.regress_type == 'normal' or generator.regress_type == '3pcr' or generator.regress_type == '5pcr' :
        write_csv(generator.r2s, 'r2', abc, date)
        write_csv(generator.tvalues, 'tvalue',abc ,date)
    
    if generator.regress_type == 'lasso':
        write_csv(generator.mseS, 'mse',abc ,date)
        write_csv(generator.lambdaS, 'lambda',abc ,date)  
        
def abc_detector():
    if generator.x_months == 1 and generator.xfit_lags == 0 and generator.xpred_lags == 0:
        abc = 'a'
    if generator.x_months == 1 and generator.xfit_lags == 1 and generator.xpred_lags == 0:
        abc = 'b'
    if generator.x_months == 1 and generator.xfit_lags == 1 and generator.xpred_lags == 1:
        abc = 'c'
    if generator.x_months == 3 and generator.xfit_lags == 0 and generator.xpred_lags == 0:
        abc = 'd'
    if generator.x_months == 3 and  generator.xfit_lags == 1 and generator.xpred_lags == 0:
        abc = 'e'
    if generator.x_months ==3 and generator.xfit_lags == 1 and generator.xpred_lags == 1:
        abc = 'f'
    if generator.x_months == 12 and generator.xfit_lags == 0 and generator.xpred_lags == 0:
        abc = 'g'
    if generator.x_months == 12 and generator.xfit_lags == 1 and generator.xpred_lags == 0:
        abc = 'h'
    if generator.x_months == 12 and generator.xfit_lags == 1 and generator.xpred_lags == 1:
        abc = 'i'
    return abc


if __name__ == '__main__':
    print('hi')
    a,b,c,d,e = initializing_df('1/1/1990', '12/31/2000')
    generator = Regression_Chars_Generator(a,b,c,d,e,'1/1/1990', '12/31/2000')
    generator.char_generate(x_months = 3, xfit_lags = 0, xpred_lags = 0, regress_type = "normal")  # tested
    
    chars = generator.merge_chars() #tested
    port_return, sort_record = form_factor_mp()
    
    write_the_csvs('5jul20')
