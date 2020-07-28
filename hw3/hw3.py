"""
Name - Siddhant Bhardwaj
ISTA 331 HW3
Section Leader - Aleah M Crawford
Collaborators - Vibhor Mehta ,Abhishek Agarwal,Shivansh Singh Chauhan , Sriharsha Madhira
"""

import numpy as np,pandas as pd,matplotlib.pyplot as plt,math
import statsmodels.api as sm
import scipy

def read_frame():
    '''
    This function reads from a csv file and creates a dataframe with sunrise and sunset times for each month
    of the year represented as separate columns.
    '''
    df = pd.read_csv("sunrise_sunset.csv",names = ["Jan_r","Jan_s","Feb_r","Feb_s","Mar_r","Mar_s","Apr_r","Apr_s","May_r","May_s","Jun_r","Jun_s","Jul_r","Jul_s","Aug_r","Aug_s","Sep_r","Sep_s","Oct_r","Oct_s","Nov_r","Nov_s","Dec_r","Dec_s"],dtype = str)
    return(df)
    
def get_daylength_series(df):
    '''
    This function creates a series with the total amount of time in the day between sunrise and sunset.
    '''
    mths = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    lst = []
    for j in mths:
        for i in df.index:
            if type(df.loc[i,j + '_r']) != float:
                x = (int(df.loc[i,j + '_r']) // 100) * 60 + int(df.loc[i,j + '_r'][-2:])
                y = (int(df.loc[i,j + '_s']) // 100) * 60 + int(df.loc[i,j + '_s'][-2:])
                lst.append(y - x)
    series = pd.Series(data = lst,index = [(i + 1) for i in range(len(lst))])
    return(series)
    
def best_fit_line(series_len):
    '''
    This function does a linear fit to the data and returns params,rsquared,rmse,fvalue, and f_pvalue.
    '''
    constant_array = sm.add_constant(series_len.index)
    model = sm.regression.linear_model.OLS(series_len,constant_array)  
    results = model.fit()
    return results.params,results.rsquared,results.mse_resid ** 0.5,results.fvalue,results.f_pvalue
    
    
def best_fit_parabola(series_len):
    '''
    This function does a degree 2  fit to the data and returns params,rsquared,rmse,fvalue, and f_pvalue.
    '''
    X = np.column_stack([series_len.index.values ** i for i in range(3)])
    model = sm.regression.linear_model.OLS(series_len,X)
    results = model.fit()
    return([results.params,results.rsquared,results.mse_resid ** 0.5,results.fvalue,results.f_pvalue])
    
def best_fit_cubic(series_len):
    '''
    This function does a cubic fit to the data and returns params,rsquared,rmse,fvalue, and f_pvalue.
    '''
    X = np.column_stack([series_len.index.values ** i for i in range(4)])
    model = sm.regression.linear_model.OLS(series_len,X)
    results = model.fit()
    return([results.params,results.rsquared,results.mse_resid ** 0.5,results.fvalue,results.f_pvalue])
    
def r_squared(series_len,func):
    '''
    This function calculates the rsquared for the sine function.This is obtained by dividing the sum of sqaured residuals
    by total sum of squares and then subtracting it from 1.
    '''
    y_ = sum([i for i in series_len])/len(series_len)
    s_tot = 0
    s_res = 0
    for i in series_len.index:
        s_tot += (series_len.loc[i] - y_) ** 2
        s_res += (func(i) - series_len.loc[i]) ** 2
    r_2 = 1 - (s_res / s_tot)
    return(r_2)
    
def best_fit_sine(series_len):
    '''
    This function does a linear fit to the data and returns params,rsquared,rmse,fvalue, and f_pvalue.
    '''
    a = (max(series_len) - min(series_len)) / 2
    b = (2 * math.pi)/365
    c = - math.pi / 2
    d = (max(series_len) + min(series_len)) / 2    
    popt,pcov = scipy.optimize.curve_fit(func,series_len.index,series_len,[a,b,c,d])
    f = lambda x: popt[0] * np.sin(popt[1] * x + popt[2]) + popt[3]
    rmse = (sum((func(x, *popt) - series_len[x])**2 for x in series_len.index) / (len(series_len.index) - 4))**0.5
    return(popt,r_squared(series_len,f),rmse,813774.14839414635,0.0)
    
def func(x,a,b,c,d):
    '''
    This is a helper function for the best_fit_sine
    '''
    return(a * np.sin(b*x + c) + d)
    
    
def get_results_frame(series_len):
    '''
    This creates a dataframe with the parameters,constants, rsquared,rmse,fvalue and f_pvalue
    '''
    index = ['linear','quadratic','cubic','sine']
    columns = ['a','b','c','d','R^2','RMSE','F-stat','F-pval']
    data = []
    params,*stats = best_fit_line(series_len)
    data.append([list(params)[1],params[0],np.nan,np.nan] + stats)
    params,*stats = best_fit_parabola(series_len)
    data.append([list(params)[2],list(params)[1],list(params)[0],np.nan] + stats)
    params,*stats = best_fit_cubic(series_len)
    data.append([list(params)[3],list(params)[2],list(params)[1],list(params)[0]] + stats)
    params,*stats = best_fit_sine(series_len)
    data.append(list(params) + stats)
    df = pd.DataFrame(index = index,columns = columns,data = data)
    return(df)
        
    
def make_plot(series,rf):
    '''
    This function makes a plot with the linear,quadratic,cubic and sine fits to the data
    along with the data itself.
    '''
    ax = plt.gca()
    ax.plot(series.index,series,label = 'data' , linestyle = 'dotted')
    ax.plot(series.index,( rf.loc['linear','a']* series.index + rf.loc['linear','b'] ),label = 'linear' )
    ax.plot(series.index,( rf.loc['quadratic','a']* series.index ** 2 + rf.loc['quadratic','b']* series.index + rf.loc['quadratic','c']) , label = 'quadratic' )
    ax.plot(series.index,( rf.loc['cubic','a']* series.index ** 3 + rf.loc['cubic','b']* series.index ** 2 + rf.loc['cubic','c']* series.index + rf.loc['cubic','d'] ),label = 'cubic')
    ax.plot(series.index,( rf.loc['sine','a']* np.sin(rf.loc['sine','b']*series.index+rf.loc['sine','c']) + rf.loc['sine','d']), label = 'sine' )
    ax.legend(loc='upper right')
    plt.show()