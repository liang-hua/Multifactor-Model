import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm
class Facter_analys():
    def __init__(self,pct_chg,facters):
        self.pct_chg = pct_chg
        self.facters = facters
        print('收益率',self.pct_chg)
        print('因子暴露', self.facters)

    #离群值处理。dataframe的index为时间，columns为股票列表
    def MAD(self,dataframe,n):
        result = dataframe*0
        for i in range(len(dataframe.index)):
            median = (dataframe.iloc[i]).median()
            mad =  (dataframe.iloc[i]).mad()
            upline = median + n * mad
            downline = median - n * mad
            result.iloc[i] = np.clip(dataframe.iloc[i],downline,upline)
        return result
    #标准化
    def z_score(self,dataframe):
        dataframe = scale(dataframe, axis=1) + dataframe * 0
        return dataframe
    #中性化。X为该时刻的其他因子数据，通常是市值和行业因子，Y为需进行中性化的因子
    def  neut(self,X,Y):
        x = X.fillna(0)
        y = Y.fillna(0)
        x = X.values.reshape(-1,1)
        y = Y.values.reshape(-1,1)
        reg = LinearRegression()
        reg.fit(x, y)
        predictions = reg.predict(x)
        predictions = predictions.reshape(-1)
        result = Y - predictions
        return result
    #回归分析，返回回归系数
    def coef(self,X,Y):
        x = X.fillna(0)
        y = Y.fillna(0)
        x = X.values.reshape(-1, 1)
        y = Y.values.reshape(-1, 1)
        reg = LinearRegression()
        reg.fit(x, y)
        return reg.coef_[0][0]
    #IC法分析，返回单个时间点的IC值
    def IC(self,X,Y):
        x = X.fillna(0)
        y = Y.fillna(0)
        ic = x.corr(y)
        return ic








#========================= 示 例 ==========================
pct_chg = pd.DataFrame([[1,1,5,1,1],[11,11,55,11,11]],index= ['20200827','20200828'])
facters = pd.DataFrame([[2,2,7,2,2],[22,22,77,22,22]],index= ['20200827','20200828'])
a = Facter_analys(pct_chg,facters)
pct_chg1 = a.MAD(pct_chg,3)
facters1 = a.MAD(facters,3)
#print(pct_chg,facters)
pct_chg2 = a.z_score(pct_chg1)
facters2 = a.z_score(pct_chg1)
print(pct_chg2,facters2)
#print(facters2.index)
facters3 = facters2*0
for i in list(facters2.index):
    X1 = facters2.loc[i]
    Y1 = facters2.loc[i]
    #print(type(X1),type(Y1))
    facters3.loc[i] = a.neut(X1,Y1)
print(facters3)
ic_list = []
for i in list(facters2.index):
    X1 = facters2.loc[i]
    Y1 = pct_chg2.loc[i]
    #print(type(X1),type(Y1))
    ic = a.IC(X1,Y1)
    ic_list.append(ic)
#print('ic_list',ic_list)
