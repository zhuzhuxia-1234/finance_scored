# -*- coding: utf-8 -*-
"""
Created on Tue May 21 15:20:39 2019

@author: zhuxibing
"""

import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
import numpy as np

data=pd.read_csv('C:/Users/zhuxibing/Desktop/api_access_fix.csv')

print('size:',data.shape)
print(data.head())

data=data.set_index("date")
data.index=pd.to_datetime(data.index)  #把date设为索引并设为时间数据

data['count'].plot()
plt.show()

#### 总共有10080条数据，每分钟会产生一条，7填的数据，我们把前六天作为训练集，把第七天作为测试集

train=data['count'].iloc[:8640]
test=data['count'].iloc[8640:]

## 对训练数据进行平滑处理，消除数据的毛刺，可以用移动平均法，我这里没有采用，因为我试过发现对于我的数据来说，移动平均处理完后并不能使数据平滑，
##我这里采用的方法很简单，但效果还不错：把每个点与上一点的变化值作为一个新的序列，对这里边的异常值，也就是变化比较离谱的值剃掉，用前后数据的均值填充，注意可能会连续出现变化较大的点

train_diff=train.diff().dropna()
train_diff_des=train_diff.describe()  #获得差分序列后的描述性统计
low=train_diff_des['25%']-1.5*(train_diff_des['75%']-train_diff_des['25%'])
high=train_diff_des['75%']+1.5*(train_diff_des['75%']-train_diff_des['25%'])

## 找出离群点的index
outlier=train_diff[(train_diff>high)|(train_diff<low)].index

i=0
while i< len(outlier)-1:
    n=1  #发现连续有多少个点变化幅度过大，大部分只有单个点
    start=outlier[i]
    while outlier[i+n]==start+timedelta(minutes=n):
        n+=1
    i +=n-1
    end=outlier[i]
    value = np.linspace(train[start - timedelta(minutes=1)], train[end + timedelta(minutes=1)], n)
    train[start: end] = value
    i += 1
train.plot()
plt.show()


###### 将数据进行周期性分解
from statsmodels.tsa.seasonal import seasonal_decompose

decomposition=seasonal_decompose(train,freq=1440,two_sided=False)  #freq周期是一条，即1440分钟

trend=decomposition.trend
seasonal=decomposition.seasonal
residual=decomposition.resid

decomposition.plot()
plt.show()






#对分解出来的趋势部分单独用arima模型做训练：

from statsmodels.tsa.arima_model import ARIMA
train_trend=trend.dropna()
trend_model = ARIMA(train_trend, (1,1,3)).fit(disp=-1, method='css')

##预测未来一条的趋势
predict_trend=trend_model.forecast(1440)[0]

#为预测出的趋势数据添加周期数据和残差数据
train_seasonal=seasonal
pred_time_index= pd.date_range(start=train.index[-1], periods=1440+1, freq='1min')[1:]

values=[]
low_conf_values=[]
high_conf_values=[]

for i, t in enumerate(pred_time_index):
    trend_part=predict_trend[i]                                                #趋势部分
    season_part=train_seasonal[train_seasonal.index.time == t.time()].mean()   #周期部分
    desc=residual.describe()
    
    delta = desc['75%'] - desc['25%']
    low_error,high_error = (desc['25%']-1 *delta,desc['75%'] + 1 * delta)      #残差部分
    
    
    #预测值
    predict = trend_part + season_part
    #预测的上下界
    predict_low_conf=trend_part + season_part+low_error
    predict_high_conf=trend_part + season_part+high_error
    
    values.append(predict)
    low_conf_values.append(predict_low_conf)
    high_conf_values.append(predict_high_conf)



###预测的效果对比
final_pred = pd.Series(values, index=pred_time_index, name='predict')    
    
final_pred.plot(color='salmon', label='Predict')
test.plot(color='steelblue', label='Original')
plt.title('RMSE: %.4f' % np.sqrt(sum((final_pred.values - test.values) ** 2) / test.size))
plt.tight_layout()
plt.show()
















