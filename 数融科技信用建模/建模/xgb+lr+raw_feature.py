# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 14:55:20 2018

@author: bdfus001
"""

import pandas as pd 
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] 
plt.rcParams['axes.unicode_minus']=False
import xgboost as xgb
import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from scipy.sparse import hstack

data=pd.read_csv("C:/Users/bdfus001/Desktop/credit-model/preprocess_data/end_data.csv")
Y=data["Y"]
train=data.drop("Y",axis=1)
dummied=pd.get_dummies(train["sex"])
train=train.drop("sex",axis=1)
train=pd.concat([train,dummied],axis=1)

train_x,test_x,train_y,test_y=train_test_split(train,Y,test_size=0.3,random_state=1)

xgb = xgb.XGBClassifier(objective= 'binary:logistic',seed=123,learning_rate=0.01,n_estimators=800,
                        max_depth=8,colsample_bytree=0.8,subsample=0.8,min_child_weight=2,gamma=1,reg_alpha=2,reg_lambda=2)

model_lgb=lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=45, reg_alpha=2, reg_lambda=2,
        max_depth=5, n_estimators=800, objective='binary',
         subsample_freq=1,bagging_fraction= 0.6, feature_fraction= 0.6,
        learning_rate=0.01, min_child_weight=2, random_state=20, n_jobs=4)


gbdt = GradientBoostingClassifier(learning_rate=0.005, n_estimators=2400, max_depth=3, min_samples_split=800,
                                      min_samples_leaf=60, max_features=9, subsample=0.7, random_state=0)

gbdt.fit(train_x, train_y)


######xgboost对原始特征编码

train_x_leaves=gbdt.apply(train_x)[:, :, 0]
test_x_leaves=gbdt.apply(test_x)[:, :, 0]

##合并编码后的训练数据和测试数据

All_leaves=np.concatenate((train_x_leaves,test_x_leaves),axis=0)
All_leaves=All_leaves.astype(np.int32)

##对所有特恒one_hot

one_hot=OneHotEncoder()
x_trans=one_hot.fit_transform(All_leaves)

(train_row,cols)=train_x_leaves.shape

train_x_ext=hstack([train_x_leaves,train_x])
test_x_ext=hstack([test_x_leaves,test_x])

lr=LogisticRegression()
lr.fit(train_x_ext,train_y)
y_pred=lr.predict_proba(test_x_ext)[:,1]

auc=roc_auc_score(test_y,y_pred)

print("xgboost+lightgbm的auc是{}".format(auc))



















