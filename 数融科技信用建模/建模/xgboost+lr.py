# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 11:24:24 2018

@author: bdfus001
"""

import pandas as pd 
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] 
plt.rcParams['axes.unicode_minus']=False
import xgboost as xgb
import lightgbm as lgb
from time import time
import numpy as np
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier

data=pd.read_csv("C:/Users/bdfus001/Desktop/credit-model/preprocess_data/end_data.csv")
Y=data["Y"]
train=data.drop("Y",axis=1)
dummied=pd.get_dummies(train["sex"])
train=train.drop("sex",axis=1)
train=pd.concat([train,dummied],axis=1)

train_x,test_x,train_y,test_y=train_test_split(train,Y,test_size=0.3,random_state=1)


train_x,train_x_lr,train_y,train_y_lr=train_test_split(train_x,train_y,test_size=0.3,random_state=12)

def XGBoost():
    XGB = xgb.XGBClassifier(objective= 'binary:logistic',seed=123,learning_rate=0.01,n_estimators=800,
                        max_depth=8,colsample_bytree=0.8,subsample=0.8,min_child_weight=2,gamma=1,reg_alpha=2,reg_lambda=2)
    XGB.fit(train_x, train_y)
    Y_pred = XGB.predict_proba(test_x)[:, 1]
    fpr, tpr, _ = roc_curve(test_y, Y_pred)
    auc = roc_auc_score(test_y, Y_pred)
    print('XGBoost: ', auc)
    return fpr, tpr


def XGBoostLR():
    XGB = xgb.XGBClassifier(objective= 'binary:logistic',seed=123,learning_rate=0.01,n_estimators=800,
                        max_depth=8,colsample_bytree=0.8,subsample=0.8,min_child_weight=2,gamma=1,reg_alpha=2,reg_lambda=2)
    XGB.fit(train_x, train_y)
    OHE = OneHotEncoder()
    OHE.fit(XGB.apply(train_x))
    LR = LogisticRegression(n_jobs=4, C=0.1, penalty='l1')
    
    LR.fit(OHE.transform(XGB.apply(train_x_lr)), train_y_lr)
    
    Y_pred = LR.predict_proba(OHE.transform(XGB.apply(test_x)))[:, 1]
    fpr, tpr, _ = roc_curve(test_y, Y_pred)
    auc = roc_auc_score(test_y, Y_pred)
    print('XGBoost + LogisticRegression: ', auc)
    return fpr, tpr


def gbdtLR():
    GBDT= GradientBoostingClassifier(learning_rate=0.005,n_estimators=2400,max_depth=3,min_samples_split=800,min_samples_leaf=600,
                                     max_features=9,subsample=0.7,random_state=20)
    GBDT.fit(train_x, train_y)
    OHE = OneHotEncoder()
    OHE.fit(GBDT.apply(train_x)[:, :, 0])
    LR =  LogisticRegression(n_jobs=4, C=0.1, penalty='l1')
    
    LR.fit(OHE.transform(GBDT.apply(train_x_lr)[:, :, 0]), train_y_lr)
    
    Y_pred = LR.predict_proba(OHE.transform(GBDT.apply(test_x)[:, :, 0]))[:, 1]
    fpr, tpr, _ = roc_curve(test_y, Y_pred)
    auc = roc_auc_score(test_y, Y_pred)
    print('GBDT + LogisticRegression: ', auc)
    return fpr, tpr

fpr_xgb_lr, tpr_xgb_lr = XGBoostLR()
fpr_xgb, tpr_xgb = XGBoost()
fpr_gbdt,tpr_gbdt=gbdtLR()

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_xgb, tpr_xgb, label='XGB')
plt.plot(fpr_xgb_lr, tpr_xgb_lr, label='XGB + LR')
plt.plot(fpr_gbdt, tpr_gbdt, label='GBDT + LR')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()



















