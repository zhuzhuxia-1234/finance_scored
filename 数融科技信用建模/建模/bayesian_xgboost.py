# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 09:44:25 2018

@author: bdfus001
"""

import pandas as pd 
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] 
plt.rcParams['axes.unicode_minus']=False

data=pd.read_csv("C:/Users/bdfus001/Desktop/credit-model/preprocess_data/end_data.csv")

from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score,accuracy_score,roc_curve
import xgboost as xgb
import lightgbm as lgb
from sklearn.svm import SVC
import matplotlib.pyplot as plt


Y=data["Y"]
train=data.drop("Y",axis=1)
dummied=pd.get_dummies(train["sex"])
train=train.drop("sex",axis=1)
train=pd.concat([train,dummied],axis=1)

train_x,test_x,train_y,test_y=train_test_split(train,Y,test_size=0.3,random_state=1)

from hyperopt import fmin, tpe, hp,space_eval,rand,Trials,partial,STATUS_OK

def GBM(argsDict):
    max_depth = argsDict["max_depth"] + 5
    n_estimators = argsDict['n_estimators'] * 50 + 500
    learning_rate = argsDict["learning_rate"] * 0.02 + 0.05
    subsample = argsDict["subsample"] * 0.1 + 0.7
    min_child_weight = argsDict["min_child_weight"]+1
    print("max_depth:" + str(max_depth))
    print("n_estimator:" + str(n_estimators))
    print("learning_rate:" + str(learning_rate))
    print("subsample:" + str(subsample))
    print("min_child_weight:" + str(min_child_weight))
    global train,Y

    gbm = xgb.XGBClassifier(nthread=4,    #进程数
                            max_depth=max_depth,  #最大深度
                            n_estimators=n_estimators,   #树的数量
                            learning_rate=learning_rate, #学习率
                            subsample=subsample,      #采样数
                            min_child_weight=min_child_weight,   #孩子数
                            max_delta_step = 10,  #10步不降则停止
                            objective="binary:logistic")

    metric = cross_val_score(gbm,train,Y,cv=5,scoring="roc_auc").mean()
    print(metric)
    return metric

space = {"max_depth":hp.randint("max_depth",15),
         "n_estimators":hp.randint("n_estimators",10),  #[0,1,2,3,4,5] -> [50,]
         "learning_rate":hp.randint("learning_rate",6),  #[0,1,2,3,4,5] -> 0.05,0.06
         "subsample":hp.randint("subsample",4),#[0,1,2,3] -> [0.7,0.8,0.9,1.0]
         "min_child_weight":hp.randint("min_child_weight",5), #
        }
algo = partial(tpe.suggest,n_startup_jobs=1)
best = fmin(GBM,space,algo=algo,max_evals=4)

print(best)
print(GBM(best))