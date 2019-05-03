# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 15:38:29 2018

@author: bdfus001
"""

import pandas as pd 
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] 
plt.rcParams['axes.unicode_minus']=False

data=pd.read_csv("end_data.csv")

from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import roc_auc_score,roc_curve,precision_recall_curve
import xgboost as xgb
import lightgbm as lgb

Y=data["Y"]
train=data.drop("Y",axis=1)
dummied=pd.get_dummies(train["sex"])
train=train.drop("sex",axis=1)
train=pd.concat([train,dummied],axis=1)

train_x,test_x,train_y,test_y=train_test_split(train,Y,test_size=0.3,random_state=1)

from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=2018) 
X, y = sm.fit_sample(train_x, train_y)
print('通过SMOTE方法平衡正负样本后')
n_sample = y.shape[0]
n_pos_sample = y[y == 0].shape[0]
n_neg_sample = y[y == 1].shape[0]
print('样本个数：{}; 正样本占{:.2%}; 负样本占{:.2%}'.format(n_sample,n_pos_sample / n_sample,n_neg_sample / n_sample))


model_lgb=lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=45, reg_alpha=2, reg_lambda=2,
        max_depth=5, n_estimators=800, objective='binary',
         subsample_freq=1,bagging_fraction= 0.6, feature_fraction= 0.6,
        learning_rate=0.01, min_child_weight=2, random_state=20, n_jobs=4)


model_lgb.fit(X,y)
y_pred=model_lgb.predict_proba(test_x)[:,1]
auc=roc_auc_score(test_y, y_pred)
fpr,tpr,thresholds=roc_curve(test_y, y_pred)
ks=max(tpr-fpr)
#score=cross_val_score(model_xgb,X,y,cv=5,scoring='roc_auc')

#print(score.max(),score.min(),score.mean(),score.std())
print("SMOTE之后的AUC:{}".format(auc))
print("SMOTE之后的KS:{}".format(ks))

precision, recall, _ = precision_recall_curve(test_y, y_pred)
plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])

from sklearn.metrics import average_precision_score
average_precision = average_precision_score(test_y, y_pred)

print('Average precision-recall score: {0:0.2f}'.format(average_precision))
