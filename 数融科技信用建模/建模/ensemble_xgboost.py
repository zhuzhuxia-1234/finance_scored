# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 10:18:50 2018

@author: bdfus001
"""

###############################################不平衡样本的 Ensemble-XGBoost 模型 
import pandas as pd 
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] 
plt.rcParams['axes.unicode_minus']=False
import xgboost as xgb
import numpy as np
from sklearn import metrics
from imblearn.ensemble import BalanceCascade 
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,roc_curve,precision_recall_curve
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
import matplotlib.pyplot as plt
import lightgbm as lgb

data=pd.read_csv("C:/Users/bdfus001/Desktop/credit-model/preprocess_data/end_data.csv")
Y=data["Y"]
train=data.drop("Y",axis=1)
dummied=pd.get_dummies(train["sex"])
train=train.drop("sex",axis=1)
train=pd.concat([train,dummied],axis=1)

train_x,test_x,train_y,test_y=train_test_split(train,Y,test_size=0.3,random_state=1)
print("正负样本数：正样本数:{}\n负样本数:{}\n正负样本比例:{}".format(train_y.value_counts()[0],train_y.value_counts()[1],train_y.value_counts()[0]/train_y.value_counts()[1]))


#针对不平衡数据，采样生成新的15个数据集

n_subset=15
BC = BalanceCascade(ratio='auto', n_max_subset=n_subset,random_state=123)
train_xBC,train_yBC = BC.fit_sample(train_x,train_y)
#for ii in np.arange(n_subset):
#    print(pd.value_counts(train_yBC[ii,:]))


#可以发现每个数据集中每类均为1308个样本。下面针对每一对数据集，训练一个xgboost分类器。
lgbmodels = []
n_estimator = np.arange(500,800,50)
n_estimator = np.random.choice(n_estimator,n_subset)
max_depth = np.arange(6,10,1)
max_depth = np.random.choice(max_depth,n_subset)
num_leaves=np.arange(40,70,5)
num_leaves= np.random.choice(num_leaves,n_subset)
reg_alpha=np.arange(0,2,0.5)
reg_alpha= np.random.choice(reg_alpha,n_subset)
feature_fraction=[0.6,0.7,0.8,0.9]
feature_fraction=np.random.choice(feature_fraction,n_subset)
bagging_fraction=[0.6,0.7,0.8,0.9]
bagging_fraction=np.random.choice(bagging_fraction,n_subset)


## 随机森林模型
for i in range(n_subset):
    model =lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=num_leaves[i], reg_alpha=reg_alpha[i], reg_lambda=2,
        max_depth=max_depth[i], n_estimators=n_estimator[i], objective='binary',
         subsample_freq=1,bagging_fraction= bagging_fraction[i], feature_fraction= feature_fraction[i],
        learning_rate=0.01, min_child_weight=2, random_state=20, n_jobs=4)

    model.fit(train_xBC[i,:],train_yBC[i,:])
    lgbmodels.append(model)

#随机森林模型训练好后，对测试集进行预测。在预测时，每个随机森林分类器获得一个预测结果。
## 对测试集进行预测

print("模型预测中......")
rfmodels_y = np.zeros((test_x.shape[0],n_subset))
for i in range(n_subset):
    rfmm = lgbmodels[i]
    rf_pre = rfmm.predict_proba(test_x)[:,1]
    rfmodels_y[:,i] = rf_pre
    
y_pred_pro=rfmodels_y.mean(axis=1)
auc=roc_auc_score(test_y, y_pred_pro)
fpr,tpr,thresholds=roc_curve(test_y, y_pred_pro)

print("ensemble_randomforst的AUC:{}".format(auc))
print("ensemble_randomforst的ks值:{}".format(max(tpr-fpr)))

precision, recall, _ = precision_recall_curve(test_y, y_pred_pro)
plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])

from sklearn.metrics import average_precision_score
average_precision = average_precision_score(test_y, y_pred_pro)

print('Average precision-recall score: {0:0.2f}'.format(average_precision))
























