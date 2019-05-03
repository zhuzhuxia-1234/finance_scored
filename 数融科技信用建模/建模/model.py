# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 14:21:46 2018

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

train_x,test_x,train_y,test_y=train_test_split(train,Y,test_size=0.3,random_state=2)
model_xgb = xgb.XGBClassifier(objective= 'binary:logistic',seed=123,learning_rate=0.01,n_estimators=800,
                        max_depth=7,colsample_bytree=0.8,subsample=0.8,min_child_weight=2,gamma=1,reg_alpha=2,reg_lambda=2)

model_lgb=lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=45, reg_alpha=2, reg_lambda=2,
        max_depth=5, n_estimators=800, objective='binary',
         subsample_freq=1,bagging_fraction= 0.6, feature_fraction= 0.6,
        learning_rate=0.01, min_child_weight=2, random_state=20, n_jobs=4)


#model_xgb.fit(train_x, train_y)                         
#y_pred=model_xgb.predict_proba(test_x)


model_lgb.fit(train_x,train_y)
y_pred=model_lgb.predict_proba(test_x)

fpr,tpr,thresholds=roc_curve(test_y, y_pred[:,1])
auc=roc_auc_score(test_y, y_pred[:,1])
print("AUC的值为：{}".format(auc))


"""
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr,tpr)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.ylim([0.0, 1.0])
plt.xlim([0.0, 1.0])
plt.show()
print("xgboost的ks值:{}".format(max(tpr-fpr)))
print("xgboost的auc:{}".format(auc))
"""
"""
fig,ax = plt.subplots(figsize=(20,15))
lgb.plot_importance(model_lgb,ax=ax)
plt.title("Featurertances")
plt.savefig("lightgbm.png")
plt.show()
"""

#score=cross_val_score(model_lgb,train,Y,cv=5,scoring='roc_auc')
#print(score.max(),score.min(),score.mean(),score.std())


###################################################################################################
#                                     lightgbm调参
###################################################################################################

#Step1. 学习率和估计器及其数目
"""
params = {
    'boosting_type': 'gbdt', 
    'objective': 'binary', 
    'learning_rate': 0.1,
    'num_leaves': 50, 
    'max_depth': 6,
    'subsample': 0.8, 
    'colsample_bytree': 0.8
    }

data_train = lgb.Dataset(train, Y, silent=True)
cv_results = lgb.cv(
    params, data_train, num_boost_round=1000, nfold=5, stratified=False, shuffle=True, metrics='auc',
    early_stopping_rounds=50, verbose_eval=50, show_stdv=True, seed=0)

print('best n_estimators:', len(cv_results['auc-mean']))
print('best cv score:', cv_results['auc-mean'][-1])
"""

#Step2. max_depth 和 num_leaves
###粗调
"""
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=50,
                              learning_rate=0.1, n_estimators=40, max_depth=6,
                              metric='rmse', bagging_fraction = 0.8,feature_fraction = 0.8)

params_test1={
    'max_depth': range(3,8,2),
    'num_leaves':range(50, 170, 30)
}
gsearch1 = GridSearchCV(estimator=model_lgb, param_grid=params_test1, scoring='roc_auc', cv=5, verbose=1, n_jobs=4)
gsearch1.fit(train, Y)
print(gsearch1.best_score_, gsearch1.best_params_)

"""
###微调
"""
params_test2={
    'max_depth': [5,6,7],
    'num_leaves':[45,45,50,55,60]
}

gsearch2 = GridSearchCV(estimator=model_lgb, param_grid=params_test2, scoring='roc_auc', cv=5, verbose=1, n_jobs=4)
gsearch2.fit(train, Y)
print(gsearch2.best_params_, gsearch2.best_score_)

"""

#Step3: min_data_in_leaf 和 min_sum_hessian_in_leaf
"""
params_test3={
    'min_child_samples': [18, 19, 20, 21, 22],
    'min_child_weight':[0.001, 0.002]
}
gsearch3 = GridSearchCV(estimator=model_lgb, param_grid=params_test3, scoring='roc_auc', cv=5, verbose=1, n_jobs=4)
gsearch3.fit(train, Y)
print(gsearch3.best_params_, gsearch3.best_score_)
"""

#Step4: feature_fraction 和 bagging_fraction
"""
params_test4={
    'feature_fraction': [0.5, 0.6, 0.7, 0.8, 0.9],
    'bagging_fraction': [0.6, 0.7, 0.8, 0.9, 1.0]
}
gsearch4= GridSearchCV(estimator=model_lgb, param_grid=params_test4, scoring='roc_auc', cv=5, verbose=1, n_jobs=4)
gsearch4.fit(train, Y)
print(gsearch4.best_params_, gsearch4.best_score_)
"""

"""
#Step5: 正则化参数
params_test6={
    'reg_alpha': [0.5,1,2,3,4,5],
    'reg_lambda': [0.5,1,2,3,4,5]
}
gsearch6= GridSearchCV(estimator=model_lgb, param_grid=params_test6, scoring='roc_auc', cv=5, verbose=1, n_jobs=4)
gsearch6.fit(train, Y)
print(gsearch6.best_params_, gsearch6.best_score_)
"""



