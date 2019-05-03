# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 11:43:11 2018

@author: bdfus001
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] 
plt.rcParams['axes.unicode_minus']=False
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score,accuracy_score,roc_curve
import xgboost as xgb
import lightgbm as lgb
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from mlxtend.classifier import StackingClassifier
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from minepy import MINE



data=pd.read_csv("C:/Users/bdfus001/Desktop/credit-model/preprocess_data/end_data.csv")
Y=data["Y"]
train=data.drop("Y",axis=1)
dummied=pd.get_dummies(train["sex"])
train=train.drop("sex",axis=1)
train=pd.concat([train,dummied],axis=1)

train_x,test_x,train_y,test_y=train_test_split(train,Y,test_size=0.3,random_state=1)

gbdt = GradientBoostingClassifier(learning_rate=0.005, n_estimators=2400, max_depth=3, min_samples_split=800,
                                      min_samples_leaf=60, max_features=9, subsample=0.7, random_state=0)
rf = RandomForestClassifier(n_estimators=150, max_depth=13, min_samples_split=100,
                            min_samples_leaf=50, max_features=17, random_state=0)

xgb = xgb.XGBClassifier(objective= 'binary:logistic',seed=123,learning_rate=0.01,n_estimators=800,
                        max_depth=7,colsample_bytree=0.8,subsample=0.8,min_child_weight=2,gamma=1,reg_alpha=2,reg_lambda=2)

lgb=lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=45, reg_alpha=2, reg_lambda=2,
        max_depth=5, n_estimators=800, objective='binary',
         subsample_freq=1,bagging_fraction= 0.6, feature_fraction= 0.6,
        learning_rate=0.01, min_child_weight=2, random_state=20, n_jobs=4)

lr = LogisticRegression()

lr.fit(train_x,train_y)
lr_pred=lr.predict_proba(test_x)[:,1]

lgb.fit(train_x,train_y)
lgb_pred=lgb.predict_proba(test_x)[:,1]


gbdt.fit(train_x,train_y)
gbdt_pred=gbdt.predict_proba(test_x)[:,1]

xgb.fit(train_x,train_y)
xgb_pred=xgb.predict_proba(test_x)[:,1]

y_pred=0.7*lgb_pred+0.15*xgb_pred+0.15*gbdt_pred

auc=roc_auc_score(test_y,y_pred)
print("xgboost+lightgbm+gbdt的加权auc是{}".format(auc))

mine = MINE()
mine.compute_score(lr_pred, xgb_pred)

print("lr和xgb的mic:{}".format(mine.mic()))
"""

class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
   
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict_proba(X[holdout_index])[:,1]
                out_of_fold_predictions[holdout_index, i] = y_pred
                
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
   
  
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict_proba(X)[:,1] for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict_proba(meta_features)[:,1]
    
stacked_averaged_models = StackingAveragedModels(base_models = (gbdt,lgb,rf,xgb),
                                                 meta_model = lr)


scores = cross_val_score(stacked_averaged_models, train.values, Y.values, cv=5, scoring='roc_auc')
print("AUC: %0.6f(+/-%0.6f) " % (scores.mean(), scores.std()))

"""
"""
from heamy.dataset import Dataset
from heamy.estimator import Classifier
from heamy.pipeline import ModelsPipeline


dataset = Dataset(train_x,train_y,test_x)

model_gbdt = Classifier(dataset=dataset, estimator=GradientBoostingClassifier, parameters={'n_estimators': 2400,"learning_rate":0.005,"max_depth":3},name='gbdt')
model_lgb = Classifier(dataset=dataset, estimator=LGBMClassifier, parameters={'n_estimators': 800,"num_leaves":45,"max_depth":5},name='lgb')
model_xgb = Classifier(dataset=dataset, estimator=XGBClassifier, parameters={'n_estimators':800,"learning_rate":0.01,"max_depth":7},name='xgb')

pipeline = ModelsPipeline(model_gbdt,model_lgb,model_xgb)

weights = pipeline.find_weights(roc_auc_score)
result = pipeline.weight(weights)
 
"""












