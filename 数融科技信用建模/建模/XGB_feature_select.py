# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 14:53:15 2018

@author: bdfus001
"""
################################################ xgboost特征选择
import pandas as pd 
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] 
plt.rcParams['axes.unicode_minus']=False

data=pd.read_csv("C:/Users/bdfus001/Desktop/credit-model/preprocess_data/end_data.csv")

from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score,accuracy_score,roc_curve,precision_recall_curve
import xgboost as xgb
import lightgbm as lgb

Y=data["Y"]
train=data.drop("Y",axis=1)
dummied=pd.get_dummies(train["sex"])
train=train.drop("sex",axis=1)
train=pd.concat([train,dummied],axis=1)

train_x,test_x,train_y,test_y=train_test_split(train,Y,test_size=0.3,random_state=1)

model = xgb.XGBClassifier()
model.fit(train, Y)
ax =xgb.plot_importance(model)
fig = ax.figure
fig.set_size_inches(15, 20)
plt.title('Feature under xgboostClassifier', y=1.05, size=15)
plt.savefig("Feature under XGB.jpg", dpi=300)
plt.show()

drop_col=["贷记卡12个月从未有过逾期的账户数","AGENT_huifusdb","HAS_FUND_1.0","HAS_FUND_0.0",
          "lnd_latest_6m_used_avg_amount","scheduled_payment_max","loan_amount_max",
          "HIGHEST_OA_PER_MON","lnd_account_count","WORK_PROVINCE_缺失","EDU_LEVEL_本科",
          "overdue_amount_sum"]

col_new=train.columns.drop(drop_col)
new_train=train[col_new]

#model_xgb = xgb.XGBClassifier(objective= 'binary:logistic',seed=123,learning_rate=0.01,n_estimators=800,
#                        max_depth=8,colsample_bytree=0.8,subsample=0.8,min_child_weight=2,gamma=1,reg_alpha=2,reg_lambda=2)
#
#score=cross_val_score(model_xgb,new_train,Y,cv=5,scoring='roc_auc')
#
#print(score.max(),score.min(),score.mean(),score.std())


train_X,test_X,train_Y,test_Y=train_test_split(new_train,Y,test_size=0.3,random_state=1)
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=2018) 
X, y = sm.fit_sample(new_train,Y)
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
y_pred=model_lgb.predict_proba(test_X)[:,1]
auc=roc_auc_score(test_Y, y_pred)
fpr,tpr,thresholds=roc_curve(test_Y, y_pred)
ks=max(tpr-fpr)
#score=cross_val_score(model_xgb,X,y,cv=5,scoring='roc_auc')

#print(score.max(),score.min(),score.mean(),score.std())
print("SMOTE之后的AUC:{}".format(auc))
print("SMOTE之后的KS:{}".format(ks))

precision, recall, _ = precision_recall_curve(test_Y, y_pred)
plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])

from sklearn.metrics import average_precision_score
average_precision = average_precision_score(test_Y, y_pred)

print('Average precision-recall score: {0:0.2f}'.format(average_precision))