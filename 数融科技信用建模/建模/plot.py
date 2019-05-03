# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 15:51:15 2018

@author: bdfus001
"""

import pandas as pd 
import numpy as np
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
from sklearn.externals.six import StringIO
from sklearn.svm import SVC
import seaborn as sns
from sklearn import tree
import pydotplus
import graphviz
import os

Y=data["Y"]
train=data.drop("Y",axis=1)
dummied=pd.get_dummies(train["sex"])
train=train.drop("sex",axis=1)
train=pd.concat([train,dummied],axis=1)

train_x,test_x,train_y,test_y=train_test_split(train,Y,test_size=0.3,random_state=1)

model_lgb=lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=45, reg_alpha=2, reg_lambda=2,
        max_depth=5, n_estimators=800, objective='binary',
         subsample_freq=1,bagging_fraction= 0.6, feature_fraction= 0.6,
        learning_rate=0.01, min_child_weight=2, random_state=20, n_jobs=4)
model_xgb = xgb.XGBClassifier(objective= 'binary:logistic',seed=123,learning_rate=0.01,n_estimators=800,
                        max_depth=7,colsample_bytree=0.8,subsample=0.8,min_child_weight=2,gamma=1,reg_alpha=2,reg_lambda=2)


#model_lgb.fit(train_x,train_y)
#y_pred=model_lgb.predict_proba(test_x)
#
#importance_score=model_lgb.feature_importances_
#importance_feature=list(train_x.columns)
#
#importance=pd.DataFrame({"feature":importance_feature,"importance_score":importance_score})
#importance.sort_values(by="importance_score",ascending=False,inplace=True)
#
#
#clf = tree.DecisionTreeClassifier(max_depth=7)    
#clf= clf.fit(train_x,train_y)
#y_pred=clf.predict_proba(test_x)[:,1]
#auc=roc_auc_score(test_y, y_pred)
#
#feature_name=train_x.columns
#target_name=["class_0","class_1"]
#dot_data = StringIO()
#tree.export_graphviz(clf, out_file=dot_data, 
#                         feature_names=train_x.columns,  
#                         class_names=target_name,  
#                         filled=True, rounded=True, special_characters=True) 
#
#graph = pydotplus.graph_from_dot_data(dot_data.getvalue().replace("helvetica",'"Microsoft YaHei"'))
#graph.write_pdf("decision_tree_1.pdf")
#print(auc)


"""

entropy_thresholds = np.linspace(0, 1, 100)
gini_thresholds = np.linspace(0, 0.2, 100)
#设置参数矩阵：
param_grid = [{'criterion': ['entropy'], 'min_impurity_decrease': entropy_thresholds},
              {'criterion': ['gini'], 'min_impurity_decrease': gini_thresholds},
              {'max_depth': np.arange(2,10)},
              {'min_samples_split': np.arange(2,30,2)}]
clf = GridSearchCV(tree.DecisionTreeClassifier(), param_grid, cv=5,scoring='roc_auc',n_jobs=4)
clf.fit(train_x,train_y)
print("best param:{0}\nbest score:{1}".format(clf.best_params_, clf.best_score_))
"""


"""
for i in range(10):
    n=i+1
    digraph=lgb.create_tree_digraph(model_lgb, tree_index=n)
    digraph.format = 'pdf'
    digraph.view('plot_tree/lgb_'+str(n))
    #digraph.save("plot_tree/lgb_"+str(n)+".pdf")
"""


##提取前10个重要的变量，使用svm
"""
top_10=list(importance.iloc[:10,:].feature)

train_top_10=train[top_10]
train_top_10_x,test_top_10_x,train_top_10_y,test_top_10_y=train_test_split(train_top_10,Y,test_size=0.3,random_state=1)

clf_rbf = SVC(random_state=0,probability=True)
model_xgb.fit(train_top_10_x, train_top_10_y)

y_pred = model_xgb.predict_proba(test_top_10_x)[:,1]

AUC=roc_auc_score(test_top_10_y,y_pred)

print("AUC的值为:{}".format(AUC))
"""


#### 可能的规则

train_x=pd.concat([train_x,train_y],axis=1)
##guize=train_x[(train_x["WORK_PROVINCE_东北华北"]>0.5) & (train_x["贷记卡总共享总额"]>50510.0) & (train_x["AGENT_missing"]>0.5) &
##              (train_x["FINANCE_CORP_COUNT"] > 8.5) & (train_x["lnd_used_credit_limit"] <= 69147.5)] 
#
###研究生学历以上的只有一个是违约的           28,1    
#edu_level=train_x[train_x['EDU_LEVEL_硕士及以上']==1]
#fig = plt.figure()
#edu_level.Y.value_counts().plot(kind='bar')
#plt.xticks(range(2),['没有违约','违约'])
#plt.xticks(rotation=360)
#plt.title('教育程度是研究生以上的违约情况')
#plt.show()
#
###学历是其他的    200,5
#edu_level_1=train_x[train_x['EDU_LEVEL_其他']==1]
#fig = plt.figure()
#edu_level_1.Y.value_counts().plot(kind='bar')
#plt.xticks(range(2),['没有违约','违约'])
#plt.xticks(rotation=360)
#plt.title('教育程度是其他的违约情况')
#plt.show()
#
#
####婚姻状况   31,2
#hunyin=train_x[train_x['MARRY_STATUS_丧偶']==1]
#hunyin.Y.value_counts().plot(kind='bar')
#plt.xticks(range(2),['没有违约','违约'])
#plt.xticks(rotation=360)
#plt.title('丧偶的违约情况')
#plt.show()
#
######婚姻是其他  97,2
#hunyin_1=train_x[train_x['MARRY_STATUS_其他']==1]
#hunyin_1.Y.value_counts().plot(kind='bar')
#plt.xticks(range(2),['没有违约','违约'])
#plt.xticks(rotation=360)
#plt.title('其他的违约情况')
#plt.show()


neg=train_x[train_x["Y"]==1]






