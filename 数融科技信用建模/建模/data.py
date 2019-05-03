# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 14:34:50 2018

@author: bdfus001
"""

import pandas as pd 
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] 
plt.rcParams['axes.unicode_minus']=False

all_data=pd.read_csv("C:/Users/bdfus001/Desktop/credit/all_data.csv")
contest_basic_test=pd.read_csv("C:/Users/bdfus001/Desktop/credit/data/contest_basic_test.tsv",sep="\t")
contest_fraud=pd.read_csv("C:/Users/bdfus001/Desktop/credit/data/contest_fraud.tsv",sep="\t")
data=pd.read_csv("C:/Users/bdfus001/Desktop/credit-model/preprocess_data/lnd_features.csv")
sex=pd.read_csv("C:/Users/bdfus001/Desktop/credit-model/preprocess_data/end_data.csv")
data.rename(columns={"report_id":"REPORT_ID"},inplace=True)


all_data=all_data.merge(data,on='REPORT_ID',how="left")
all_data["信贷账户授信总额"]=all_data["贷记卡总授信额度"]+all_data["loan_amount_sum"]
all_data["信贷账户应还金额"]=all_data["scheduled_payment_sum"]+all_data["贷记卡当前本月应还总额"]
all_data["信贷账户实际还款金额"]=all_data["贷记卡当前本月实还总额"]+all_data["actural_payment_sum"]

missing_values=pd.DataFrame({"missing_count":all_data.isnull().sum(),"missing_rate":all_data.isnull().sum()/40000,"feature":all_data.columns})
missing_values.sort_values(by="missing_rate",ascending=False,inplace=True)

missing_delete=missing_values[(missing_values["missing_rate"])>0.9].feature
all_data=all_data.drop(missing_delete,axis=1)

##数据基本结构
all_data_info=all_data.info()
all_data_desc=all_data.describe()

#按行统计缺失值
all_data["missing_count"]=all_data.isnull().sum(axis=1)

##去掉行缺失值过多的样本
#all_data=all_data[all_data["missing_count"]<50]


train_data=all_data[all_data["Y"]!=-99]
#print(train_data["Y"].value_counts())

missing_values=pd.DataFrame({"missing_count":train_data.isnull().sum(),"missing_rate":train_data.isnull().sum()/40000})
missing_values.sort_values(by="missing_rate",ascending=False,inplace=True)

#类别变量编
train_data["daymax"]=sex["信贷账户最大账龄"]
train_data["daymin"]=sex["信贷账户最小账龄"]
train_data["sex"]=sex["sex"]
train_data["sex"]=train_data["sex"].map({1:'male',0:"female"})
train_data["MARRY_STATUS"]=train_data["MARRY_STATUS"].replace("离异","离婚")
train_data["HAS_FUND"]=train_data["HAS_FUND"].fillna(0)
train_data["HAS_FUND"]=train_data["HAS_FUND"].replace(0,"T")
train_data["HAS_FUND"]=train_data["HAS_FUND"].replace(1,"F")
train_data["EDU_LEVEL"]=train_data["EDU_LEVEL"].fillna("missing")
train_data["EDU_LEVEL"]=train_data["EDU_LEVEL"].replace("高中","专科及以下")
train_data["EDU_LEVEL"]=train_data["EDU_LEVEL"].replace("专科","专科及以下")
train_data["EDU_LEVEL"]=train_data["EDU_LEVEL"].replace("初中","专科及以下")
train_data["EDU_LEVEL"]=train_data["EDU_LEVEL"].replace("硕士研究生","硕士及以上")
train_data["EDU_LEVEL"]=train_data["EDU_LEVEL"].replace("博士研究生","硕士及以上")
train_data["AGENT"]=train_data["AGENT"].fillna("missing")


train_data["WORK_PROVINCE"][train_data["WORK_PROVINCE"]<250000]=1
train_data["WORK_PROVINCE"][(train_data["WORK_PROVINCE"]<=350000) & (train_data["WORK_PROVINCE"]>250000)]=2
train_data["WORK_PROVINCE"][train_data["WORK_PROVINCE"]>350000]=3
train_data["WORK_PROVINCE"]=train_data["WORK_PROVINCE"].fillna("missing")
#print(train_data["EDU_LEVEL"].unique())


dummies_list=["IS_LOCAL","QUERY_ORG","QUERY_REASON","MARRY_STATUS","HAS_FUND","EDU_LEVEL","AGENT","WORK_PROVINCE",'sex']
dummied=pd.get_dummies(train_data[dummies_list])

Y=train_data["Y"]
train_data=train_data.drop(dummies_list,axis=1)
train_data=train_data.drop(["REPORT_ID","Y"],axis=1)

train=pd.concat([train_data,dummied],axis=1)


from sklearn.model_selection import train_test_split,GridSearchCV,KFold,cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score,accuracy_score
import xgboost as xgb


train_x,test_x,train_y,test_y=train_test_split(train,Y,test_size=0.3,random_state=1)



##xgboost 默认参数下的auc,accuracy

clf = xgb.XGBClassifier(objective= 'binary:logistic',seed=123)
"""
clf.fit(train_x,train_y)
y_pred_pro=clf.predict_proba(test_x)[:,1]
y_pred=clf.predict(test_x)
auc=roc_auc_score(test_y, y_pred_pro)
accuracy=accuracy_score(test_y,y_pred)

fig, ax = plt.subplots(1,1,figsize=(15,20))
xgb.plot_importance(clf,ax=ax)
plt.show()
print("xgboost默认参数的AUC：%f" % auc)
"""

###5折交叉验证
score=cross_val_score(clf,train,Y,cv=5,scoring='roc_auc')
print(score.max(),score.min(),score.mean().score.std())
