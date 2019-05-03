# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 12:26:45 2018

@author: bdfus001
"""

import pandas as pd
import numpy as np

contest_basic_test=pd.read_csv("data/contest_basic_test.tsv",sep="\t")
contest_basic_train=pd.read_csv("data/contest_basic_train.tsv",sep="\t")
contest_ext_crd_cd_ln=pd.read_csv("data/contest_ext_crd_cd_ln.tsv",sep="\t")
contest_ext_crd_cd_ln_spl=pd.read_csv("data/contest_ext_crd_cd_ln_spl.tsv",sep="\t")
contest_ext_crd_cd_lnd=pd.read_csv("data/contest_ext_crd_cd_lnd.tsv",sep="\t")
contest_ext_crd_qr_recorddtlinfo=pd.read_csv("data/contest_ext_crd_qr_recorddtlinfo.tsv",sep="\t")
contest_ext_crd_qr_recordsmr=pd.read_csv("data/contest_ext_crd_qr_recordsmr.tsv",sep="\t")
contest_fraud=pd.read_csv("data/contest_fraud.tsv",sep="\t")

contest_ext_crd_cd_lnd_ovd=pd.read_csv("data/contest_ext_crd_cd_lnd_ovd.csv")
contest_ext_crd_hd_report=pd.read_csv("data/contest_ext_crd_hd_report.csv")
contest_ext_crd_is_creditcue=pd.read_csv("data/contest_ext_crd_is_creditcue.csv")
contest_ext_crd_is_ovdsummary=pd.read_csv("data/contest_ext_crd_is_ovdsummary.csv")
contest_ext_crd_is_sharedebt=pd.read_csv("data/contest_ext_crd_is_sharedebt.csv")

print(contest_ext_crd_cd_ln["state"].unique())

normal=contest_ext_crd_cd_ln[contest_ext_crd_cd_ln["state"]=="正常"]
jieqing=contest_ext_crd_cd_ln[contest_ext_crd_cd_ln["state"]=="结清"]
yuqi=contest_ext_crd_cd_ln[contest_ext_crd_cd_ln["state"]=="逾期"]
daizhang=contest_ext_crd_cd_ln[contest_ext_crd_cd_ln["state"]=="呆账"]
zhuanchu=contest_ext_crd_cd_ln[contest_ext_crd_cd_ln["state"]=="转出"]

print(contest_ext_crd_cd_lnd["state"].unique())

normal_lnd=contest_ext_crd_cd_lnd[contest_ext_crd_cd_lnd["state"]=="正常"]
xiaohu_lnd=contest_ext_crd_cd_lnd[contest_ext_crd_cd_lnd["state"]=="销户"]
weijihuo_lnd=contest_ext_crd_cd_lnd[contest_ext_crd_cd_lnd["state"]=="未激活"]
daizhang_lnd=contest_ext_crd_cd_lnd[contest_ext_crd_cd_lnd["state"]=="呆账"]
zhifu_lnd=contest_ext_crd_cd_lnd[contest_ext_crd_cd_lnd["state"]=="止付"]
dongjie_lnd=contest_ext_crd_cd_lnd[contest_ext_crd_cd_lnd["state"]=="冻结"]

print(contest_fraud["Y_FRAUD"].value_counts())
print(contest_fraud["Y_FRAUD"].unique())


#查询原因和查询结构
query_reason_and_org=contest_ext_crd_hd_report[["REPORT_ID","QUERY_REASON","QUERY_ORG"]]                      

##个人信息
inidivial_information=contest_basic_train.drop(["ID_CARD","LOAN_DATE"],axis=1)

#信用提示
creditcue=contest_ext_crd_is_creditcue.drop(["FIRST_LOAN_OPEN_MONTH","FIRST_LOANCARD_OPEN_MONTH","FIRST_SL_OPEN_MONTH","ANNOUNCE_COUNT","DISSENT_COUNT"],axis=1)


#sharedebt
sharedebt=contest_ext_crd_is_sharedebt[contest_ext_crd_is_sharedebt["TYPE_DW"]=="未结清贷款信息汇总"].drop(["TYPE_DW","MAX_CREDIT_LIMIT_PER_ORG","MIN_CREDIT_LIMIT_PER_ORG","USED_CREDIT_LIMIT"],axis=1)

#overdue_summary
overdue_summary=contest_ext_crd_is_ovdsummary[contest_ext_crd_is_ovdsummary["TYPE_DW"]=="贷款逾期"][["REPORT_ID","COUNT_DW","HIGHEST_OA_PER_MON","MAX_DURATION","MONTHS"]]

contest_basic_test["Y"]=-99
contest_basic_test=contest_basic_test[inidivial_information.columns]
all_data=pd.concat([contest_basic_test,inidivial_information],axis=0)
all_data=all_data.merge(query_reason_and_org,on="REPORT_ID",how="left")
all_data=all_data.merge(creditcue,on="REPORT_ID",how="left")   ###39970条记录 ，这里会产生30个缺失值，之后填补
all_data=all_data.merge(sharedebt,on="REPORT_ID",how="left")   ###33733条记录，未结清贷款或贷记卡
all_data=all_data.merge(overdue_summary,on="REPORT_ID",how="left")  ###25404 逾期记录，没有逾期记录


#ln表 每个客户结清的账户数，以及结清总金额
loan_clear=pd.DataFrame()
loan_clear["loan_count"]=jieqing["loan_id"].groupby(jieqing["report_id"]).count()
loan_clear["loan_amount_sum"]=jieqing["credit_limit_amount"].groupby(jieqing["report_id"]).sum()
loan_clear["loan_amount_max"]=jieqing["credit_limit_amount"].groupby(jieqing["report_id"]).max()
loan_clear["loan_amount_mean"]=jieqing["credit_limit_amount"].groupby(jieqing["report_id"]).mean()

##ln表  还在用的贷款的账户的信息
new_contest_ext_crd_cd_ln=pd.DataFrame()
#contest_ext_crd_cd_ln=contest_ext_crd_cd_ln.dropna()
new_contest_ext_crd_cd_ln["loan_count"]=contest_ext_crd_cd_ln["type_dw"].groupby(contest_ext_crd_cd_ln["report_id"]).count()
#ln表 使用的账户数
loan_count_use_rate=contest_ext_crd_cd_ln[["payment_state","report_id"]].dropna()["payment_state"].groupby(contest_ext_crd_cd_ln["report_id"]).count().reset_index(drop=False)

#ln表 合同金额总和
new_contest_ext_crd_cd_ln["loan_amount_sum"]=contest_ext_crd_cd_ln["credit_limit_amount"].groupby(contest_ext_crd_cd_ln["report_id"]).sum()
new_contest_ext_crd_cd_ln["loan_amount_mean"]=contest_ext_crd_cd_ln["credit_limit_amount"].groupby(contest_ext_crd_cd_ln["report_id"]).mean()
new_contest_ext_crd_cd_ln["loan_amount_max"]=contest_ext_crd_cd_ln["credit_limit_amount"].groupby(contest_ext_crd_cd_ln["report_id"]).max()
#ln表 当前逾期金额总和
new_contest_ext_crd_cd_ln["overdue_amount_sum"]=contest_ext_crd_cd_ln["curr_overdue_amount"].groupby(contest_ext_crd_cd_ln["report_id"]).sum()
new_contest_ext_crd_cd_ln["overdue_amount_mean"]=contest_ext_crd_cd_ln["curr_overdue_amount"].groupby(contest_ext_crd_cd_ln["report_id"]).mean()
new_contest_ext_crd_cd_ln["overdue_amount_max"]=contest_ext_crd_cd_ln["curr_overdue_amount"].groupby(contest_ext_crd_cd_ln["report_id"]).max()

#ln表 最大逾期期数
new_contest_ext_crd_cd_ln["overdue_max_cyc"]=contest_ext_crd_cd_ln["curr_overdue_cyc"].groupby(contest_ext_crd_cd_ln["report_id"]).max()


#当前应还金额总和
new_contest_ext_crd_cd_ln["scheduled_payment_sum"]=contest_ext_crd_cd_ln["scheduled_payment_amount"].groupby(contest_ext_crd_cd_ln["report_id"]).sum()
new_contest_ext_crd_cd_ln["scheduled_payment_max"]=contest_ext_crd_cd_ln["scheduled_payment_amount"].groupby(contest_ext_crd_cd_ln["report_id"]).max()
new_contest_ext_crd_cd_ln["scheduled_payment_mean"]=contest_ext_crd_cd_ln["scheduled_payment_amount"].groupby(contest_ext_crd_cd_ln["report_id"]).mean()

#当前实际支付总和
new_contest_ext_crd_cd_ln["actural_payment_sum"]=contest_ext_crd_cd_ln["actual_payment_amount"].groupby(contest_ext_crd_cd_ln["report_id"]).sum()
new_contest_ext_crd_cd_ln["actural_payment_max"]=contest_ext_crd_cd_ln["actual_payment_amount"].groupby(contest_ext_crd_cd_ln["report_id"]).max()
new_contest_ext_crd_cd_ln["actural_payment_mean"]=contest_ext_crd_cd_ln["actual_payment_amount"].groupby(contest_ext_crd_cd_ln["report_id"]).mean()

new_contest_ext_crd_cd_ln["report_id"]=new_contest_ext_crd_cd_ln.index

def last_n_month_overdue_org(payment_state,n):
    data=payment_state[["report_id","payment_state"]]
    data=data.dropna()
    data["payment_state"]=data["payment_state"].apply(lambda x:str(x)[-n:])
    data["count_/"]=data["payment_state"].apply(lambda x:x.count("/"))
    data["count_N"]=data["payment_state"].apply(lambda x:x.count("N"))
    data["count_*"]=data["payment_state"].apply(lambda x:x.count("*"))
    data["count_#"]=data["payment_state"].apply(lambda x:x.count("#"))
    data["count_C"]=data["payment_state"].apply(lambda x:x.count("C"))
    data["count_G"]=data["payment_state"].apply(lambda x:x.count("G"))
    data["count_1"]=data["payment_state"].apply(lambda x:x.count("1"))
    data["count_2"]=data["payment_state"].apply(lambda x:x.count("2"))
    data["count_3"]=data["payment_state"].apply(lambda x:x.count("3"))
    data["count_4"]=data["payment_state"].apply(lambda x:x.count("4"))
    data["count_5"]=data["payment_state"].apply(lambda x:x.count("5"))
    data["count_6"]=data["payment_state"].apply(lambda x:x.count("6"))
    data["count_7"]=data["payment_state"].apply(lambda x:x.count("7"))
    data["non_overdue_count"]=data["count_/"]+data["count_N"]+data["count_C"]+data["count_*"]+data["count_G"]+data["count_#"]
    data["overdue_count"]=n-data["non_overdue_count"]
    
    temp=data.groupby(["report_id","non_overdue_count"],as_index=False).count()
    name = 'last_' + str(n) + '_month_non_overdue_org'
    temp1=temp[temp["non_overdue_count"]==n][["payment_state","report_id"]]
    temp1.rename(columns={"payment_state":name},inplace=True)
    return temp1


last_6_month=last_n_month_overdue_org(contest_ext_crd_cd_ln,6)
last_12_month=last_n_month_overdue_org(contest_ext_crd_cd_ln,12)
last_24_month=last_n_month_overdue_org(contest_ext_crd_cd_ln,24)

new_contest_ext_crd_cd_ln=new_contest_ext_crd_cd_ln.reset_index(drop=True)

new_contest_ext_crd_cd_ln=new_contest_ext_crd_cd_ln.merge(last_6_month,on="report_id",how="left")
new_contest_ext_crd_cd_ln=new_contest_ext_crd_cd_ln.merge(last_12_month,on="report_id",how="left")
new_contest_ext_crd_cd_ln=new_contest_ext_crd_cd_ln.merge(last_24_month,on="report_id",how="left")
new_contest_ext_crd_cd_ln=new_contest_ext_crd_cd_ln.merge(loan_count_use_rate,on="report_id",how="left")

new_contest_ext_crd_cd_ln.rename(columns={"payment_state":"loan_count_use_rate","report_id":"REPORT_ID"},inplace=True)
new_contest_ext_crd_cd_ln=new_contest_ext_crd_cd_ln.fillna(0)

all_data=all_data.merge(new_contest_ext_crd_cd_ln,on="REPORT_ID",how="left")
all_data.to_csv("all_data.csv",index=None)











