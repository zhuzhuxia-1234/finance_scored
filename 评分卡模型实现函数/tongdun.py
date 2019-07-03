# -*- coding: utf-8 -*-
"""
Created on Wed May 29 09:35:28 2019

@author: heye
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 10:12:38 2019

@author: heye
"""

import pandas as pd
import numpy as np
import math
df_new = pd.read_excel('C:/Users/zhuxibing/Desktop/overseas_new.xls')
df_re = pd.read_excel('C:/Users/zhuxibing/Desktop/overseas_reapply.xls')
df_all = pd.read_excel('C:/Users/zhuxibing/Desktop/tongdun_test.xlsx')

df_neww = df_new.drop(['apply_date','full_name','user_mobile'],axis=1)
df_ree = df_re.drop(['apply_date','full_name','user_mobile'],axis=1)


data_new = pd.merge(df_neww, df_all,  on='loan_num',how='inner')
date_re =  pd.merge(df_re, df_all,  on='loan_num',how='inner')

date_neww = data_new.drop(['credential_no_x','credential_no_y','user_mobile','real_name'],axis=1)
date_ree = date_re.drop(['apply_date','user_mobile_x','user_mobile_y','credential_no_x','credential_no_y','full_name','real_name'],axis=1)

#nv = date_neww.shape[1]
#nv_list = list(date_neww.columns[0:nv+1])
#for i in nv_list:
##    date_neww[i] = date_neww[i].convert_objects(convert_numeric=True)
##    date_ree[i] = date_ree[i].convert_objects(convert_numeric=True)
#    date_neww[i] = date_neww[i].astype(float)
#    date_ree[i] = date_ree[i].astype(float)
#    print(date_neww[i].dtype)

tongdun_t0_new = date_neww.drop(['loan_num','overduet3','overduet10'],axis=1)
tongdun_t3_new = date_neww.drop(['loan_num','overduet0','overduet10'],axis=1)
tongdun_t10_new = date_neww.drop(['loan_num','overduet0','overduet0'],axis=1)

tongdun_t0_re = date_ree.drop(['loan_num','overduet3','overduet10'],axis=1)
tongdun_t3_re = date_ree.drop(['loan_num','overduet0','overduet10'],axis=1)
tongdun_t10_re = date_ree.drop(['loan_num','overduet0','overduet0'],axis=1)

def auto_iv(df,y,n,filename):
   
    Y=df[y].astype(int)
    df=df.drop([y],axis=1)
    n_var = df.shape[1]
    var_list = list(df.columns[0:n_var])
    df_summary = pd.DataFrame(columns = ["variable", "iv", "total", "bad", "good", "bad_ratio"]) 
    #df_empty = pd.DataFrame({"variable":"", "iv":"", "total":"", "bad":"", "good":"", "bad_ratio":""},index=["0"]) 
    
    for i in range(n_var):
        #i=4
        print(i)
        a=var_list[i]
        X=df[a]
        #  
        X.name=var_list[i]
        #X=X.replace('(null)',0)
        #X=X.apply(pd.to_numeric, errors='ignore')
        if_str= X.dtype!='int' and X.dtype!='float' #and X.dtype!='O'
        print(X.name)    
        
#        if X.dtype!='int' and X.dtype!='float' or X.value_counts().count()<=1.5*n:
#            X=X.astype(str)
#            d1 = pd.DataFrame({X.name: X, "Y": Y}) 
#            df_all = d1.pivot_table(index=X.name, values='Y', aggfunc='count')
#            df_bad = d1.pivot_table(index=X.name, values='Y', aggfunc='sum')            
        
#        else:
#        X=X.astype(float)
        d1 = pd.DataFrame({"X": X, "Y": Y,  X.name: pd.qcut(X, n,duplicates='drop')}) 
        df_all = d1.pivot_table(index=X.name, values='Y', aggfunc='count')
        df_bad = d1.pivot_table(index=X.name, values='Y', aggfunc='sum')  
            

            
        df_all=df_all.rename(columns={'Y':'total'})
        df_bad=df_bad.rename(columns={'Y':'bad'})   
         
        df00 = df_all.join(df_bad, how='inner')
        df00['good'] = df00['total']-df00['bad']   
        total_good=df00['good'].sum()
        total_bad=df00['bad'].sum()
        total=df00['total'].sum()
        df00['bad_ratio'] = df00['bad']/df00['total']
        df00['coverage'] = df00['total']/df00['total'].sum()
        
        
        df00['woe']= ((df00['good']/total_good)/(df00['bad']/total_bad)).apply(lambda x:math.log(max(x,0.000001)))
        df00['woe'][np.isinf(df00['woe'])]=0
        df00['woe'][np.isnan(df00['woe'])]=0
        df00['woe']=df00['woe'].fillna(0)

        
        df00['c'] = ((df00['good']/total_good)-(df00['bad']/total_bad))
        df00['iv'] = df00['woe']*df00['c']
        
        
        df0 = pd.DataFrame({"variable":X.name,'iv':[df00['iv'].sum()],'total':total,'bad':total_bad,'good':total_good,'bad_ratio':total_bad/total})
        #df00 = df00.append(df_empty,ignore_index=True)
        df_summary = df_summary.append(df0,ignore_index=True)
        df_empty = pd.DataFrame(["=" * 50 + '我是分割线' + "="*50])
        
        df00.to_csv(filename+'.csv', mode='a', encoding='utf_8_sig')
        df_empty.to_csv(filename+'.csv', mode='a', encoding='utf_8_sig')
                
    df_summary.to_csv(filename+'.csv', mode='a', encoding='utf_8_sig')
n=20
filename1 = 'tongdun_t0_new'
filename2 = 'tongdun_t3_new'
filename3 = 'tongdun_t10_new'
filename4 = 'tongdun_t0_re'
filename5 = 'tongdun_t3_re'
filename6 = 'tongdun_t10_re'


auto_iv(tongdun_t0_new,'overduet0',n,filename1)
auto_iv(tongdun_t3_new,'overduet3',n,filename2)
auto_iv(tongdun_t10_new,'overduet3',n,filename3)
auto_iv(tongdun_t0_re,'overduet0',n,filename4)
auto_iv(tongdun_t3_re,'overduet3',n,filename5)
auto_iv(tongdun_t10_re,'overduet10',n,filename6)


def auto_select_bin(df, y, n, br, size,filename):

    list11=df.columns
    for a in range(len(df.columns)-1):
        
        if df[list11[a]].value_counts().count()<2:
            print(list11[a])
            print(df[list11[a]].value_counts().count())
            df=df.drop(list11[a],axis=1)
            print('success')

    Y=df[y].astype(int)
    dfx=df.drop([y],axis=1)
    n_var=dfx.shape[1]
    var_list = list(dfx.columns[0:n_var]) #列名 表
    n_varible = len(var_list)#列总数
    sample_size = Y.count()
    sample_bad = Y.sum()
    
    list1 = var_list
    df_summary = pd.DataFrame(columns = ["variable_x","variable_z", "iv", "total", "bad", "good", "bad_ratio"])
    var_bin=[]
    selected_bin=[]


    for i in range(n_varible):
        a=list1[i]
        X=df[a]
        X.name=list1[i]    #遍历列名 取第一个列变量
        for j in range(n_varible-1):
            b=list1[j]
            Z=df[b]
            Z.name=list1[j]    #继续遍历列名 取第二个列变量

            if j!=i and X.name+Z.name not in var_bin and Z.name+X.name not in var_bin:
            

                print(i)
                print(j)
    
#                if Z.dtype=='int' or Z.dtype=='float':
#                    if X.dtype=='int' or X.dtype=='float':
                X=X.astype(float)
                Z=Z.astype(float)
                d1 = pd.DataFrame({"X": X, "Z": Z, "Y": Y, "Group_"+X.name: pd.qcut(X, n,duplicates='drop'), "Group_"+Z.name: pd.qcut(Z, n,duplicates='drop')}) 
                df_all1 = d1.pivot_table(index="Group_"+X.name, columns="Group_"+Z.name, values='Y', aggfunc='count')
                df_bad1 = d1.pivot_table(index="Group_"+X.name, columns="Group_"+Z.name, values='Y', aggfunc='sum')  
           
#                    else:
#                        Z=Z.astype(float)
#                        X=X.fillna("missing").astype(str)
#                        d1 = pd.DataFrame({"X": X, "Z": Z, "Y": Y,  "Group_"+Z.name: pd.qcut(Z, n,duplicates='drop')}) 
#                        df_all1 = d1.pivot_table(index=X, columns="Group_"+Z.name, values='Y', aggfunc='count')
#                        df_bad1 = d1.pivot_table(index=X, columns="Group_"+Z.name, values='Y', aggfunc='sum')            
#                              
#                else:       
#                    if X.dtype=='int' or X.dtype=='float':
#                        Z=Z.fillna("missing").astype(str)
#                        X=X.astype(float)
#                        d1 = pd.DataFrame({"X": X, "Z": Z, "Y": Y, "Group_"+X.name: pd.qcut(X, n,duplicates='drop')}) 
#                        df_all1 = d1.pivot_table(index="Group_"+X.name, columns=Z, values='Y', aggfunc='count')
#                        df_bad1 = d1.pivot_table(index="Group_"+X.name, columns=Z, values='Y', aggfunc='sum')                  
#             
#                    else:
#                        X=X.fillna("missing").astype(str)
#                        Z=Z.fillna("missing").astype(str)
#                        d1 = pd.DataFrame({"X": X, "Z": Z, "Y": Y}) 
#                        df_all1 = d1.pivot_table(index=X, columns=Z, values='Y', aggfunc='count')
#                        df_bad1 = d1.pivot_table(index=X, columns=Z, values='Y', aggfunc='sum')   
                        
                df_all1.index = df_all1.index.astype("object")   
                df_bad1.index = df_bad1.index.astype("object")
                df_all1.loc['Row_sum'] = df_all1.apply(lambda x: x.sum())
                df_bad1.loc['Row_sum'] = df_bad1.apply(lambda x: x.sum())
                                      
                df_all1.columns = df_all1.columns.astype("object")
                df_bad1.columns = df_bad1.columns.astype("object")
                df_all1['Col_sum'] = df_all1.apply(lambda x: x.sum(),axis=1)
                df_bad1['Col_sum']= df_bad1.apply(lambda x: x.sum(),axis=1)    
                df3 = pd.DataFrame([".=" * 30+X.name+'&'+Z.name + '坏账率' + "="*30])
                
                df_bad_ratio1=df_bad1/df_all1
                df_sample_ratio = df_all1/sample_size
                for a in range(len(df_bad1.index)-1):
                    for b in range(len(df_bad1.columns)-1):
                        if df_bad_ratio1.iloc[a,b]>=br and (a>=len(df_bad1.index)-2 or a<1) and (b>=len(df_bad1.columns)-2 or b<1)  and df_all1.iloc[a,b]>=size and X.name+Z.name not in selected_bin:
                                                     
                            df1 = pd.DataFrame([".=" * 30 + '样本数' + "="*30])           
                            df2 = pd.DataFrame([".=" * 30 + '坏样本数' + "="*30])     
                            df4 = pd.DataFrame([".=" * 30 + '覆盖率' + "="*30])
                            
                            
                            print (df3)
                            print (df_bad_ratio1)   
                      
                            df1.to_csv(filename+'.csv', mode='a',encoding="utf_8_sig")
                            df_all1.to_csv(filename+'.csv', mode='a',encoding="utf_8_sig")
                            
                            df2.to_csv(filename+'.csv', mode='a',encoding="utf_8_sig")
                            df_bad1.to_csv(filename+'.csv', mode='a',encoding="utf_8_sig")
                          
                            df3.to_csv(filename+'.csv', mode='a',encoding="utf_8_sig")
                            df_bad_ratio1.to_csv(filename+'.csv', mode='a',encoding="utf_8_sig")
                          
                            df4.to_csv(filename+'.csv', mode='a',encoding="utf_8_sig")
                            df_sample_ratio.to_csv(filename+'.csv', mode='a',encoding="utf_8_sig")
                            
                            selected_bin.append(X.name+'&'+Z.name)
                #df0 = pd.DataFrame(columns = ["variable_x","variable_z", "total", "bad",  "bad_ratio"])
                #df0.variable_x=X.name
                #df0.variable_z=Z.name
                #df0.total=df_all1.iloc[df_bad_ratio1.iloc[a,b].index]
                #df_summary = df_summary.append(df0,ignore_index=True)
                var_bin.append(X.name+'&'+Z.name)

size=100


filename1 = 'tongdun_t0_new'
filename2 = 'tongdun_t3_new'
filename3 = 'tongdun_t10_new'
filename4 = 'tongdun_t0_re'
filename5 = 'tongdun_t3_re'
filename6 = 'tongdun_t10_re'  

m=20

auto_select_bin(tongdun_t0_new, 'overduet0', m, 0.4, size,filename1)  
