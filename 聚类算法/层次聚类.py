# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 11:46:19 2019

@author: zhuxibing
"""

####################先随机生成一些样本数据用于计算###############################
import numpy as np
np.random.seed(123)
#seed( ) 用于指定随机数生成时所用算法开始的整数值，
#如果使用相同的seed( )值，则每次生成的随即数都相同，如果不设置这个值，则系统根据时间来自己选择这个值，此时每次生成的随机数因时间差异而不同
 
variables=['X','Y','Z']#列表样本的不同特征
labels=['ID_0','ID_1','ID_2','ID_3','ID_4']#5个不同的样本
x=np.random.random_sample([5,3])*10#返回随机的浮点数，在半开区间 [0.0, 1.0)
import pandas as pd
df=pd.DataFrame(x,columns=variables,index=labels)
#DataFrame是Pandas中的一个表结构的数据结构，包括三部分信息，表头（列的名称），表的内容（二维矩阵），索引（每行一个唯一的标记）。
print(df)
 
##########################基于距离矩阵进行层次聚类###############################
#使用SciPy中的子模块spatial.distance中的pdist函数来计算距离矩阵
from scipy.spatial.distance import pdist,squareform
#pdist观测值（n维）两两之间的距离。距离值越大，相关度越小
#squareform将向量形式的距离表示转换成dense矩阵形式。
row_dist = pd.DataFrame(squareform(pdist(df, metric='euclidean')),columns=labels,index=labels)
#基于样本的特征X、Y和Z，使用欧几里得距离计算了样本间的两两距离。
#通过将pdist函数的返回值输入到squareform函数中，获得一个记录成对样本间距离的对称矩阵
print(row_dist) 
 
#使用cluster.hierarchy子模块下的linkage函数。此函数以全连接作为距离判定标准，它能够返回一个关联矩阵
from scipy.cluster.hierarchy import linkage
 
##############################分析聚类结果######################################
#使用通过squareform函数得到的距离矩阵：
row_clusters1 = linkage(pdist(df, metric='euclidean'), method='complete')
df1=pd.DataFrame(row_clusters1,
             columns=['row label 1', 'row label 2',
                      'distance', 'no. of items in clust.'],
             index=['cluster %d' % (i + 1) 
                    for i in range(row_clusters1.shape[0])])
print("使用通过squareform函数得到的距离矩阵：")
print(df1)
 
#使用稠密距离矩阵：
row_clusters2 = linkage(pdist(df, metric='euclidean'), method='complete')
#进一步分析聚类结果
df2=pd.DataFrame(row_clusters2,
             columns=['row label 1', 'row label 2',
                      'distance', 'no. of items in clust.'],
             index=['cluster %d' % (i + 1) for i in range(row_clusters2.shape[0])])
print("使用稠密距离矩阵:")
print(df2)
 
#以矩阵格式的示例数据作为输入：
row_clusters3 = linkage(df.values, method='complete', metric='euclidean')
df3=pd.DataFrame(row_clusters3,
             columns=['row label 1', 'row label 2',
                      'distance', 'no. of items in clust.'],
             index=['cluster %d' % (i + 1)
                    for i in range(row_clusters3.shape[0])])
print("以矩阵格式的示例数据作为输入：")
print(df3)
 
###################采用树状图的形式对聚类结果进行可视化展示#######################
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
 
row_dendr = dendrogram(row_clusters1, labels=labels,)
 
plt.tight_layout()
plt.ylabel('Euclidean distance')
 
plt.show()
