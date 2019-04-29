# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 11:10:29 2019

@author: zhuxibing
"""

######################################################
#                 Kmeans++聚类
######################################################
from sklearn.datasets import make_blobs
x,y=make_blobs(n_samples=150,n_features=2,centers=3,cluster_std=0.5,shuffle=True,random_state=0)
#scikit中的make_blobs方法常被用来生成聚类算法的测试数据
#直观地说，make_blobs会根据用户指定的特征数量、中心点数量、范围等来生成几类数据，这些数据可用于测试聚类算法的效果。
 
 
 
######################采用scikit-learn中的KMeans################################
from sklearn.cluster import KMeans
km=KMeans(n_clusters=3,init='k-means++',n_init=10,max_iter=300,tol=1e-04,random_state=0)
#将簇的数量设定为3个。设置n_init=10，程序能够基于不同的随机初始中心独立运行算法10次，并从中选择SSE最小的作为最终模型。
#通过max_iter参数，指定算法每轮运行的迭代次数。
#tol=1e-04参数控制对簇内误差平方和的容忍度
#对于sklearn中的k-means算法，如果模型收敛了，即使未达到预定迭代次数，算法也会终止
#注意，将init='random'改为init='k-means++'（默认值）就由k-means算法变为k-means++算法了
y_km=km.fit_predict(x)
 
 
 
############################做可视化处理########################################
import matplotlib.pyplot as plt
plt.scatter(x[y_km == 0, 0],x[y_km == 0, 1],s=50, c='lightgreen',marker='s', edgecolor='black',label='cluster 1')
plt.scatter(x[y_km == 1, 0],x[y_km == 1, 1],s=50, c='orange', marker='o', edgecolor='black',label='cluster 2')
plt.scatter(x[y_km == 2, 0],x[y_km == 2, 1],s=50, c='lightblue',marker='v', edgecolor='black',label='cluster 3')
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],s=250, marker='*',c='red', edgecolor='black',label='centroids')
#簇中心保存在KMeans对象的centers_属性中
plt.legend(scatterpoints=1)
plt.grid()
plt.tight_layout()
plt.show()
