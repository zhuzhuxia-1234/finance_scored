# -*- coding: utf-8 -*-
"""
Created on Sun May  5 11:26:16 2019
# -*- coding: utf8 -*-
@author: zhuxibing
"""
# =============================================================================
# 
# 我们将关注的数据集来自航空业。它有一些关于航线的基本信息。有某段旅程的起始点和目的地。
# 还有一些列表示每段旅程的到达和起飞时间。如你所想，这个数据集非常适合作为图进行分析。
# 想象一下通过航线（边）连接的几个城市（节点）。如果你是航空公司，你可以问如下几个问题：
# 
#  
# 
# 从A到B的最短途径是什么？分别从距离和时间角度考虑。
# 
# 有没有办法从C到D？
# 
# 哪些机场的交通最繁忙？
# 
# 哪个机场位于大多数其他机场“之间”？这样它就可以变成当地的一个中转站。
# 
# =============================================================================


import pandas as pd
import numpy as np

data=pd.read_csv("C:/Users/zhuxibing/Desktop/Airlines.csv")

#预计出发时间，实际出发时间，实际到达时间。预计到达时间转化
data['std'] = data.sched_dep_time.astype(str).str.replace('(\d{2}$)', '') + ':' + data.sched_dep_time.astype(str).str.extract('(\d{2}$)', expand=False) + ':00'
data['sta'] = data.sched_arr_time.astype(str).str.replace('(\d{2}$)', '') + ':' + data.sched_arr_time.astype(str).str.extract('(\d{2}$)', expand=False) + ':00'
data['atd'] = data.dep_time.fillna(0).astype(np.int64).astype(str).str.replace('(\d{2}$)', '') + ':' + data.dep_time.fillna(0).astype(np.int64).astype(str).str.extract('(\d{2}$)', expand=False) + ':00'
data['ata'] = data.arr_time.fillna(0).astype(np.int64).astype(str).str.replace('(\d{2}$)', '') + ':' + data.arr_time.fillna(0).astype(np.int64).astype(str).str.extract('(\d{2}$)', expand=False) + ':00'

data['date'] = pd.to_datetime(data[['year', 'month', 'day']])
data = data.drop(['year', 'month', 'day'],axis=1)

import networkx as nx
FG = nx.from_pandas_dataframe(data, source='origin', target='dest', edge_attr=True)

print(FG.nodes())
print(FG.edges())
nx.draw_networkx(FG, with_labels=True)

print(nx.algorithms.degree_centrality(FG))
print(nx.density(FG))
print(nx.average_shortest_path_length(FG))
print(nx.average_degree_connectivity(FG))
# =============================================================================
# 
# 假如想要计算2个机场之间的最短路线。我们可以想到几种方法：
# 
# 距离最短的路径。
# 
# 飞行时间最短的路径。
# =============================================================================
for path in nx.all_simple_paths(FG, source='JAX', target='DFW'):
    print(path)


dijpath = nx.dijkstra_path(FG, source='JAX', target='DFW')
print(dijpath)


shortpath = nx.dijkstra_path(FG, source='JAX', target='DFW', weight='air_time')
print(shortpath)







