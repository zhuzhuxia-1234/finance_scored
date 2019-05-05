# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 11:41:09 2019

@author: zhuxibing
"""

from py2neo import Graph,Node,Relationship,NodeMatcher


test_graph= Graph("http://localhost:7474", username="neo4j", password="940205")
 
test_graph.delete_all()

##节点的建立
test_node_1 = Node("Person",name = "test_node_1")
test_node_2 = Node("Person",name = "test_node_2")
test_graph.create(test_node_1)
test_graph.create(test_node_2)

##节点间关系的建立
node_1_call_node_2 = Relationship(test_node_1,'CALL',test_node_2)
node_1_call_node_2['count'] = 1
node_2_call_node_1 = Relationship(test_node_2,'CALL',test_node_1)
node_2_call_node_1['count'] = 2
test_graph.create(node_1_call_node_2)
test_graph.create(node_2_call_node_1)

print(test_node_1, test_node_2, node_2_call_node_1)
##节点/关系的属性赋值以及属性值的更新
#以关系建立里的 node_1_call_node_2 为例，让它的count加1，再更新到图数据库里面。

node_1_call_node_2['count']+=3
test_graph.push(node_1_call_node_2)

#更新属性值就使用push函数来进行更新即可

#通过属性值来查找节点和关系（find,find_one）find和find_one的区别在于：
#find_one的返回结果是一个具体的节点/关系，可以直接查看它的属性和值。如果没有这个节点/关系，返回None。
#find查找的结果是一个游标，可以通过循环取到所找到的所有节点/关系。

#查找节点
find = NodeMatcher(test_graph).match('Person')
for f in find:
    print(f['name'])
    
find1 = NodeMatcher(test_graph).match('Person').first()
print(find1)
    
