# -*- coding: utf-8 -*-
"""
Created on Wed May  8 16:39:31 2019

@author: zhuxibing
"""

from py2neo import Node, Graph, Relationship,NodeMatcher


class DataToNeo4j(object):
    """将excel中数据存入neo4j"""

    def __init__(self):
        """建立连接"""
        link = Graph("http://localhost:7474", username="neo4j", password="940205")
        self.graph = link
        # 定义label
        self.invoice_name = '发票名称'
        self.invoice_value = '发票值'
        self.graph.delete_all()

    def create_node(self, node_list_key, node_list_value):
        """建立节点"""
        for name in node_list_key:
            name_node = Node(self.invoice_name, name=name)
            self.graph.create(name_node)
        for name in node_list_value:
            value_node = Node(self.invoice_value, name=name)
            self.graph.create(value_node)

    def create_relation(self, df_data):
        """建立联系"""

        m = 0
        for m in range(0, len(df_data)):
            try:
                matcher = NodeMatcher(self.graph)
                rel = Relationship(matcher.match(self.invoice_name, name=df_data['name'][m]).first(),
                                   df_data['relation'][m], 
                                  matcher.match(self.invoice_value, name=df_data['name2'][m]).first())
                self.graph.create(rel)
            except AttributeError as e:
                print(e, m)