#!/usr/bin/env python
"""
Parse and dump csv file containing network structure and performance data.
"""

import pandas as pd
import networkx as nx
from schedule.cost_graph import Network
import ast

def _parse_cost(df: pd.DataFrame):
    """
    op_id,op_type,suc,PType1_0,PType2_1...
    conv1,Conv,[relu1,conv2],10,20,...
    """
    # all_processors: ["CPU", "GPU"]
    # net: nx.DiGraph
    # cost: {"node1" : {"CPU": 10, "GPU": 15}, "node2": {"CPU": 20, "GPU": 15}}

    all_processors = df.columns[4:]

    net = nx.DiGraph()
    cost = {}
    for i in range(len(df)):
        node = df.loc[i]
        node_id = str(node["op_id"])
        node_type = str(node["op_id"])
        suc_nodes = list(ast.literal_eval(node["suc"]))
        suc_nodes = [str(i) for i in suc_nodes]

        net.add_node(node_id, label=str(node["op_type"]))
        for suc in suc_nodes:
            net.add_edge(node_id, suc)

        cost[node_id] = {}
        for p in all_processors:
            cost[node_id][p] = node[p]

    return net, cost 


def parse_perf_time(csv_file) -> Network:
    df = pd.read_csv(csv_file)
    
    net, cost = _parse_cost(df)

    network = Network()
    network.nx_graph = net
    network.exec

    pass