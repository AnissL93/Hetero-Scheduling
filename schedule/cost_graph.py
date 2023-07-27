#!/usr/bin/env python
"""
The basic network structure.
"""
import networkx as nx
import logging
import pandas as pd
import ast
import networkx as nx
from .color_combin import get_color_combination
from .processor import *
import copy

from .processor import Chip, Processor

class OpCost(object):
    def __init__(self, b) -> None:
        # make all cost to int
        self.backends = {}
        for k,v in b.items():
            self.backends[k] = int(v)

    def get_backend_ids(self):
        return list(self.backends.keys())

    def get_cost_of_device(self, d):
        if not d in self.get_backend_ids():
            assert False
        else:
            return self.backends[d]

class GraphCost(object):
    def __init__(self, g : nx.DiGraph, cost = {}, types = {}):
        # The graph structure
        self.nx_graph = g
        self.topo_sort_ops = list(nx.topological_sort(self.nx_graph))

        # The cost of a operation: {"op_name" : {"CPU": 1, "GPU": 2}}
        self.op_cost = cost
        self.op_types = types

    def topo_sort(self):
        return self.topo_sort_ops

    def to_graphviz(self, pdf_file):
        tmp = nx.nx_agraph.to_agraph(self.nx_graph)  # convert to a graphviz graph
        tmp.draw(pdf_file, prog="dot")  # Draw with pygraphviz

    def get_exit_ops(self):
        return [n for n in self.topo_sort_ops if self.is_exit(n)]

    def get_entry_ops(self):
        return [n for n in self.topo_sort_ops if self.is_entry(n)]

    def is_entry(self, op):
        return len(self.prevs(op)) == 0

    def is_exit(self, op):
        return len(self.sucs(op)) == 0

    def sucs(self, op: str) -> list:
        return list(self.nx_graph.successors(op))

    def prevs(self, op: str) -> list:
        return list(self.nx_graph.predecessors(op))

    def get_op_type(self, id):
        if isinstance(id, int):
            name = self.topo_sort_ops[id]
            return self.op_types[name]

        elif isinstance(id, str):
            return self.op_types[id]

        else:
            logging.error(f"Fail to find type for {id}")
            assert False

    def get_op_cost(self, id):
        if isinstance(id, int):
            name = self.topo_sort_ops[id]
            return self.op_cost[name]

        elif isinstance(id, str):
            return self.op_cost[id]

        else:
            logging.error(f"Fail to find cost for {id}")
            assert False
        
    def get_op_cost_one_device(self, id, d):
        return self.get_op_cost(id).get_cost_of_device(d)

    def draw_graph_structure(self, pdf_file):
        tmp = nx.nx_agraph.to_agraph(self.nx_graph)
        tmp.draw(pdf_file, prog="dot")

    def to_df(self):
        backends = self.get_op_cost(0).get_backend_ids()
        columns = ["op_id", "op_type", "suc"] + backends
        data = []
        for op in self.topo_sort():
            data.append([
                op, 
                self.get_op_type(op),
                self.sucs(op)
                ]
                + 
                [self.get_op_cost_one_device(op, d) for d in backends]
                )

        df = pd.DataFrame(data, columns=columns)
        return df

    def to_csv(self, file_name):
        """Dump the file to CSV format, converting with pandas."""
        self.to_df().to_csv(file_name)


class DispatchedGraph(GraphCost):
    def __init__(self, graph : GraphCost = None, dispatch = {}):
        if graph is not None:
            self.__dict__.update(graph.__dict__)
            self.dispatch_results = dispatch

    def set_dispatch(self, n, p: Processor):
        """
        Dispatch node n to Processor p
        """

        assert n in self.topo_sort()
        self.dispatch_results[n] = p

    def validate(self):
        for k, v in self.dispatch_results.items():
            if v is None:
                print(f"Error: node {k} is not dispatched!")
                return False

        return True


    def draw_results(self, chip : Chip, pdf_file):
        # Update node assignment
        col = get_color_combination(len(chip.processors))
        for i in self.get_exec_order():
            for index, j in enumerate(chip.processors):
                if self.dispatch_results[i].id == j.id:
                    self.nx_graph.nodes[i]["color"] = col[index]
                    self.nx_graph.nodes[i]["shape"] = "Mrecord"
                    self.nx_graph.nodes[i]["style"] = "filled"
                    self.nx_graph.nodes[i]["fontname"] = "Helvetica"

        
        # add a legend for graph

        legend_head = "<<TABLE BORDER=\"0\" CELLBORDER=\"1\" CELLSPACING=\"0\" CELLPADDING=\"4\"> <TR> <TD COLSPAN=\"1\"><B>Legend</B></TD> </TR>"
        for i, p in enumerate(chip.processors):
            legend_head += "<TR><TD BGCOLOR=\"{}\">{}</TD></TR>\n".format(col[i], p.id)

        legend_head += "</TABLE>>"

        tmp_graph = copy.deepcopy(self.nx_graph)
        tmp_graph.add_node("Legend", label = legend_head, shape = "box")

        # tmp_graph.add_edge("Legend", self.get_exec_order()[0], style="invis")
        tmp = nx.nx_agraph.to_agraph(tmp_graph)  # convert to a graphviz graph
        tmp.draw(pdf_file, prog="dot")  # Draw with pygraphviz

    def get_dispatch(self, n):
        assert n in self.topo_sort() 
        return self.dispatch_results[n]

    def get_exec_order(self):
        return list(self.dispatch_results.keys())

    def dispatch_to_df(self):
        assign = [self.dispatch_results[n] for n in self.get_exec_order()]
        data = {
            "op_id" : self.get_exec_order(),
            "dispatch" : assign
        }
        ddf = pd.DataFrame(data)
        return ddf

    def dispatch_to_csv(self, dispatch_csv_file):
        gdf = self.dispatch_to_df()
        gdf.to_csv(dispatch_csv_file)

        
def read_csv(csv_file, dispatch_file = None, chip = None):
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
        types = {}
        for i in range(len(df)):
            node = df.loc[i]
            node_id = str(node["op_id"])

            node_type = str(node["op_type"])
            types[node_id] = node_type

            suc_nodes = list(ast.literal_eval(node["suc"]))
            suc_nodes = [str(i) for i in suc_nodes]

            net.add_node(node_id, label=str(node_type))
            for suc in suc_nodes:
                net.add_edge(node_id, suc)

            this_cost = {}
            for p in all_processors:
                this_cost[p] = node[p]

            cost[node_id] = OpCost(this_cost)


        return net, cost, types 

    df = pd.read_csv(csv_file)
    net, cost, types = _parse_cost(df)
    graph = GraphCost(net, cost, types)
    
    if dispatch_file is None:
        return graph
    else:
        # parse dispatch file
        df = pd.read_csv(dispatch_file)
        dispatch = {}
        for i in range(len(df)):
            dispatch[str(df.loc[i]["op_id"])] = chip.get_processor_by_id(df.loc[i]["dispatch"])

        dgraph = DispatchedGraph(graph, dispatch)
        return dgraph
    
if __name__ == "__main__":
    graph = read_csv("../data/net_perf/bst/bert_with_shape.csv")
    print(graph.to_df())