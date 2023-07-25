"""
This file will model and solve the problem with ILP
"""
from schedule import *

class Solver(object):

    def __init__(self, nx_graph : nx.DiGraph, cost, hardware : Processors) -> None:
        self.nx_graph = nx_graph
        self.cost = cost
        self.hardware = hardware
        self.dag = {}
        self.pre_dag = {}
        self.operations = None

        self.dispatch_results = {}
        self.prepare()

    def prepare(self):
        self.operations = list(nx.topological_sort(self.nx_graph))

        for i, node in enumerate(self.operations):
            # sucs = list(nx.bfs_successors(self.nx_graph, node, depth_limit=0))[0][1]
            sucs = list(self.nx_graph.successors(self.operations[i]))
            pre = list(self.nx_graph.predecessors(self.operations[i]))

            self.dag[node] = set([s for s in sucs])
            self.pre_dag[node] = set([p for p in pre])


    def get_cost(self, i, p: Processor):
        return self.cost[i][p.type]

    def draw_results(self, pdf_file):
        # Update node assignment
        for i in self.operations:
            for j in self.hardware.processors:
                if self.dispatch_results[i].id == j.id:
                    self.nx_graph.nodes[i]["color"] = j.color

        tmp = nx.nx_agraph.to_agraph(self.nx_graph)  # convert to a graphviz graph
        tmp.draw(pdf_file, prog="dot")  # Draw with pygraphviz
        pass

    def rectify(self):
        for op in self.operations:
            if self.cost[op][self.dispatch_results[op].type] <= 0:
                sel_p = None
                min_time = 2**32
                for p in self.hardware.processors:
                    c = self.cost[op][p.type]
                    if c <= 0:
                        continue

                    if c < min_time:
                        min_time = self.cost[op][p.type]
                        sel_p = p

                self.dispatch_results[op] = sel_p
                print("wrong for ", op)
            



class MinimalSolver(Solver):

    def __init__(self, nx_graph, cost, hardware: Processors) -> None:
        super().__init__(nx_graph, cost, hardware)

    def solve(self):
        for op in self.operations:
            sel_p = None
            min_time = 2**32
            for p in self.hardware.processors:
                c = self.cost[op][p.type]
                if c <= 0:
                    continue

                if c < min_time:
                    min_time = self.cost[op][p.type]
                    sel_p = p

            self.dispatch_results[op] = sel_p
