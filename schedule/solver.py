
"""
This file will model and solve the problem with ILP
"""

import gurobipy as gp
import logging
from gurobipy import GRB
from pprint import pprint as pp
import numpy as np
from collections import defaultdict
import sys

from .cost_graph import *


MAX_FLOAT = 1e20

class Solver(object):

    def __init__(self, g : GraphCost, chip : Chip) -> None:
        self.graph = g
        self.chip = chip
        self.dispatch_results = {}

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

    def __init__(self, g : GraphCost, chip : Chip) -> None:
        super().__init__(g, chip)

    def solve(self):
        for op in self.operations:
            sel_p = None
            min_time = 2**32
            for proc in self.chip.processors:
                c = self.graph.get_op_cost_one_device(op, proc.type)
                if c <= 0:
                    continue

                if c < min_time:
                    min_time = c
                    sel_p = proc

            self.dispatch_results[op] = sel_p

class ILPSolver(Solver):
    def __init__(self, g, chip: Chip) -> None:
        super().__init__(g, chip)
        self.prepare()

    def prepare(self):
        self.avg_cost = {}

        # Create a minimization problem
        self.problem = gp.Model("Minimize_Cost")

        # Define decision variables
        # should consider the parallel
        self.x = {}
        for op in self.graph.topo_sort():
            for h_id in self.chip.ids():
                self.x[op, h_id] = self.problem.addVar(
                    vtype=GRB.INTEGER, lb=0, ub=1, name=f"x_{op}_{h_id}"
                )

        self.st = {}
        for op in self.graph.topo_sort():
            self.st[op] = self.problem.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"st_{op}")

        self.avail_proc = {}
        for op in self.graph.topo_sort():
            for h_id in self.chip.ids():
                self.avail_proc[op, h_id] = self.problem.addVar(
                    vtype=GRB.CONTINUOUS, lb=0, name=f"avail_proc_{op}_{h_id}"
                )

        self.ft = {
            node: self.st[node] + self.cost_of_op(node) for node in self.graph.topo_sort()
        }

        self.limit_resource()

    def limit_resource(self):
        print(self.problem.Params.Threads)
        print(self.problem.Params.OutputFlag)
        print(self.problem.Params.NodeLimit)
        print(self.problem.Params.SolutionLimit)
        print(self.problem.Params.NodefileStart)

        self.problem.Params.Threads = 16
        self.problem.Params.NodefileStart = 1024*32
        self.problem.Params.NodeLimit = 10000
        self.problem.Params.SolutionLimit = 10


    def get_execution_order(self):
        st_node = []
        for node in self.graph.topo_sort():
            st_node.append((self.st[node].X, node))

        sorted_pairs = sorted(st_node, key=lambda x: x[0])
        return [x[1] for x in sorted_pairs]

    def cost_of_op(self, op_idx):
        return gp.quicksum(
            self.x[op_idx, proc.id] * self.graph.get_op_cost_one_device(op_idx, proc.type)
            for proc in self.chip.processors
        )

    def objective_func(self):
        """The objective is to minimize the eft of the last operations"""
        exit_nodes = self.graph.get_exit_ops()
        if len(exit_nodes) == 1:
            self.problem.setObjective(self.ft[exit_nodes[0]])
        else:
            last_node = self.problem.addVar(vtype=GRB.INTEGER, name="x_exit_node")
            for n in exit_nodes:
                self.problem.addConstr(last_node >= self.ft[n])

            self.problem.setObjective(last_node, GRB.MINIMIZE)


    def print_problem(self):
        print("problem objective: ", self.problem.objective)
        print("problem constraints:")
        pp(self.problem.constraints)
        
        print("Start and finish time: ")
        for node in self.operations:
            print(f"{node} st : {self.st[node].value()}")
            print(f"{node} ft : {self.ft[node].value()}")


    def find_parallel_nodes(self):
        """
        return pairs of nodes that has no dependency
        (n1, n2), (n3, n4)
        """
        def __find_unreachable_pairs(graph):
            pairs = []
            nodes = list(nx.topological_sort(graph))
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    node1 = nodes[i]
                    node2 = nodes[j]
                    print(f"{node1} vs {node2}")
                    if not nx.has_path(graph, node1, node2):
                        print(f"add {node1} -> {node2}")
                        pairs.append((node1, node2))
                        continue
                    
            return pairs

        return __find_unreachable_pairs(self.graph.nx_graph)
        
    def add_constraints(self):
        """
        for (n, m) n, and m are parallel
        """

        def constraint_x():
            """
            Constr1: only one processor is assigned to True
            Constr2: processor with -1 as the performance cost should be assigned to False
            """
            for node in self.graph.topo_sort():
                self.problem.addConstr(
                    gp.quicksum(self.x[node, h.id] for h in self.chip.processors) == 1
                )

                cost = self.graph.get_op_cost(node)
                for proc in self.chip.processors:
                    if cost.get_cost_of_device(proc.type) == -1:
                        self.problem.addConstr(self.x[node, proc.id] == 0)


        def constraint_st_ft():
            """
            1. Constraints that node a starts after its previous nodes
            for prev in prevs:
              st[node] >= ft[prev]
            """
            for node in self.graph.topo_sort():
                prevs = self.graph.prevs(node)
                for prev in prevs:
                    cond = self.st[node] >= self.ft[prev]
                    self.problem.addConstr(cond, f"dep_{node}-{prev}")

        def constraint_proc_assign():
            """
            Constraints every node
            if x[n][h] == x[m][h] == 1: 
                ft[n] <= st[m] or ft[m] <= st[n]  // n and m can not be overlapped
            """

            def __max_cost(cost):
                max_float = 0
                for node in cost.keys():
                    max_v = max(list(cost[node].backends.values()))
                    max_float = max(max_v, max_float)
                return max_float

            parallel_nodes = self.find_parallel_nodes()
            print("Parallel nodes:")
            print(parallel_nodes)
            M = __max_cost(self.graph.op_cost) + 100000000.0
            print("Max float is ", M)
            y = {}
            for n, m in parallel_nodes:
                y[n, m] = self.problem.addVar(vtype=GRB.BINARY, name=f"y_{n}_{m}")

            for h in self.chip.processors:
                for n, m in parallel_nodes:
                    yy = y[n, m]
                    xn = self.x[n, h.id]
                    xm = self.x[m, h.id]
                    cond = self.ft[n] - self.st[m] + (2 * M) * (xn + xm - 2) <= M * yy
                    self.problem.addConstr(cond, f"pair_{n}_{m}_{h.id}")

                    cond = self.ft[m] - self.st[n] + (2 * M) * (xn + xm - 2) <= M * (1 - yy)
                    self.problem.addConstr(cond, f"pair_{m}_{n}_{h.id}")

        constraint_x()
        constraint_st_ft()
        constraint_proc_assign()

    def solve(self):
        self.objective_func()
        self.add_constraints()
        self.problem.optimize()
        #self.print_problem()
        self.problem.write("schedule.lp")
        self.get_device_dispatch_results()
        # self.rectify()

    # def print_results(self):
    #     for i in self.operations:
    #         for j in self.hardware.processors:
    #             if self.x[i][j.id].value() == 1:
    #                 print(f"{self.x[i][j.id].value()} Operation {i} assigned to {j}")
    #     logging.info(f"Total cost: {pulp.value(self.problem.objective)}")

    def get_device_dispatch_results(self):
        order = self.get_execution_order()
        for i in order:
            print(i)
            has_dispatched = False
            for j in self.chip.processors:
                if self.x[i, j.id].X == 1:
                    has_dispatched = True
                    self.dispatch_results[i] = j.id

            if not has_dispatched:
                logging.warning(
                    f"Error: {i} is not dispatched, dispatch to the one with minimum cost"
                )
                for i in self.graph.topo_sort():
                    for j in self.chip.processors:
                        pp(f"x of {i}, {j.id} = {self.x[i, j.id].X}")
                exit(-1)


def solveDag(solverType, g: GraphCost, chip : Chip, f=None):
    """
    solverType: solve class, e.g., ILPSolver/BasicSolver/...
    cost: the cost of each operation, e.g.,
        {
            "op1" : {"CPU1" : 1, "CPU2" : 2, "GPU": 3}
            "op2" : {"CPU1" : 1, "CPU2" : 2, "GPU": 3}
            ...
        }
    hardware: Chip([Processor("CPU1", "CPU", "red"),
                          Processor("CPU2", "CPU", "blue"),
                          Processor("GPU", "GPU", "green"),])

    f->str : the output file name
    """
    solver = solverType(g, chip)
    solver.solve()
    if f is not None:
        solver.draw_results(f)

    return solver.dispatch_results
