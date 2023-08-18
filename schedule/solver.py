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
    def __init__(self, g: GraphCost, chip: Chip) -> None:
        self.graph = DispatchedGraph(g)
        self.chip = chip

    def rectify(self):
        for op in self.operations:
            if self.cost[op][self.graph.get_dispatch(op).type] <= 0:
                sel_p = None
                min_time = 2 ** 32
                for p in self.hardware.processors:
                    c = self.cost[op][p.type]
                    if c <= 0:
                        continue

                    if c < min_time:
                        min_time = self.cost[op][p.type]
                        sel_p = p

                self.graph.set_dispatch(op, sel_p)


class MinimalSolver(Solver):
    def __init__(self, g: GraphCost, chip: Chip) -> None:
        super().__init__(g, chip)

    def solve(self):
        for op in self.operations:
            sel_p = None
            min_time = 2 ** 32
            for proc in self.chip.processors:
                c = self.graph.get_compute_cost_one_device(op, proc.type)
                if c <= 0:
                    continue

                if c < min_time:
                    min_time = c
                    sel_p = proc

            self.graph.set_dispatch(op, sel_p)


class ILPSolver(Solver):
    def __init__(self, g, chip: Chip) -> None:
        super().__init__(g, chip)
        self.prepare()

    def prepare(self):
        # Create a minimization problem
        self.problem = gp.Model("Minimize_Cost")

        # Define decision variables
        # should consider the parallel
        self.x = {}
        for op in self.graph.topo_sort():
            for h_id in self.chip.ids():
                self.x[op, h_id] = self.problem.addVar(
                    vtype=GRB.BINARY, name=f"x_{op}_{h_id}"
                )

        # get all edges
        self.comm_sel_r = {}
        self.comm_sel_t = {}
        for f, t in self.graph.get_edges():
            self.comm_sel_r[f, t] = {}
            self.comm_sel_t[f, t] = {}
            for p1, p2 in self.chip.get_id_combinations():
                self.comm_sel_r[f, t][p1, p2] = self.problem.addVar(
                    vtype=GRB.BINARY, name=f"comm_sel_{f}_{t}_{p1}_{p2}_r"
                )
                self.comm_sel_t[f, t][p1, p2] = self.problem.addVar(
                    vtype=GRB.BINARY, name=f"comm_sel_{f}_{t}_{p1}_{p2}_t"
                )

        self.st = {}
        self.ft_compute_only = {}
        self.ft_with_comm = {}
        for op in self.graph.topo_sort():
            self.st[op] = self.problem.addVar(
                vtype=GRB.CONTINUOUS, lb=0, name=f"st_{op}"
            )
            self.ft_compute_only[op] = self.st[op] + self.get_compute_cost(op)

        for f, t in self.graph.get_edges():
            self.ft_with_comm[f, t] = self.problem.addVar(vtype=GRB.CONTINUOUS, name = f"ft_with_comm_{f}_{t}")
            self.problem.addConstr(self.ft_with_comm[f, t] >= self.get_comm_cost(f, t) + self.ft_compute_only[f])

        def __max_cost(cost):
            max_float = 0
            for node in cost.keys():
                max_v = max(list(cost[node].backends.values()))
                max_float = max(max_v, max_float)
            return max_float

        self.M = __max_cost(self.graph.op_cost) + 100000000.0
        self.limit_resource()

    def limit_resource(self):
        self.problem.Params.Threads = 32
        self.problem.Params.NodefileStart = 1024 * 32
        # set time limit to 20 hours
        self.problem.Params.TimeLimit = 1 * 3600
        # self.problem.Params.NodeLimit = 1000000
        # self.problem.Params.SolutionLimit = 10

        logging.info(self.problem.Params.Threads)
        logging.info(self.problem.Params.OutputFlag)
        logging.info(self.problem.Params.NodeLimit)
        logging.info(self.problem.Params.SolutionLimit)
        logging.info(self.problem.Params.NodefileStart)


    def get_execution_order(self):
        st_node = []
        for node in self.graph.topo_sort():
            st_node.append((self.st[node].X, node))

        sorted_pairs = sorted(st_node, key=lambda x: x[0])
        return [x[1] for x in sorted_pairs]

    def get_compute_cost(self, op_idx):
        return gp.quicksum(
            self.x[op_idx, proc_id]
            * self.graph.get_compute_cost_one_device(op_idx, proc)
            for proc_id, proc in self.chip.processors.items()
        )

    def get_comm_cost(self, from_node, to_node):

        def get_comm_cost(d1, d2):
            if d1 == d2:
                return 0
            else:
                return self.graph.get_comm_cost_for_device(from_node, to_node,
                                                self.chip.get_processor_by_id(d1),
                                                self.chip.get_processor_by_id(d2))

        return gp.quicksum(
            self.comm_sel_r[from_node, to_node][d1, d2] *
            get_comm_cost(d1, d2)
            for d1, d2 in self.chip.get_id_combinations_no_dup()
        )

    def objective_func(self):
        """The objective is to minimize the eft of the last operations"""
        exit_nodes = self.graph.get_exit_ops()
        last_node = self.problem.addVar(vtype=GRB.CONTINUOUS, name="x_exit_node")
        for n in exit_nodes:
            self.problem.addConstr(last_node >= self.ft_compute_only[n])

        self.problem.setObjective(last_node, GRB.MINIMIZE)

    def print_problem(self):
        logging.info("Start and finish time: ")
        for node in self.graph.topo_sort():
            logging.info(f"{node} st : {self.st[node].X}, ft: {self.ft_compute_only[node].getValue()}")
            for d in self.chip.ids():
                logging.info(f"  {d}: {self.x[node, d].X}")

        for f, t in self.graph.get_edges():
            logging.info(f"{f} and {t} comm costs : {self.ft_with_comm[f, t].getValue()}")
            logging.info("Communication selection:")
            for d1, d2 in self.chip.get_id_combinations():
                sel = self.comm_sel_r[f, t][d1, d2].X
                x_f = self.x[f, d1].X
                x_t = self.x[t, d2].X
                a = self.M * (1-sel) + x_f + x_t
                b = self.M * sel - (x_f + x_t)
                logging.info(f"  Device of {f}: {self.graph.get_dispatch(f)}")
                logging.info(f"  Device of {t}: {self.graph.get_dispatch(t)}")
                logging.info(f"  {d1}, {d2}: {sel}, {a}, {a>=2}, {b}, {b>=-1}")
                


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
                    gp.quicksum(self.x[node, h_id] for h_id in self.chip.ids()) == 1
                )

                cost = self.graph.get_compute_cost(node)
                for id in self.chip.ids():
                    proc = self.chip.get_processor_by_id(id)
                    if cost.get_by_type(proc.type) <= 0:
                        self.problem.addConstr(self.x[node, id] == 0)

        def constraint_comm_sel():
            """
            Only one selection is true:
                Sum(comm_sel[e1, e2]) == 1 for all device pairs
            """
            for f, t in self.graph.get_edges():
                self.problem.addConstr(gp.quicksum(
                    [
                        self.comm_sel_r[f, t][d1, d2]
                        for d1, d2 in self.chip.get_id_combinations()
                    ]
                ) == 1, f"constr_comm_sel_{f}-{t}_r")
                self.problem.addConstr(gp.quicksum(
                    [
                        self.comm_sel_t[f, t][d1, d2]
                        for d1, d2 in self.chip.get_id_combinations()
                    ]
                ) == 1, f"constr_comm_sel_{f}-{t}_t")

                # r+ t == 1
                for d1, d2 in self.chip.get_id_combinations():
                    self.problem.addConstr(self.comm_sel_r[f, t][d1, d2] +  
                                           self.comm_sel_t[f, t][d1, d2] == 1)

        def constraint_st_ft():
            """
            1. Constraints that node a starts after its previous nodes
            for prev in prevs:
              st[node] >= ft[prev]
            """
            for node in self.graph.topo_sort():
                prevs = self.graph.prevs(node)
                for prev in prevs:
                    cond = self.st[node] >= self.ft_with_comm[prev, node]
                    self.problem.addConstr(cond, f"dep_{node}-{prev}")

        def constraint_proc_assign():
            """
            Constraints every node
            if x[n][h] == x[m][h] == 1:
                ft[n] <= st[m] or ft[m] <= st[n]  // n and m can not be overlapped
            """

            parallel_nodes = self.find_parallel_nodes()
            y = {}
            for n, m in parallel_nodes:
                for h in self.chip.ids():
                    y[n, m, h] = self.problem.addVar(
                        vtype=GRB.BINARY, name=f"y_{n}_{m}_{h}"
                    )

            for n, m in parallel_nodes:
                for h in self.chip.ids():
                    yy = y[n, m, h]
                    xn = self.x[n, h]
                    xm = self.x[m, h]
                    cond1 = self.ft_compute_only[n] - self.st[m] + 2 * self.M * (xn + xm - 2) <= self.M * yy
                    cond2 = self.ft_compute_only[m] - self.st[n] + 2 * self.M * (xn + xm - 2) <= self.M * (
                            1 - yy
                    )
                    self.problem.addConstr(cond1, f"pair_{n}_{m}_{h}")
                    self.problem.addConstr(cond2, f"pair_{m}_{n}_{h}")

        def constraint_comm_cost_new():
            """

            if x1 + x2 == 2:
                com_sel[1,2] = 1
            else:
                com_sel[1,2] = 0


            r == 1 , t == 0:
              2 <= x1 + x2 <= 2
            r == 0, t == 1:
              0 <= x1 + x2 <= 1

            ==>
                
            r + t = 1
            2r + 0 t <= x1 + x2 <= 2r + t

            ==>
            """


            print(self.x.keys())
            for f, t in self.graph.get_edges():
                for d1, d2 in self.chip.get_id_combinations():
                    r = self.comm_sel_r[f, t][d1, d2]
                    tt = self.comm_sel_t[f, t][d1, d2]
                    x_1 = self.x[f, d1]
                    x_2 = self.x[t, d2]
                    cond1 = 2 * r <= x_1 + x_2
                    cond2 = x_1 + x_2 <= 2 * r + tt
                    self.problem.addConstr(cond1)
                    self.problem.addConstr(cond2)


        def constraint_comm_cost():
            """

            if x1 + x2 == 2:
                com_sel[1,2] = 1
            else:
                com_sel[1,2] = 0

                
           2 <= (x1 + x2) <= 2 --> s = 

            """


            for f, t in self.graph.get_edges():
                for d1, d2 in self.chip.get_id_combinations():
                    sel = self.comm_sel[f, t][d1, d2]
                    x_f = self.x[f, d1]
                    x_t = self.x[t, d2constraint_comm_cost_new]
                    self.problem.addConstr(self.M * (1 - sel) + x_f + x_t >= 2, f"constr_comm_cost_{f}-{t}-{d1}-{d2}-cond1")
                    self.problem.addConstr(x_f + x_t <= 1 + self.M * sel , f"constr_comm_cost_{f}-{t}-{d1}-{d2}-cond2")

        constraint_x()
        constraint_st_ft()
        constraint_comm_sel()
        constraint_comm_cost_new()
        constraint_proc_assign()

    def solve(self):
        self.objective_func()
        self.add_constraints()
        self.problem.optimize()
        self.problem.write("schedule.lp")
        self.get_device_dispatch_results()
        self.print_problem()
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
            has_dispatched = False
            for j in self.chip.ids():
                if self.x[i, j].X == 1:
                    has_dispatched = True
                    self.graph.set_dispatch(i, j)

            if not has_dispatched:
                logging.warning(
                    f"Error: {i} is not dispatched, dispatch to the one with minimum cost"
                )
                for i in self.graph.topo_sort():
                    for j in self.chip.ids():
                        pp(f"x of {i}, {j} = {self.x[i, j].X}")
                exit(-1)


def solveDag(solverType, g: GraphCost, chip: Chip, f=None):
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

    return solver.graph
