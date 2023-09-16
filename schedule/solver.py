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
from .emulator import async_emulation

MAX_FLOAT = 1e20
RESULT_THR = 0.0001
FORCE_MAIN_CORE = False

class Solver(object):
    def __init__(self, g: GraphCost, chip: Chip, name = None) -> None:
        self.graph = g
        self.chip = chip
        self.model_name = name

    # def rectify(self):
    #     for op in self.operations:
    #         if self.cost[op][self.graph.get_dispatch(op).type] <= 0:
    #             sel_p = None
    #             min_time = 2 ** 32
    #             for p in self.chip.processors:
    #                 c = self.cost[op][p.type]
    #                 if c <= 0:
    #                     continue

    #                 if c < min_time:
    #                     min_time = self.cost[op][p.type]
    #                     sel_p = p

    #             self.graph.set_dispatch(op, sel_p.id)

    def assign_to_one(self):
        if len(self.chip.ids()) == 1:
            p = self.chip.ids()[0]
            ret = DispatchResult(self.graph.graph_id)
            for i, op in enumerate(self.graph.topo_sort()):
                ret.set(op, i, p)
            return ret
        else:
            return None

class MinimalSolver(Solver):
    ID = "naive"

    def __init__(self, g: GraphCost, chip: Chip, model_name) -> None:
        super().__init__(g, chip, model_name)

    def solve(self):
        to_one = self.assign_to_one()
        if to_one is not None:
            return to_one

        ret = DispatchResult(self.graph.graph_id)
        for order, op in enumerate(self.graph.topo_sort()):
            sel_p = None
            min_time = 2 ** 32
            for pid, proc in self.chip.processors.items():
                c = self.graph.get_compute_cost_one_device(op, proc)
                if c <= 0:
                    continue

                if c < min_time:
                    min_time = c
                    sel_p = pid
            ret.set(op, order, sel_p) 

        return ret


class ILPSolver(Solver):

    ID = "ilp"

    def __init__(self, g, chip: Chip, model_name = None) -> None:
        super().__init__(g, chip, model_name)
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
                    vtype=GRB.BINARY, name=f"x_{op}_{h_id}"
                )

        # get all edges
        self.comm_sel = {}
        for f, t in self.graph.get_edges():
            self.comm_sel[f, t] = {}
            for p1, p2 in self.chip.get_id_combinations():
                self.comm_sel[f, t][p1, p2] = self.problem.addVar(
                    vtype=GRB.BINARY, name=f"comm_sel_{f}_{t}_{p1}_{p2}"
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
            self.ft_with_comm[f, t] = self.get_comm_cost(f, t) + self.ft_compute_only[f]

        def __max_cost(cost):
            max_float = 0
            for node in cost.keys():
                max_v = max(list(cost[node].backends.values()))
                max_float = max_v + max_float
                # max_float = max(max_v, max_float)
            return max_float

        self.M = __max_cost(self.graph.op_cost)
        logging.info(f"Big M is {self.M}")
        self.limit_resource()

    def limit_resource(self):
        self.problem.Params.Threads = 32
        self.problem.Params.NodefileStart = 1024 * 32
        # set time limit to 20 hours
        self.problem.Params.TimeLimit = 60 * 60 * 6
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
            self.comm_sel[from_node, to_node][d1, d2] *
            get_comm_cost(d1, d2)
            for d1, d2 in self.chip.get_id_combinations()
        )

    def objective_func(self):
        """The objective is to minimize the eft of the last operations"""
        exit_nodes = self.graph.get_exit_ops()
        last_node = self.problem.addVar(vtype=GRB.CONTINUOUS, name="x_exit_node")
        for n in exit_nodes:
            self.problem.addConstr(last_node >= self.ft_compute_only[n])

        self.problem.setObjective(last_node, GRB.MINIMIZE)

    def print_problem(self):
        # print("Start and finish time: ")
        # for node in self.operations:
        #     print(f"{node} st : {self.st[node].value()}")
        #     print(f"{node} ft : {self.ft[node].value()}")
        pass

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
                        self.comm_sel[f, t][d1, d2]
                        for d1, d2 in self.chip.get_id_combinations()
                    ]
                ) == 1)

            for f, t in self.graph.get_edges():
                for d1, d2 in self.chip.get_id_combinations():
                    xf1 = self.x[f, d1]
                    xt2 = self.x[t, d2]
                    y = self.comm_sel[f, t][d1, d2]
                    expr = xf1 + xt2 - 2*y
                    self.problem.addConstr(expr >= 0)
                    self.problem.addConstr(expr <= 1)

            
        def constraint_force_main_core():
            """
            Force node with multiple outputs or inputs to big core
            """
            main_core = self.chip.get_main_core()
            for node in self.graph.topo_sort():
                if len(self.graph.sucs(node)) > 1 or len(self.graph.prevs(node)) > 1:
                    self.problem.addConstr(self.x[node, main_core] == 1)


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

        def constraint_proc_assign_with_comm():
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
                st_m = self.st[m]
                st_n = self.st[n]
                for h in self.chip.ids():
                    xn = self.x[n, h]
                    xm = self.x[m, h]
                    for suc_n in self.graph.sucs(n):
                        for suc_m in self.graph.sucs(m):
                            ed_m = self.ft_with_comm[m, suc_m]
                            ed_n = self.ft_with_comm[n, suc_n]
                            y_var = self.problem.addVar(
                                vtype=GRB.BINARY
                            )
                            cond1 = ed_n - st_m <= self.M * (2 - xn - xm) + self.M * y_var
                            cond2 = ed_m - st_n <= self.M * (2 - xn - xm) + self.M * (1-y_var)
                            self.problem.addConstr(cond1)
                            self.problem.addConstr(cond2)
                

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
                    #cond = self.ft_compute_only[n] - self.st[m] + 2* self.M * (xn + xm - 2) <= self.M * yy
                    cond = self.ft_compute_only[n] - self.st[m] <= self.M * (2 - xn - xm) + self.M * yy
                    self.problem.addConstr(cond, f"pair_{n}_{m}_{h}")

                    cond = self.ft_compute_only[m] - self.st[n] <= self.M * (2 - xn - xm) + self.M * (1-yy)
                    # cond = self.ft_compute_only[m] - self.st[n] + 2* self.M * (xn + xm - 2) <= self.M * (
                            # 1 - yy
                    # )
                    self.problem.addConstr(cond, f"pair_{m}_{n}_{h}")

        constraint_x()
        constraint_comm_sel()
        constraint_st_ft()
        constraint_proc_assign()
        # constraint_proc_assign_with_comm()
        if FORCE_MAIN_CORE:
            constraint_force_main_core()

    def solve(self):
        to_one = self.assign_to_one()
        if to_one is not None:
            return to_one

        self.objective_func()
        self.add_constraints()
        self.problem.optimize()

        if self.model_name is not None:
            pass
            # self.problem.write("results/ilp/" + self.model_name + ".lp")
            # self.problem.write("results/ilp/" + self.model_name + ".mst")
            # self.problem.write("results/ilp/" + self.model_name + ".sol")

        self.print_problem()
        return self.get_device_dispatch_results()

    def get_device_dispatch_results(self):
        order = self.get_execution_order()
        dist = DispatchResult(self.graph.graph_id)
        for idx, op in enumerate(order):
            has_dispatched = False
            for j in self.chip.ids():
                if abs(1. - self.x[op, j].X) < RESULT_THR:
                    has_dispatched = True
                    dist.set(op, idx, j, self.graph.graph_id)

            if not has_dispatched:
                logging.warning(
                    f"Error: {idx} is not dispatched, dispatch to the one with minimum cost"
                )
                for i in self.graph.topo_sort():
                    for j in self.chip.ids():
                        pp(f"x of {i}, {j} = {self.x[i, j].X}")
                exit(-1)

        return dist

class Solution(object):

    def __init__(self, g : GraphCost, chip : Chip, solver_type : Solver, model_name):
        assert len(chip.groups) > 0
        self.origin_graph = g
        self.chip = chip
        self.model_name = model_name
        self.dispatch_results = {}
        self.emulation_results = {}
        self.solver_type = solver_type

    def get_dispatch(self, group : str, sg_id):
        return self.dispatch_results[group, sg_id]

    def get_emulation_time(self, group : str, sg_id):
        return self.emulation_results[group, sg_id]

    def run_group(self, group):
        chip = self.chip.get_group_as_chip(group)
        g = self.origin_graph
        dispatch = DispatchResult()
        print(self.dispatch_results.keys())
        if len(g.subgraphs) > 0:
            for i, sg in g.subgraphs.items():
                if self.dispatch_results[group, sg.graph_id] is None:
                    logging.info(f"Skip subgraph {sg.graph_id} which does not support chip {str(chip)}")
                    continue

                dist = self.dispatch_results[group, sg.graph_id]
                total_time = async_emulation(sg, dist, chip).get_total_time()
                self.emulation_results[group, sg.graph_id] = total_time

        else:
            dist = self.dispatch_results[group, g.graph_id]
            total_time = async_emulation(g, dist, chip).get_total_time()
            self.emulation_results[group, g.graph_id] = total_time

    def solve_group(self, SolverType, group : str):
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

        the dispatched results will be stored in each subgraph or a graph
        """
        ## init solution
        chip = self.chip.get_group_as_chip(group)
        g = self.origin_graph
        dispatch = DispatchResult()
        if len(g.subgraphs) > 0:
            for i, sg in g.subgraphs.items():
                if not sg.can_support_chip(chip):
                    logging.info(f"Subgraph {sg.graph_id} does not support chip {str(chip)}")
                    self.dispatch_results[group, sg.graph_id] = None
                    self.emulation_results[group, sg.graph_id] = None
                    continue

                solver = SolverType(sg, chip, f"{sg.graph_id}_{self.model_name}")
                dist = solver.solve()
                # gather all dispatch info to parent graph
                self.dispatch_results[group, sg.graph_id] = dist

        else:
            solver = SolverType(g, chip, f"{g.graph_id}_{self.model_name}")
            dispatch = solver.solve()
            self.dispatch_results[group, g.graph_id] = dispatch

    def solve_and_run(self, skip_solver = False):
        """Get the dispatch results and get estimated cost for <group, subgraph>
        """
        for group in self.chip.groups.keys():
            logging.info(f"Solve group {group}")
            if not skip_solver:
                self.solve_group(self.solver_type, group)
            else:
                logging.info(">>>>>>>>>>> Skip solver!")
            self.run_group(group)

    def dispatch_to_df(self):
        """
        "group", "graph_id", "op_id", order, dispatch
        """
        array = []
        for (group, sg_id), dispatch in self.dispatch_results.items():
            for op_id in self.origin_graph.subgraphs[sg_id].topo_sort():
                if dispatch is None:
                    order, disp = None, None
                else:
                    order ,disp = dispatch.get(op_id, sg_id)

                array.append([
                    group,
                    sg_id,
                    op_id,
                    order,
                    disp
                ])
        return pd.DataFrame(array, columns = ["group_id", "graph_id","op_id", "order", "dispatch"])

    def emu_time_to_df(self):
        array = []
        for (group, sg_id), time in self.emulation_results.items():
            array.append([ group, sg_id, time ])

        return pd.DataFrame(array, columns=["group_id", "graph_id", "time"])
