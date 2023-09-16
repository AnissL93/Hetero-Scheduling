#!/usr/bin/env python
import datetime
import pathlib
import argparse
import logging

import pandas as pd

from .emulator import async_emulation
from .processor import *
from .solver import *
from .cost_graph import *
from .file import dump_graph
import os
import sys
import networkx as nx


# project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.insert(0, project_dir)


class Pipeline(object):

    def __init__(self, solution: Solution, factor=1e-9):
        """Estimate results and find all split point with pipeline

        Args:
            solution (dict): _description_
            chip (Chip): _description_
            stages (list): stage's chip group, e.g., [["group0", "group1"], ["group1", "group0"]]
        """
        self.solution = solution
        self.performance = {}
        self.stage_costs = {}
        self.factor = factor
        pass

    def _enumerate_split_point(self, num_point):
        """Get all possible stage combination of subgraphs 

          The subgraphs are got by removing successor edges of split point 
          and return components containing the split point 

        """
        logging.info(f">>> Enumerate split points for {num_point}")
        nx_subgraph = self.solution.origin_graph.nx_subgraph

        def get_split_points():
            # get all nodes as the last node of a stage
            if num_point == 0:
                exit_nodes = [n for n in nx_subgraph.nodes if len(list(
                    nx_subgraph.successors(n))) == 0]
                return [[exit_nodes[0]]]
            elif num_point == 1:
                return [[n] for n in nx_subgraph.nodes if len(list(
                    nx_subgraph.successors(n))) > 0]
            elif num_point == 2:
                ret = []
                possible_nodes = [n for n in nx_subgraph.nodes if
                                  len(list(nx_subgraph.successors(n))) > 0]
                possible_nodes = sorted(possible_nodes)
                for i in range(len(possible_nodes)):
                    for j in range(i + 1, len(possible_nodes)):
                        ret.append([possible_nodes[i], possible_nodes[j]])
            return ret

        def get_components(points, origin_sg):
            logging.info(f"Components of points {points}")
            sg = origin_sg.copy().to_undirected()
            ret = []
            for p in points:
                if p not in sg.nodes:
                    return None

                for e1, e2 in origin_sg.out_edges(p):
                    logging.info(f"Remove edge {e1} and {e2}")
                    sg.remove_edge(e1, e2)

                component = nx.node_connected_component(sg, p)
                logging.info(f"Add subgraph {list(component)}")
                comp_g = sg.subgraph(component).copy()
                ret.append(comp_g)
                # remove component from the graph
                sg.remove_nodes_from(component)

            # add the rest of the sg
            if len(sg.nodes) > 0:
                ret.append(sg)

            return ret

        sp = get_split_points()
        logging.info(f">>> Split points: {sp}")
        all_strategys = []
        for points in sp:
            comp = get_components(points, nx_subgraph)

            if comp is not None:
                all_strategys.append(comp)

        return all_strategys

    def estimate(self, stage: list, proc_group: str):
        """Estimate for the time of a stage, on a certain proc_group
        """
        ret = 0
        for sg in stage:
            cost = self.solution.get_emulation_time(proc_group, int(sg))
            if cost is None or cost < 0:
                return None

            ret += cost

        return ret

    def est_lantency(self, stage_costs):
        ## return latency in ms
        return sum(stage_costs) * self.factor

    def est_throughput(self, batch_num, stage_costs):
        ## return fps
        max_cost = max(stage_costs)
        if len(stage_costs) == 2:
            c0 = stage_costs[0]
            c1 = stage_costs[1]
            return float(batch_num) / ((c0 + max_cost * (batch_num - 1) + c1) * self.factor)
        elif len(stage_costs) == 3:
            c0 = stage_costs[0]
            c1 = stage_costs[1]
            c2 = stage_costs[2]
            return float(batch_num) / ((c0 + max(c0, c1) + max_cost * (batch_num - 2) + max(c1, c2) + c2) * self.factor)
        elif len(stage_costs) == 1:
            return float(batch_num) / (stage_costs[0] * batch_num * self.factor)
        else:
            logging.fatal("Unsupported stage number")
            assert False

    def _strategy_to_list(self, s):
        ret = []
        for stg in s:
            ret.append(list(stg.nodes))
        return ret

    def dump_stage_strategy(self, all_strategys):
        """
        Dump all strategy to df
        ,strategy_id,strategy
        0,0,[[1,2,3],[4,5,6]]
        """

        array = []
        for strategy_id, strategy in enumerate(all_strategys):
            array.append([
                strategy_id,
                str(self._strategy_to_list(strategy))
            ])

        return pd.DataFrame(array, columns=["strategy_id", "strategy"])

    def to_df(self):
        strategy = list(self.performance.keys())
        perf = list(self.performance.values())
        data = {
            "strategy": strategy,
            "perf": perf
        }
        return pd.DataFrame(data)

    def _get_cost_proc_to_stage(self, strategy, proc):
        """
        Return the cost of running stage on proc
        """
        cost = self.estimate(strategy, proc)


    def eval_all(self, batch_num):
        """traverse for <proc-strategy, split_points> -> latency, throughput
        """
        stage_strategys_cache = {}
        proc_strategys = self.solution.chip.get_proc_groups()

        for proc_strategy in proc_strategys:
            logging.info(f">>> Estimating s{proc_strategy}")
            # proc_strategy = ["group0", "group1"]
            num_point = len(proc_strategy) - 1
            if num_point not in stage_strategys_cache.keys():
                stage_strategys = self._enumerate_split_point(num_point)
                stage_strategys_cache[num_point] = stage_strategys
            else:
                stage_strategys = stage_strategys_cache[num_point]

            # get pipelined cost for each split point strategy
            logging.info(f">>> Found {len(stage_strategys)} possible strategies")
            logging.info(stage_strategys[0])
            for strategy in stage_strategys:
                logging.info(f"    >>> Mapping {self._strategy_to_list(strategy)} to processor {proc_strategy}")
                assert len(strategy) == len(proc_strategy)
                # estimate cost of each stage in one strategy
                stage_cost = [self.estimate(strategy[i], proc_strategy[i]) for i in range(len(strategy))]
                # add perf if this is a valid split point
                if None not in stage_cost:
                    # logging.info(f"    >>> Add strategy: {stage_cost}")

                    latency = self.est_lantency(stage_cost)
                    throughput = self.est_throughput(batch_num, stage_cost)
                    self.performance[str(proc_strategy), str(self._strategy_to_list(strategy))] = (latency, throughput, stage_cost)
