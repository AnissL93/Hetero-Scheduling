#!/usr/bin/env python
import datetime
import pathlib
import argparse
import logging
from .emulator import async_emulation
from .processor import *
from .solver import *
from .cost_graph import *
import os
import sys
import networkx as nx

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_dir)


class Pipeline(object):
    def __init__(self, solution: Solution, factor = 1e-9):
        """Estimate results and find all split point with pipeline

        Args:
            solution (dict): _description_
            chip (Chip): _description_
            stages (list): stage's chip group, e.g., [["group0", "group1"], ["group1", "group0"]]
        """
        self.solution = solution
        self.performance = {}
        self.factor = factor
        pass

    def traverse_split_point(self, num_point, nx_subgraph: nx.DiGraph):
        """Get all possible stage combination of subgraphs 

          The subgraphs are got by removing successor edges of split point 
          and return components containing the split point 

        """
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
                for i in range(len(num_point)):
                    for j in range(i+1, len(num_point)):
                        ret.append([num_point[i], num_point[j]])
            return ret

        def get_components(points, origin_sg):
            sg = origin_sg.copy().to_undirected()
            ret = []
            for p in points:
                if p not in sg.nodes:
                    return None

                for e1, e2 in origin_sg.out_edges(p):
                    sg.remove_edge(e1, e2)
                    print("remove ", e1, e2)

                component = nx.node_connected_component(sg, p)
                comp_g = sg.subgraph(component).copy()
                ret.append(comp_g)
                # remove component from the graph
                sg.remove_nodes_from(component)

            # add the rest of the sg
            if len(sg.nodes) > 0:
                ret.append(sg)

            return ret

        sp = get_split_points()
        all_configs = []
        for points in sp:
            comp = get_components(points, nx_subgraph)

            if comp is not None:
                all_configs.append(comp)

        return all_configs

    def estimate(self, stage: list, proc_group: str):
        """Estimate for the time of a stage, on a certain proc_group
        """
        ret = 0
        for sg in stage:
            cost = self.solution.get_emulation_time(proc_group, sg)
            if cost is None:
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
            return  float(batch_num) / ((c0 + max_cost * (batch_num-1) + c1) * self.factor)
        elif len(stage_costs) == 3:
            c0 = stage_costs[0]
            c1 = stage_costs[1]
            c2 = stage_costs[2]
            return  float(batch_num) / ((c0 + max(c0, c1) + max_cost * (batch_num - 2) + max(c1, c2) + c2) * self.factor)
        elif len(stage_costs) == 1:
            return float(batch_num) / (stage_costs[0]*batch_num * self.factor)
        else:
            logging.fatal("Unsupported stage number")
            assert False
        

    def to_df(self):
        config = list(self.performance.keys())
        perf = list(self.performance.values())
        data = {
            "config" : config,
            "perf" : perf
        }
        return pd.DataFrame(data)

    def eval_all(self, batch_num):
        """traverse for <proc-config, split_points> -> latency, throughput
        """
        stage_configs_cache = {}
        proc_configs = self.solution.chip.get_proc_groups()

        for proc_config in proc_configs:
            # proc_config = ["group0", "group1"]
            num_point = len(proc_config) - 1
            if num_point not in stage_configs_cache.keys():
                stage_configs = self.traverse_split_point(
                    num_point, self.solution.origin_graph.nx_subgraph)
                print(stage_configs)
                stage_configs_cache[num_point] = stage_configs
            else:
                stage_configs = stage_configs_cache[num_point]

            # get pipelined cost for each split point strategy
            for config_id, config in enumerate(stage_configs):
                stage_cost = []
                assert len(config) == len(proc_config)
                # get cost of each stage
                # config: a subgraph
                for i in range(len(config)):
                    # map i proc to i stage
                    cost = self.estimate(config[i], proc_config[i])
                    print("cost is None, skip")
                    if cost is not None:
                        stage_cost.append(cost)

                print(f"============= {config} stage cost {stage_cost}")

                # add perf if this is a valid split point
                if len(stage_cost) == len(config):
                    latency = self.est_lantency(stage_cost)
                    throughput = self.est_throughput(batch_num, stage_cost)
                    self.performance[str(proc_config), config_id] = (latency, throughput)
