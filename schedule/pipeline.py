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

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_dir)


class Pipeline(object):
    def __init__(self, solution, chip: Chip, stages: list):
        """Estimate results and find all split point with pipeline

        Args:
            solution (dict): _description_
            chip (Chip): _description_
            stages (list): _description_
        """
        self.stages = stages
        self.solution = solution
        self.chip = chip
        pass

    def traverse_split_point(self, num_point):
        # get all nodes as the last node of a stage
        all_points = [n for n in self.graph.nx_subgraph.nodes if len(
            self.graph.nx_subgraph.successors(n)) == 0]

        if num_point == 1:
            return all_points
        elif num_point == 2:
            ret = []
            for i in range(len(num_point)):
                for j in range(i+1, len(num_point)):
                    ret.append((num_point[i], num_point[j]))
                    
        return ret

    def est_throughput(self, batch_num, split_point : tuple):
        ## (cost_of_stage0 + (batch_num) * cost_of_stage1) / batch_num
        if len(split_point) == 1:
            return stages[i] 
            