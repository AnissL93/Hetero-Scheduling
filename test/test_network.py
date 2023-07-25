#!/usr/bin/env python

import os
import sys

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_dir)

from schedule.cost_graph import *
from schedule.solver import *
from schedule.processor import bst_chip
from schedule.emulator import async_emulation


def test_cost_graph_read():
    graph = read_csv("data/net_perf/bst/inception_v1.csv")
    print(graph.to_df())

    results = solveDag(ILPSolver, graph, bst_chip)
    print(results)

    exec_time = async_emulation(graph, bst_chip, list(results.keys()), results)
    print(exec_time.get_total_time())


test_cost_graph_read()
