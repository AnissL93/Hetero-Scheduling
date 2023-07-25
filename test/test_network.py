#!/usr/bin/env python

import os
import sys

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_dir)

from schedule.cost_graph import *
from schedule.solver import *
from schedule.processor import bst_chip,arm_chip
from schedule.emulator import async_emulation

def run_network_scheduling(csv_file, chip):
    graph = read_csv(csv_file)
    print(graph.to_df())
    graph.draw_graph_structure("a.pdf")

    results = solveDag(ILPSolver, graph, chip)
    print(results)

    exec_time = async_emulation(graph, chip, list(results.keys()), results)
    print(exec_time.get_total_time())

def test_arm():
    run_network_scheduling("../data/net_perf/arm/InceptionV1.csv", arm_chip)

