#!/usr/bin/env python

import os
import sys

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_dir)

from schedule.cost_graph import *
from schedule.solver import *
from schedule.processor import bst_chip,khadas_chip
from schedule.emulator import async_emulation

def run_network_scheduling(csv_file, chip):
    graph = read_csv(csv_file)

    results = solveDag(ILPSolver, graph, chip)

    exec_time = async_emulation(results, chip)
    return results, exec_time

# after order [7, 8, 9, 11, 12, 14, 10, 15, 13, 16]
# 7: CPU_B
# 8: CPU_S
# 9: GPU
# 11: CPU_B
# 12: CPU_B
# 14: GPU
# 10: CPU_B
# 15: GPU
# 13: GPU
# 16: CPU_B
def test_arm():
    r, t = run_network_scheduling("data/net_perf/arm/InceptionV3_block.csv", khadas_chip)
    r.info()
    assert t.get_total_time() == 8241


