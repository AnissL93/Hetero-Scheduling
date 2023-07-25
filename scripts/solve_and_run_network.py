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

    results = solveDag(ILPSolver, graph, chip)
    print(results)

    exec_time = async_emulation(graph, chip, list(results.keys()), results)
    print(exec_time.get_total_time())

def main():
    if len(sys.argv) != 3:
        print("./solve_and_run_network.py <model_path> <chip_type>")
        return
        
    model = sys.argv[1]
    chip = sys.argv[2]
    if chip == "arm":
        c = arm_chip
    elif chip == "bst":
        c = bst_chip

    run_network_scheduling(model, c)

main()

