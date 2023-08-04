#!/usr/bin/env python

import os
import sys

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_dir)

from schedule.cost_graph import *
from schedule.solver import *
from schedule.processor import *
from schedule.emulator import async_emulation
import logging
import argparse
import pathlib

logging.basicConfig(
    level=logging.INFO,                # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s',  # Set the log message format
    datefmt='%Y-%m-%d %H:%M:%S'         # (Optional) Set the date format
)

def run_network_scheduling(graph_structure, compute_cost, comm_cost, chip):
    df_graph = pd.read_csv(graph_structure)
    df_compute = pd.read_csv(compute_cost)
    df_comm = pd.read_csv(comm_cost)

    graph = GraphCost(df_graph, df_compute, df_comm, chip)

    results = solveDag(ILPSolver, graph, chip)

    exec_time = async_emulation(results, chip)
    return results, exec_time

    
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str, help="Model file in csv format")
    parser.add_argument("--compute_cost", required=True, type=str, help="Computation cost file in csv format")
    parser.add_argument("--comm_cost", required=True, type=str, help="Communication cost file in csv format")
    parser.add_argument("--chip", type = str, required=True, help="Chip type, supporting bst, khadas and khadas_cpu_only")
    parser.add_argument("--dump", type=str, help="The prefix of dumping path")

    return parser.parse_args()

def print_parameter(args):
    logging.info("============ Parameters ============")
    logging.info(f"model: {args.model}")
    logging.info(f"chip: {args.chip}")
    pass

def main():
    args = get_args()
    print_parameter(args)

    model = args.model
    compute_cost = args.compute_cost
    comm_cost = args.comm_cost
    chip = args.chip
    dump = args.dump

    if chip in supported_chips.keys():
        r, t = run_network_scheduling(model, compute_cost, comm_cost, supported_chips[chip])
        if dump is not None:
            p = pathlib.Path(dump)
            r.draw_results(supported_chips[chip], p.with_suffix(".pdf"))
            r.dispatch_to_csv(dispatch_csv_file= p.with_suffix(".dispatch.csv"))
            pass

        logging.critical("Total time: {}".format(t.get_total_time()))

    else:
        logging.error(f"Unsupported backends, try: {list(supported_chips.keys())}")

main()

