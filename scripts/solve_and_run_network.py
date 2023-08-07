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
import datetime

# Get the current timestamp with microsecond precision

def get_log_file_name(model_file):
    file_name = pathlib.Path(model_file).stem
    current_timestamp = datetime.datetime.now().timestamp()
    date_time = datetime.datetime.fromtimestamp(current_timestamp)
    str_date_time = date_time.strftime("%d-%m-%Y-%H:%M:%S")
    home = os.environ.get("HETERO_SCHEDULE_HOME")
    log_file = home + "/log/" + file_name + "-" + str_date_time + ".log"
    return log_file

def run_network_scheduling(model, chip):
    df_graph = pd.read_csv(model)
    graph = GraphCost(df_graph, chip)
    results = solveDag(ILPSolver, graph, chip)
    exec_time = async_emulation(results, chip)
    return results, exec_time

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str,
                        help="Model file in csv format, including graph structure, compute cost and communication cost.")
    parser.add_argument("--chip", type=str, required=True, help="Chip type, supporting bst, khadas and khadas_cpu_only")
    parser.add_argument("--dump", type=str, help="The prefix of dumping path")
    parser.add_argument("--log", help="Log file", action="store_true")
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
    chip = args.chip
    dump = args.dump
    log = args.log

    if log is not None:
        full_path = get_log_file_name(model)
        print(f"Enable logging, store log to {full_path}")
        logging.basicConfig(
            level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            format='%(asctime)s - %(levelname)s - %(message)s',  # Set the log message format
            datefmt='%Y-%m-%d %H:%M:%S',  # (Optional) Set the date format
            filename=full_path
        )

    if chip in supported_chips.keys():
        r, t = run_network_scheduling(model, supported_chips[chip])
        logging.critical("Total time: {}".format(t.get_total_time()))

        if dump is not None:
            p = pathlib.Path(dump)
            r.draw_results(supported_chips[chip], p.with_suffix(".pdf"))
            r.dispatch_to_csv(dispatch_csv_file=p.with_suffix(".dispatch.csv"))

    else:
        logging.error(f"Unsupported backends, try: {list(supported_chips.keys())}")

main()

