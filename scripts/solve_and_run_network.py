#!/usr/bin/env python
import os
import sys

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_dir)

from schedule.cost_graph import *
from schedule.solver import *
from schedule.processor import *
from schedule.emulator import async_emulation
from schedule.execute import *
import logging
import argparse
import pathlib
import datetime
import yaml

logging.basicConfig(
    level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(levelname)s - %(message)s",  # Set the log message format
    datefmt="%Y-%m-%d %H:%M:%S",  # (Optional) Set the date format
)

# Get the current timestamp with microsecond precision

def get_model_name(model_file, chip, solver):
    file_name = pathlib.Path(model_file).stem
    current_timestamp = datetime.datetime.now().timestamp()
    date_time = datetime.datetime.fromtimestamp(current_timestamp)
    str_date_time = date_time.strftime("%d-%m-%Y-%H:%M:%S")
    return f"{chip}-{file_name}-{solver}-{str_date_time}"

def get_log_file_name(model_file, chip, solver):
    home = os.environ.get("HETERO_SCHEDULE_HOME")
    log_file = home + "/log/" + get_model_name(model_file, chip, solver) + ".log"
    return log_file

def run_network_scheduling(model, chip, solver):
    c = supported_chips[chip]
    df_graph = pd.read_csv(model)
    graph = GraphCost(df_graph, c)
    if solver == "ILP":
        results = solveDag(ILPSolver, graph, c, get_model_name(model, chip, solver))
    else:
        results = solveDag(MinimalSolver, graph, c, get_model_name(model, chip, solver))
    exec_time = async_emulation(results, c)
    return results, exec_time

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        type=str,
        help="Model file in csv format, including graph structure, compute cost and communication cost.",
    )
    parser.add_argument(
        "--chip",
        type=str,
        required=True,
        help="Chip type, supporting bst, khadas and khadas_cpu_only",
    )
    parser.add_argument("--dump", type=str, help="The prefix of dumping path")
    parser.add_argument("--log", help="Log file", action="store_true")
    parser.add_argument("--solver", help="The solver to use", type=str, default="ILP")
    return parser.parse_args()

def get_args_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="Config file"
    )
    return parser.parse_args()

def print_parameter(args):
    logging.info("============ Parameters ============")
    logging.info(f"model: {args.model}")
    logging.info(f"chip: {args.chip}")
    logging.info(f"dump: {args.dump}")
    pass


def end2end():
    args = get_args_config()
    with open(args.config, 'r') as yaml_file:
        data = yaml.load(yaml_file, Loader=yaml.FullLoader)

    model = data.get("network")
    chips = data.get("chips")
    dump_path = data.get("dump_path")
    subgraph = data.get("subgraph")
    assert os.path.exists(subgraph)
    assert os.path.exists(model)

    # log file
    log_path = get_log_file_name(model, str(chips), "all")
    file_handler = logging.FileHandler(log_path)
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger("").addHandler(file_handler)

    for c in chips:
        assert c in supported_chips.keys()
        chip = supported_chips[c]
        solution = solve(model, chip, subgraph)
        est_results = estimate_for_groups(solution, chip)

        if len(dump_path) > 0:
            p = pathlib.Path(dump_path)
            p = p.with_stem(p.stem + "-" + c)
            dump_results(solution, p, chip)
            est_results.to_csv(p.with_suffix(".time.csv"))
    

def main():
    args = get_args()
    print_parameter(args)

    model = args.model
    chip = args.chip
    dump = args.dump
    log = args.log
    solver = args.solver
    if log is not None:
        log_path = get_log_file_name(model, chip, solver)
        print(f"Enable logging, store log to {log_path}")

    if chip in supported_chips.keys():
        r, t = run_network_scheduling(model, chip, solver)
        logging.critical("Total time: {}".format(t.get_total_time()))

        if dump is not None:
            p = pathlib.Path(dump)
            r.draw_results(supported_chips[chip], p.with_suffix(".pdf"))
            r.dispatch_to_csv(dispatch_csv_file=p.with_suffix(".dispatch.csv"))

    else:
        logging.error(f"Unsupported backends, try: {list(supported_chips.keys())}")


#main()
end2end()
