import os
import sys

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_dir)

from schedule.cost_graph import *
from schedule.solver import *
from schedule.processor import *
from schedule.emulator import async_emulation
import argparse
import pathlib
import logging

logging.basicConfig(
    level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(levelname)s - %(message)s",  # Set the log message format
    datefmt="%Y-%m-%d %H:%M:%S",  # (Optional) Set the date format
)


def run_network_scheduling(model, dispatch, chip):
    df_graph = pd.read_csv(model)
    df_dispatch = pd.read_csv(dispatch)
    graph = DispatchedGraph(GraphCost(df_graph, chip), df_dispatch)
    exec_time = async_emulation(graph, chip)
    return graph, exec_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="The model file with dispatched processor information",
    )
    parser.add_argument("--dispatch", type=str, required=True, help="The dispatch file")
    parser.add_argument(
        "--chip",
        type=str,
        required=True,
        help="Chip type, supporting bst, khadas and khadas_cpu_only",
    )
    parser.add_argument("--dump", type=str, help="The prefix of dumping path")
    args = parser.parse_args()
    model = args.model
    chip = args.chip
    dump = args.dump
    dispatch = args.dispatch

    if chip in supported_chips.keys():
        r, t = run_network_scheduling(model, dispatch, supported_chips[chip])
        if dump is not None:
            p = pathlib.Path(dump)
            r.draw_results(supported_chips[chip], p.with_suffix(".pdf"))

        logging.critical("Total time: {}".format(t.get_total_time()))

    else:
        print(f"Error: Unsupported backends, try: {supported_chips.keys()}")


main()
