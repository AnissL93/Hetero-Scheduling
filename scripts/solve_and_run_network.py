#!/usr/bin/env python
import os
import sys

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_dir)

from sacred.observers import MongoObserver
from sacred import Experiment
from pprint import pprint

from schedule.cost_graph import *
from schedule.solver import *
from schedule.processor import *
from schedule.emulator import async_emulation
from schedule.execute import *
from schedule.plot import *
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


ex = Experiment("Hetero-sched")
db = MongoObserver()
ex.observers.append(db)

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


@ex.capture
def end2end(model, subgraph, chip_id, dump_path):
    # model = data.get("network")
    # chips = data.get("chips")
    # dump_path = data.get("dump_path")
    # subgraph = data.get("subgraph")
    assert os.path.exists(subgraph)
    assert os.path.exists(model)

    ex.add_resource(model)
    ex.add_resource(subgraph)

    assert chip_id in supported_chips.keys()
    chip = supported_chips[chip_id]

    sol = solve(model, subgraph, chip, ILPSolver)
    dispatch_df = sol.dispatch_to_df()
    time_df = sol.emu_time_to_df()

    dispatch_df_f = dump_path + "-ilp.dispatch.csv"
    time_df_f = dump_path + "-ilp.emu_time.csv"

    dispatch_df.to_csv(dispatch_df_f)
    time_df.to_csv(time_df_f)

    ex.add_artifact(dispatch_df_f)
    ex.add_artifact(time_df_f)

    pipe = pipeline(sol)
    pipe_df = p.to_df()

    pipe_df_f = dump_path + "-ilp.pipe_time.csv"

    pipe_df.to_csv(pipe_df_f)
    ex.add_artifact(pipe_df_f)

    plot_f = dump_path + "-ilp.pareto.pdf"
    plot_scatter(pipe_df, plot_f)
    ex.add_artifact(plot_f)


ex.add_config("config/bev_former.yaml")
ex.add_config("config/inception_v1_khadas.yaml")

@ex.automain
def main():
    end2end()
    pass


# if __name__ == "__main__":
#     # df_net = "data/net_perf/bst_comm/bevformer_with_shape_detail_comm.csv"
#     # df_sg = "third_party/Partitioning-Algorithm/mapping_strategy/subgraphs/bevformer_bst.csv"
#     # sol = solve(df_net, df_sg, bst_chip, ILPSolver)
#     # sol_df = sol.dispatch_to_df()
#     # time_df = sol.emu_time_to_df()
#     # print(sol_df)
#     # print(time_df)
#     # print(sol_df.to_csv("sol-ilp.csv"))
#     # print(time_df.to_csv("time-ilp.csv"))

#     # p = pipeline(sol)
#     # pp = p.to_df()
#     # pp.to_csv("all-time-ilp.csv")

#     final = pd.read_csv("all-time-ilp.csv")
#     plot_scatter(final, "all-time-ilp.pdf")

