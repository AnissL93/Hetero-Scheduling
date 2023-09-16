#!/usr/bin/env python
import os
import sys

import matplotlib.pyplot as plt

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_dir)

from sacred.observers import MongoObserver
from sacred import Experiment, Ingredient
from pprint import pprint

from schedule.cost_graph import *
from schedule.solver import *
from schedule.processor import *
from schedule.emulator import async_emulation
from schedule.execute import *
from schedule.plot import *
from schedule.file import *
import logging
import argparse
import pathlib
import datetime
import yaml

logging.basicConfig(
    level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="[%(filename)s:%(lineno)d] - %(levelname)s - %(message)s",  # Set the log message format
    datefmt="%Y-%m-%d %H:%M:%S",  # (Optional) Set the date format
)

ex = Experiment("Hetero-sched")
db = MongoObserver()
ex.observers.append(db)

# Get the current timestamp with microsecond precision
def draw_dispatch(sol : Solution, dot_file_prefix):
    graph = sol.origin_graph
    for group in sol.chip.get_proc_groups():
        dispatch = sol.dispatch_results[group]
        dgraph = DispatchedGraph(graph)
        dgraph.dispatch_results = dispatch
        fname = f"{dot_file_prefix}.{group}.pdf"
        dgraph.draw_results(sol.chip.get_group_as_chip(group), fname)
    pass

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

@ex.config
def cfg():
    chip_id = "khadas_chip"
    factor = 1e-9
    if "khadas" in chip_id:
        factor = 1e-6

    solver = "ilp"
    skip_solve = False

    config_file = None


@ex.capture
def end2end(model, subgraph, chip_id, dump_path, factor, solver, skip_solve, config_file):
    # model = data.get("network")
    # chips = data.get("chips")
    # dump_path = data.get("dump_path")
    # subgraph = data.get("subgraph")
    assert os.path.exists(subgraph)
    assert os.path.exists(model)

    ex.add_resource(model)
    ex.add_resource(subgraph)

    assert chip_id in supported_chips.keys()

    if solver == "naive":
        solver_type = MinimalSolver
    else:
        solver_type = ILPSolver

    file_name = FileName(dump_path, solver)
    dispatch = None
    if skip_solve:
        dispatch = load_dispatch(file_name.dispatch())

    sol = solve(model, subgraph, chip_id, solver_type, dispatch)

    dispatch_df = sol.dispatch_to_df()
    time_df = sol.emu_time_to_df()

    dispatch_df.to_csv(file_name.dispatch())
    time_df.to_csv(file_name.emu_time())

    ex.add_artifact(file_name.dispatch())
    ex.add_artifact(file_name.emu_time())

    pipe = pipeline(sol, factor)
    pipe_df = pipe.to_df()

    pipe_df_f = file_name.pipe_time()
    pipe_df.to_csv(pipe_df_f)
    ex.add_artifact(pipe_df_f)

    plot_f = file_name.pareto()
    fig, ax = plt.subplots()
    plot_pareto(ax, pipe_df)
    plt.tight_layout()
    plt.savefig(plot_f)

    ex.add_artifact(plot_f)

    # dump_dot_f = dump_path + f"-{solver}.dot."
    # draw_dispatch(sol, dump_dot_f)
    # ex.add_artifact(dump_dot_f)

    # dump_str_f = dump_path + f"-{solver}.csv"


ex.add_named_config("v4_bst" ,"config/inception_v4_bst.yaml")
ex.add_named_config("v1_bst" ,"config/inception_v1_bst.yaml")
ex.add_named_config("v3_bst" ,"config/inception_v3_bst.yaml")
ex.add_named_config("v2_bst" ,"config/inception_resnet_v2_bst.yaml")
ex.add_named_config("bevconv_bst" ,"config/bev_conv.yaml")
ex.add_named_config("bevformer_bst","config/bev_former.yaml")
# ex.add_config("config/inception_resnet_v2_khadas.yaml")
# ex.add_config("config/inception_v3_khadas.yaml")
# ex.add_config("config/inception_v4_khadas.yaml")
# ex.add_config("config/bev_former.yaml")
# ex.add_config("config/bev_conv.yaml")

@ex.automain
def main():
    end2end()
    pass


# final = pd.read_csv(sys.argv[1])
# plot_scatter(final, sys.argv[2])

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


