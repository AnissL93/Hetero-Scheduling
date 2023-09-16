#!/usr/bin/env python
import datetime
import pathlib
import argparse
import logging

if __name__ == "__main__":
    from emulator import async_emulation
    from processor import *
    from solver import *
    from cost_graph import *
    from pipeline import *
else:
    from .emulator import async_emulation
    from .processor import *
    from .solver import *
    from .cost_graph import *
    from .pipeline import *
    from .file import load_dispatch

import os
import sys

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_dir)


def get_model_name(model_file, chip, solver):
    file_name = pathlib.Path(model_file).stem
    current_timestamp = datetime.datetime.now().timestamp()
    date_time = datetime.datetime.fromtimestamp(current_timestamp)
    str_date_time = date_time.strftime("%d-%m-%Y-%H:%M:%S")
    return f"{chip}-{file_name}-{solver}-{str_date_time}"


def get_log_file_name(model_file, chip, solver):
    home = os.environ.get("HETERO_SCHEDULE_HOME")
    log_file = home + "/log/" + \
        get_model_name(model_file, chip, solver) + ".log"
    return log_file

def solve(model: str, subgraph : str, chip_id : str, solver : Solver, dispatch_result = None):
    """The entry function for solving for a model on chip

    Returns:
        Solution: contain dispatch graph for all possible groups
    """
    df_graph = pd.read_csv(model)
    df_subgraph = pd.read_csv(subgraph)

    by_position = False
    count_self = True

    if "bst" in chip_id:
        by_position = True
        # count this
        count_self = True

    chip = supported_chips[chip_id]
    graph = GraphCost(df_graph, df_subgraph, "g", chip, count_self, by_position)
    sol = Solution(graph, chip, solver, get_model_name(model, "bst", solver.ID))

    if dispatch_result is not None:
        sol.dispatch_results = dispatch_result

    sol.solve_and_run(dispatch_result is not None)
    return sol

def pipeline(sol, factor):
    pipeline = Pipeline(sol, factor)
    pipeline.eval_all(256)
    return pipeline

def main(model, subgraph, chip):
    ilp_sol = solve(model, chip, subgraph, ILPSolver)
    pipeline(ilp_sol)
    df = pipeline.to_df()
    ilp_name = get_model_name(model, chip, "ilp")

def dump_results(solution, p, chip : Chip):
    for i, gs in solution.items():
        c = chip.get_group_as_chip(i)
        for gi, g_one_sol in gs.items():
            g_one_sol.merge_dispatch()
            g_one_sol.draw_results(c, p.with_suffix(f".{i}.{gi}.pdf"))
            g_one_sol.dispatch_to_csv(dispatch_csv_file=p.with_suffix(f".{i}.{gi}.dispatch.csv"))
    pass
