#!/usr/bin/env python
import datetime
import pathlib
import argparse
import logging
from .emulator import async_emulation
from .processor import *
from .solver import *
from .cost_graph import *
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


def solve(model: str, chip: Chip, subgraph=None):
    df_graph = pd.read_csv(model)
    df_subgraph = pd.read_csv(subgraph)
    graph = GraphCost(df_graph, chip)

    # Make subgraphs.
    if subgraph is not None:
        graph.make_subgraphs(df_subgraph, df_graph)

    graph = graph.to_dispatch_graph()

    if chip.groups is not None:
        solution = {}
        for stg_id, stg_procs in chip.groups.items():
            sub_chip = chip.get_group_as_chip(stg_id)
            new_graph_ilp = copy.deepcopy(graph)
            new_graph_minimal = copy.deepcopy(graph)
            new_graph_ilp = solveDag(ILPSolver, new_graph_ilp, sub_chip,
                                   stg_id + "-" + get_model_name(model, chip, "ilp"))
            new_graph_minimal = solveDag(MinimalSolver, new_graph_minimal, sub_chip,
                                   stg_id + "-" + get_model_name(model, chip, "naive"))

            solution[stg_id] = {"ilp": new_graph_ilp, "naive": new_graph_minimal}

    return solution


def dump_results(solution, p, chip : Chip):
    for i, gs in solution.items():
        c = chip.get_group_as_chip(i)
        for gi, g_one_sol in gs.items():
            g_one_sol.merge_dispatch()
            g_one_sol.draw_results(c, p.with_suffix(f".{i}.{gi}.pdf"))
            g_one_sol.dispatch_to_csv(dispatch_csv_file=p.with_suffix(f".{i}.{gi}.dispatch.csv"))
    pass

def estimate_for_groups(solution : dict, chip : Chip):
    """Return estimation data for each subgraph run on all possible processor groups
    """
    _i, g = list(solution.items())[0]
    data = {
       "subgraph_id" : list(g["ilp"].subgraphs.keys()),
    }
    for i, sol in solution.items():
        for k, s in sol.items():
            l = []
            c = chip.get_group_as_chip(i)

            for j,sg in s.subgraphs.items():
                time = async_emulation(sg, c)
                l.append(time.get_total_time())

            data[i + "-" + k] = l

    return pd.DataFrame(data)