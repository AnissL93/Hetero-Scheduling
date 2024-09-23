import pathlib
import pandas as pd
import logging
import networkx as nx
import math

if __name__ == "__main__":
    from schedule.cost_graph import DispatchResult
else:
    from .cost_graph import DispatchResult


class FileName(object):
    def __init__(self, base_path, solver, use_time_stamp=False):
        self.base_path = base_path
        self.solver = solver
        self.use_time_stamp = use_time_stamp
        pass

    def dispatch(self):
        return self.complete_path("dispatch")

    def emu_time(self):
        return self.complete_path("emu_time")

    def pipe_time(self):
        return self.complete_path("pipe_time")

    def pareto(self):
        return self.complete_path("pareto", "pdf")

    def solve_time(self):
        return self.complete_path("solve_time", "csv")

    def complete_path(self, name, ext="csv"):
        return self.base_path + f"-{self.solver}.{name}.{ext}"


def load_dispatch(path):
    df = pd.read_csv(path)
    group_id = df["group_id"].unique()
    graph_id = df["graph_id"].unique()

    ret = {}

    for group in group_id:
        group_df = df[df["group_id"] == group]
        for graph in graph_id:
            graph_df = group_df[group_df["graph_id"] == graph]
            ret[group, graph] = DispatchResult(graph)
            for i in graph_df.index:
                ins = graph_df.loc[i]
                disp = ins["dispatch"]
                if disp is None or disp == "null" or pd.isna(disp):
                    logging.info(f"Skip loading {ins}")
                    continue

                # add new elements
                ret[group, graph].set(str(ins["op_id"]), ins["order"], ins["dispatch"])
    # remove group that is all None
    return ret


def dump_graph(g, pdf_file, print_dot=True):
    tmp = nx.nx_agraph.to_agraph(g)
    if print_dot:
        print(tmp.string())
    tmp.draw(pdf_file, prog="dot")


if __name__ == "__main__":
    path = "results/bst/bevconv_pipeline-naive.dispatch.csv"
    d = load_dispatch(path)
    print(d["group0", 27])
    print(d["group0", 27].get_exec_order_vec())
