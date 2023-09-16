import pathlib
import pandas as pd

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

    def complete_path(self, name, ext = "csv"):
        return self.base_path + f"-{self.solver}.{name}.{ext}"


def load_dispatch(path):
    df = pd.read_csv(path)
    group_id = df["group_id"].unique()
    graph_id = df["graph_id"].unique()

    ret = {}

    for group in group_id:
        for graph in graph_id:
            ret[group, graph] = DispatchResult()

    for i in range(len(df)):
        ins = df.loc[i]
        g1 = ins["group_id"]
        g2 = ins["graph_id"]
        ret[g1,g2].set(ins["op_id"], ins["order"], ins["dispatch"], ins["graph_id"])

    return ret


if __name__ == "__main__":
    path = "results/bst/bevconv_pipeline-naive.dispatch.csv"
    print(load_dispatch(path))
