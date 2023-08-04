#!/usr/bin/env python
"""
The basic network structure.
"""
import networkx as nx
import logging
import pandas as pd
import ast
import networkx as nx
import copy

if __name__ == "__main__":
    from color_combin import get_color_combination
    from processor import *
    from processor import Chip, Processor
else:
    from .color_combin import get_color_combination
    from .processor import *
    from .processor import Chip, Processor


class OpCost(object):
    """
    {
        Processor : int,
        ...
    }
    """

    def __init__(self) -> None:
        self.backends = {}

    def set(self, d, x):
        self.backends[d] = int(x)

    def get(self, d):
        return self.backends[d]

    def get_types(self):
        return [b.type for b in self.backends.keys()]

    def get_by_type(self, dtype):
        for d, v in self.backends.items():
            if d.type == dtype:
                return v

        logging.error(f"Can not find device with type: {dtype}")
        assert False


class CommCost(object):
    """
    {
    (pA, pB) : int
    (pA, pC) : int
    (pB, pC) : int
    }
    """

    def __init__(self) -> None:
        self.comm_cost = {}

    def set(self, pA, pB, cost):
        assert isinstance(pA, Processor)
        assert isinstance(pB, Processor)
        self.comm_cost[pA, pB] = cost

    def get(self, pA, pB):
        assert isinstance(pA, Processor)
        assert isinstance(pB, Processor)
        if pA == pB:
            return 0
        else:
            if (pA, pB) not in self.comm_cost.keys():
                logging.fatal(f"Not found: {(pA.type, pB.type)}")
                exit(-1)
            return self.comm_cost[pA, pB]


class GraphCost(object):
    """
    helper nodes: super_start, super_end
    edge: index for communication cost
    nodes: index for computational cost
    """

    # SUPER_ENTRY_NODE = "super_start"
    # SUPER_EXIT_NODE = "super_exit"

    def __init__(self, df_graph: pd.DataFrame = None,
                 c: Chip = None):
        # The graph structure
        self.nx_graph = None
        # The cost of a operation: {"op_name" : {"CPU": 1, "GPU": 2}}
        self.op_cost = {}
        self.op_types = {}
        # comm_cost[edge[0], edge[1], pA, pB]
        self.comm_cost = {}
        self.chip = c

        if df_graph is not None:
            self.init_graph(df_graph, c)

    def init_graph(self, df: pd.DataFrame, chip: Chip):

        def _parse_structure():
            """
            op_id,op_type,suc,PType1_0,PType2_1...
            conv1,Conv,[relu1,conv2],10,20,...
            """
            # all_processors: ["CPU", "GPU"]
            # net: nx.DiGraph
            # cost: {"node1" : {"CPU": 10, "GPU": 15}, "node2": {"CPU": 20, "GPU": 15}}
            self.nx_graph = nx.DiGraph()
            self.op_types = {}
            for i in range(len(df)):
                node = df.loc[i]
                node_id = str(node["op_id"])

                # type
                node_type = str(node["op_type"])
                self.op_types[node_id] = node_type

                # successor
                suc_nodes = list(ast.literal_eval(node["suc"]))
                suc_nodes = [str(i) for i in suc_nodes]

                self.nx_graph.add_node(node_id, label=str(node_type))
                for suc in suc_nodes:
                    self.nx_graph.add_edge(node_id, suc)

                # compute
                compute_c = OpCost()
                for p in chip.types_set():
                    compute_c.set(p, node[p.type])

                self.op_cost[node_id] = compute_c

                # communication
                for suc in suc_nodes:
                    comm_c = CommCost()
                    for d1, d2 in chip.get_type_combinations():
                        dkey = str([d1.type, d2.type])
                        costs = ast.literal_eval(node[dkey])
                        for c in costs:
                            comm_c.set(d1, d2, c)

                    self.comm_cost[node_id, suc] = comm_c

        _parse_structure()
        self.topo_sort_ops = list(nx.topological_sort(self.nx_graph))

    # def init_compute_cost(self, df : pd.DataFrame):
    #     for i in range(len(df)):
    #         node = df.loc[i]
    #         node_id = str(node["op_id"])
    #         c = {}
    #         for p in self.chip.types_set():
    #             c[p] = node[p.type]
    #
    #         self.op_cost[node_id] = OpCost(c)
    #
    # def init_comm_cost(self, df :pd.DataFrame):
    #     for i in range(len(df)):
    #         instance = df.loc[i]
    #         node_pair = ast.literal_eval(str(instance["op_pair"]))
    #         assert isinstance(node_pair, list) and len(node_pair) == 2
    #         cc = CommCost()
    #         for d1, d2 in self.chip.get_type_combinations():
    #             dkey = str([d1, d2])
    #             cc.set(d1, d2, instance[dkey])
    #
    #         self.comm_cost[node_pair[0], node_pair[1]] = cc

    def topo_sort(self):
        return self.topo_sort_ops

    def get_edges(self):
        """
        Get edges in single direction
        """
        return self.nx_graph.edges()

    def get_pairs(self):
        """
        Get edges in both directions
        """
        return self.nx_graph.edges(data=True)

    def to_graphviz(self, pdf_file):
        tmp = nx.nx_agraph.to_agraph(self.nx_graph)  # convert to a graphviz graph
        tmp.draw(pdf_file, prog="dot")  # Draw with pygraphviz

    def get_exit_ops(self):
        return [n for n in self.topo_sort_ops if self.is_exit(n)]

    def get_entry_ops(self):
        return [n for n in self.topo_sort_ops if self.is_entry(n)]

    def is_entry(self, op):
        return len(self.prevs(op)) == 0

    def is_exit(self, op):
        return len(self.sucs(op)) == 0

    def sucs(self, op: str) -> list:
        return list(self.nx_graph.successors(op))

    def prevs(self, op: str) -> list:
        return list(self.nx_graph.predecessors(op))

    def get_op_type(self, id):
        if isinstance(id, int):
            name = self.topo_sort_ops[id]
            return self.op_types[name]

        elif isinstance(id, str):
            return self.op_types[id]

        else:
            logging.error(f"Fail to find type for {id}")
            assert False

    def get_compute_cost(self, id):
        if isinstance(id, int):
            name = self.topo_sort_ops[id]
            return self.op_cost[name]

        elif isinstance(id, str):
            return self.op_cost[id]

        else:
            logging.error(f"Fail to find cost for {id}")
            assert False

    def get_op_comm_cost(self, from_node, to_node):
        return self.comm_cost[from_node, to_node]

    def get_comm_cost_for_device(self, from_node, to_node, d1, d2):
        return self.get_op_comm_cost(from_node, to_node).get(d1, d2)

    def get_compute_cost_one_device(self, id, d):
        assert isinstance(d, Processor)
        return self.get_compute_cost(id).get_by_type(d.type)

    def draw_graph_structure(self, pdf_file):
        tmp = nx.nx_agraph.to_agraph(self.nx_graph)
        tmp.draw(pdf_file, prog="dot")

    def to_df(self):
        """
        Convert cost data to pd.DataFrame, the format is as following:
        op_id,op_type,suc,PA,PB,[PA,PB],[PB,PA]
        ...
        """

        def get_columns():
            return (["op_id", "op_type", "suc"] + [t.type for t in self.chip.types_set()] +
                    [str([t1.type, t2.type]) for t1, t2 in self.chip.get_type_combinations()])

        def get_op_id():
            return {"op_id": self.topo_sort()}

        def get_op_type():
            return {"op_type": list(self.op_types.values())}

        def get_sucs():
            return {"suc": [str(self.sucs(op)) for op in self.topo_sort()]}

        def get_compute_costs():
            ret = {}
            for p in self.chip.types_set():
                cost_for_p = []
                for op in self.topo_sort():
                    cost_for_p.append(self.get_compute_cost_one_device(op, p))
                ret[p.type] = cost_for_p

            return ret

        def get_comm_costs():
            ret = {}
            for d1, d2 in self.chip.get_type_combinations():
                cost_for_d1_d2 = []
                for op in self.topo_sort():
                    costs = []
                    for suc in self.sucs(op):
                        costs.append(self.get_comm_cost_for_device(op, suc, d1, d2))

                    cost_for_d1_d2.append(str(costs))

                ret[str([d1.type, d2.type])] = cost_for_d1_d2

            return ret

        data = {}
        data.update(get_op_id())
        data.update(get_op_type())
        data.update(get_sucs())
        data.update(get_compute_costs())
        data.update(get_comm_costs())
        print(data)
        return pd.DataFrame(data)

    def to_csv(self, file_name):
        """Dump the file to CSV format, converting with pandas."""
        self.to_df().to_csv(file_name)


class DispatchedGraph(GraphCost):
    def __init__(self, graph: GraphCost = None, dispatch={}):
        if graph is not None:
            self.__dict__.update(graph.__dict__)
            self.dispatch_results = dispatch

    def set_dispatch(self, n, p: str):
        """
        Dispatch node n to Processor p
        """
        assert n in self.topo_sort()
        self.dispatch_results[n] = p

    def validate(self):
        for k, v in self.dispatch_results.items():
            if v is None:
                print(f"Error: node {k} is not dispatched!")
                return False

        return True

    def init_dispatch(self, df: pd.DataFrame, chip: Chip):
        self.dispatch_results = {}
        for i in range(len(df)):
            self.dispatch_results[str(df.loc[i]["op_id"])] = chip.get_processor_by_id(df.loc[i]["dispatch"])

    def draw_results(self, chip: Chip, pdf_file):
        # Update node assignment
        col = get_color_combination(len(chip.ids()))
        for i in self.get_exec_order():
            for index, j in enumerate(chip.ids()):
                if self.dispatch_results[i] == j:
                    self.nx_graph.nodes[i]["color"] = col[index]
                    self.nx_graph.nodes[i]["shape"] = "Mrecord"
                    self.nx_graph.nodes[i]["style"] = "filled"
                    self.nx_graph.nodes[i]["fontname"] = "Helvetica"

        # add a legend for graph

        legend_head = "<<TABLE BORDER=\"0\" CELLBORDER=\"1\" CELLSPACING=\"0\" CELLPADDING=\"4\"> <TR> <TD COLSPAN=\"1\"><B>Legend</B></TD> </TR>"
        for i, p in enumerate(chip.ids()):
            legend_head += "<TR><TD BGCOLOR=\"{}\">{}</TD></TR>\n".format(col[i], p)

        legend_head += "</TABLE>>"

        tmp_graph = copy.deepcopy(self.nx_graph)
        tmp_graph.add_node("Legend", label=legend_head, shape="box")

        # tmp_graph.add_edge("Legend", self.get_exec_order()[0], style="invis")
        tmp = nx.nx_agraph.to_agraph(tmp_graph)  # convert to a graphviz graph
        tmp.draw(pdf_file, prog="dot")  # Draw with pygraphviz

    def get_dispatch(self, n):
        assert n in self.topo_sort()
        return self.dispatch_results[n]

    def get_exec_order(self):
        return list(self.dispatch_results.keys())

    def dispatch_to_df(self):
        assign = [self.dispatch_results[n] for n in self.get_exec_order()]
        data = {
            "op_id": self.get_exec_order(),
            "dispatch": assign
        }
        ddf = pd.DataFrame(data)
        return ddf

    def dispatch_to_csv(self, dispatch_csv_file):
        gdf = self.dispatch_to_df()
        gdf.to_csv(dispatch_csv_file)


if __name__ == "__main__":
    df = pd.read_csv("data/net_perf/bst/inception_v1_block.csv")
    df_comm = pd.read_csv("data/net_perf/bst/inception_v1_comm.csv")
    graph = GraphCost(df, bst_chip)
    # # print(df)
    df = graph.to_df()
    print(df)


    # read communication
    def preprocess_comm():
        df = pd.read_csv("data/net_perf/bst/inception_v1_block_compute.csv")
        data = {
            "op_id": df["op_id"],
            "op_type": df["op_type"],
            "suc": df["suc"],
        }

        for p in bst_chip.types_set():
            data[p] = df[p]

        # communication
        for p1, p2 in bst_chip.get_type_combinations():
            data[str([p1, p2])] = []

        # i->suc
        for i in range(len(df["op_id"])):
            ainstance = df.loc[i]
            write_f = int(ainstance["write"])
            comm_cost = []
            for suc in ast.literal_eval(ainstance["suc"]):
                read_t = int(df[df["op_id"] == suc]["read"])
                comm_cost.append(write_f + read_t)

            for p1, p2 in bst_chip.get_type_combinations():
                data[str([p1, p2])].append(str(comm_cost))
                pass

        print("=---------------")
        print(type(data))
        print(data.keys())
        print(data["['MACA', 'CV_DSP']"])

        df_comm = pd.DataFrame(data)
        df_comm.to_csv("inception_v1_block_comm.csv")
        pass

    # preprocess_comm()
