#!/usr/bin/env python
"""
The basic network structure.
"""
import logging
import pandas as pd
import ast
import networkx as nx
import copy


if __name__ == "__main__":
    from color_combin import get_color_combination
    from processor import *
    from processor import Chip, Processor
    from schedule.device import khadas_device
else:
    from .color_combin import get_color_combination
    from .processor import *
    from .processor import Chip, Processor
    from .schedule.device import khadas_device


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
                 df_subgraph: pd.DataFrame = None,
                 gid = "g", chip: Chip = None,  count_self_comm=True):
        # The graph structure
        self.nx_graph = None
        # The cost of a operation: {"op_name" : {"CPU": 1, "GPU": 2}}
        self.op_cost = {}
        self.op_types = {}
        # comm_cost[edge[0], edge[1], pA, pB]
        self.comm_cost = {}
        self.chip = chip

        # subgraph related
        self.subgraphs = {}
        self.nx_subgraph = None
        self.topo_sort_ops = None

        self.df_graph = df_graph
        self.df_subgraph = df_subgraph
        self.graph_id = gid

        if df_graph is not None:
            self.init_graph(df_graph, chip)

        if df_subgraph is not None:
            self.make_subgraphs(df_subgraph)

    def init_graph(self, df: pd.DataFrame, chip: Chip):
        self.df_graph = df

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
                # init comm_cost to 0
                for suc in suc_nodes:
                    self.comm_cost[node_id, suc] = CommCost()

                for d1, d2 in chip.get_type_combinations():
                    dkey = str((d1.type, d2.type))
                    for i, suc in enumerate(suc_nodes):
                        if dkey in node.keys():
                            costs = ast.literal_eval(node[dkey])
                            assert len(costs) == len(suc_nodes)
                            self.comm_cost[node_id, suc].set(d1, d2, costs[i])
                        else:
                            self.comm_cost[node_id, suc].set(d1, d2, 0)
                            logging.error(
                                "No communication is found, set cummunication to 0")
                    pass

        _parse_structure()

    def make_subgraphs(self, df: pd.DataFrame):
        self.df_subgraph = df
        sg_len = len(df)
        self.nx_subgraph = nx.DiGraph()
        for i in range(sg_len):
            sg_id = str(df.loc[i]["subgraph_id"])
            self.nx_subgraph.add_node(sg_id)
            succs = list(ast.literal_eval(df.loc[i]["succs"]))
            for suc in succs:
                self.nx_subgraph.add_edge(sg_id, str(suc))

            nodes = list(ast.literal_eval(df.loc[i]["nodes"]))
            node_list = [self.df_graph.loc[j]["op_id"] for j in nodes]

            sg_cost = GraphCost()
            sg_cost.nx_graph = self.nx_graph.subgraph(node_list)

            # simply copy the the full op set which is indexed by op_id
            sg_cost.op_cost = self.op_cost
            sg_cost.op_types = self.op_types
            sg_cost.comm_cost = self.comm_cost
            sg_cost.chip = self.chip

            self.subgraphs[sg_id] = sg_cost

    def topo_sort(self):
        if self.topo_sort_ops is None:
            self.topo_sort_ops = list(nx.topological_sort(self.nx_graph))

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

    def get_data_size(self, op):
        if isinstance(op, str):
            return self.df_graph[self.df_graph["op_id"] == op]["data_size"]
        else:
            assert False

    def get_read_cost(self, op, pid):
        if pid in bst_chip.ids():
            return int(self.df_graph[self.df_graph["op_id"] == op]["read"])

        elif pid in khadas_chip.ids():
            data_size = self.get_data_size(op)
            return khadas_device[pid].read(data_size)

    def get_write_cost(self, op, pid):
        if pid in bst_chip.ids():
            return int(self.df_graph[self.df_graph["op_id"] == op]["write"])

        elif pid in khadas_chip.ids():
            data_size = self.get_data_size(op)
            return khadas_device[pid].write(data_size)

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
                        costs.append(
                            self.get_comm_cost_for_device(op, suc, d1, d2))

                    cost_for_d1_d2.append(str(costs))

                ret[str([d1.type, d2.type])] = cost_for_d1_d2

            return ret

        data = {}
        data.update(get_op_id())
        data.update(get_op_type())
        data.update(get_sucs())
        data.update(get_compute_costs())
        data.update(get_comm_costs())
        return pd.DataFrame(data)

    def to_csv(self, file_name):
        """Dump the file to CSV format, converting with pandas."""
        self.to_df().to_csv(file_name)

    def to_dispatch_graph(self):
        g = DispatchedGraph()
        g.__dict__.update(self.__dict__)

        # convert all subgraph
        if len(self.subgraphs) > 0:
            for k, sg in self.subgraphs.items():
                self.subgraphs[k] = sg.to_dispatch_graph()

        return g

    
    def draw_graph_structure(self, pdf_file):
        tmp_graph = copy.deepcopy(self.nx_graph)
        # add subgraph structure
        tmp_graph.add_nodes_from(self.nx_subgraph.nodes)
        tmp_graph.add_edges_from(self.nx_subgraph.edges)

        tmp = nx.nx_agraph.to_agraph(tmp_graph)

        for i, sg in self.subgraphs.items():
            tmp_sg = nx.nx_agraph.to_agraph(sg.nx_graph)
            tmp_sg.draw(f"sg{i}_{pdf_file}", prog="dot")

        for i, sg in self.subgraphs.items():
            B = tmp.add_subgraph(sg.topo_sort(), name=f'cluster_{i}')
            B.graph_attr["color"] = "skyblue"
            B.graph_attr["label"] = f"SG{i}"
            B.graph_attr["label"] = f"SG{i}"
            B.graph_attr["fontcolor"] = "navyblue"

        tmp.draw(pdf_file, prog="dot")

class DispatchResult(dict):
    def __init__(self, gid = "g"):
        self.gid = gid
        pass

    def set(self, op, order, d, gid = None):
        if gid is None:
            gid = self.gid
        self[gid, op] = (order, d)

    def get(self, op, order, gid = None):
        if gid is None:
            gid = self.gid
        return self[gid, op]

    def get_exec_order(self, gid = None):
        if gid is None:
            gid = self.gid

        ret = {}
        for (_gid, op), (order, device) in self.items():
            if _gid == gid:
                ret[order] = op

        return ret

    def merge(self, other):
        """Merge dispatch results from all subgraphs to the parent graph
        """
        if len(self) > 0:
            self.clear()

        self.update(other)

    def to_df(self):
        # op_id, group_id, dispatch 
        d = []
        for key, dist in self.items():
            op, gid = key
            print(key)
            instance = [op, gid, dist]
            d.append(instance)

        return pd.DataFrame(d, columns = ["graph_id", "op_id", "dispatch"])

    def from_df(self, df : pd.DataFrame):
        length = len(df)
        if "graph_id" not in df.columns:
            for i in range(length):
                op = str(df.loc[i]["op_id"])
                dis = df.loc[i]["dispatch"]
                if isinstance(dis, str):
                    order, device = i, dis
                else:
                    order ,device = ast.literal_eval(dis)
                self.set(op, order, device)
        else:
            # add subgraph
            for i in range(length):
                gid = str(df.loc[i]["graph_id"])
                op = str(df.loc[i]["op_id"])
                order ,device = ast.literal_eval(df.loc[i]["dispatch"])
                self.set(op, order, device, gid)

class DispatchedGraph(GraphCost):
    def __init__(self, graph: GraphCost = None, dispatch: pd.DataFrame = None):
        if graph is not None:
            self.__dict__.update(graph.__dict__)

        self.dispatch_results = DispatchResult()
        self.dispatch_results.from_df(dispatch)

        # dispatch to subgraphs
        if len(self.subgraphs) > 0:
            for sgid in self.subgraphs.keys(): 
                self.subgraphs[sgid].set_dispatch(d)
                pass
                
    def dispatch_all_to(self, p):
        """Set all dispatching result to processor `p`
        """
        for i, n in enumerate(self.topo_sort()):
            self.dispatch_results.set(n, i, p)

    def validate(self):
        for k, v in self.dispatch_results.items():
            if v is None:
                logging.error(f"Error: node {k} is not dispatched!")
                return False

        return True

    def draw_results(self, chip: Chip, pdf_file):
        """
        Output the graph assignments
        """
        tmp_graph = copy.deepcopy(self.nx_graph)

        # add subgraph structure
        tmp_graph.add_node_from(self.nx_subgraph.nodes)
        tmp_graph.add_edges_from(self.nx_subgraph.edges)

        logging.info(self.get_exec_order())

        # Update node assignment
        col = get_color_combination(len(chip.ids()))
        for i in self.get_exec_order():
            for index, j in enumerate(chip.ids()):
                if self.dispatch_results[i] == j:
                    tmp_graph.nodes[i]["color"] = col[index]
                    tmp_graph.nodes[i]["shape"] = "Mrecord"
                    tmp_graph.nodes[i]["style"] = "filled"
                    tmp_graph.nodes[i]["fontname"] = "FreeSans"
                    tmp_graph.nodes[i][
                        "label"] = f"{i}\\n{self.op_types[i]}\\n{self.get_dispatched_compute_cost(i)}"

        # add a legend for graph
        legend_head = "<<TABLE BORDER=\"0\" CELLBORDER=\"1\" CELLSPACING=\"0\" CELLPADDING=\"4\"> <TR> <TD COLSPAN=\"1\"><B>Legend</B></TD> </TR>"
        for i, p in enumerate(chip.ids()):
            legend_head += "<TR><TD BGCOLOR=\"{}\">{}</TD></TR>\n".format(
                col[i], p)

        legend_head += "</TABLE>>"

        tmp_graph = nx.DiGraph(tmp_graph)
        tmp_graph.add_node("Legend", label=legend_head,
                           shape="box", fontname="FreeSans")

        logging.info("edges " + str(self.get_edges()))

        for e in self.get_edges():
            tmp_graph.edges[e]["label"] = self.get_dispatched_comm_cost(
                e[0], e[1])
            tmp_graph.edges[e]["fontname"] = "FreeSans"

        tmp = nx.nx_agraph.to_agraph(tmp_graph)  # convert to a graphviz graph
        for i, sg in self.subgraphs.items():
            B = tmp.add_subgraph(sg.topo_sort(), name=f'cluster_{i}')
            B.graph_attr["color"] = "black"
            B.graph_attr["label"] = f"SG #{i}"

        # tmp_graph.add_edge("Legend", self.get_exec_order()[0], style="invis")
        tmp.draw(pdf_file, prog="dot")  # Draw with pygraphviz

    def get_dispatch(self, n):
        assert n in self.topo_sort()
        return self.dispatch_results[n]

    def get_exec_order(self, gid = None):
        """Return a dict of execution order of operations
        """
        if gid is None:
            gid = self.graph_id

        return self.dispatch_results.get_exec_order(gid)

    def dispatch_to_df(self):
        return self.dispatch_results.to_df()

    def dispatch_to_csv(self, dispatch_csv_file):
        gdf = self.dispatch_to_df()
        gdf.to_csv(dispatch_csv_file)

    def get_dispatched_comm_cost(self, f, t):
        pf = self.get_dispatch(f)
        pt = self.get_dispatch(t)
        p_f = self.chip.get_processor_by_id(pf)
        p_t = self.chip.get_processor_by_id(pt)
        return self.get_comm_cost_for_device(f, t, p_f, p_t)

    def get_dispatched_compute_cost(self, op):
        p = self.chip.get_processor_by_id(self.get_dispatch(op))
        return self.get_compute_cost_one_device(op, p)

    def merge_dispatch(self):
        """Merge dispatch results from all subgraphs to the parent graph
        """
        if len(self.dispatch_results) > 0:
            self.dispatch_results.data.clear()

        for i, sg in self.subgraphs.items():
            self.dispatch_results.data.update(sg.dispatch_results)


if __name__ == "__main__":
    # df = pd.read_csv("data/net_perf/bst/inception_v1_block.csv")
    df_net = pd.read_csv(
        "data/net_perf/bst_comm/bev_conv_loaded_fix_sim_detail_comm.csv")
    df_sg = pd.read_csv(
        "third_party/Partitioning-Algorithm/mapping_strategy/subgraphs/bev_conv_bst.csv")

    df_dispatch = pd.read_csv("results/bst/bevformer_pipeline-bst.group0.ilp.dispatch.csv")

    graph = GraphCost(df_net, df_subgraph = df_sg, chip = bst_chip.get_group_as_chip("group0"))

    dist = DispatchResult()
    dist.from_df(df_dispatch)
    print(dist.to_df())
    print(dist.get_exec_order())

    print(graph.get_read_cost("Unsqueeze_12", "maca"))
    print(graph.get_write_cost("Unsqueeze_12", "maca"))
    # # print(df)
    df = graph.to_df()
    graph.draw_graph_structure("xxx.pdf")
    # print(df)

    # read communication

    def preprocess_comm():
        df = pd.read_csv("data/net_perf/bst/inception_v1_detail.csv")
        data = {
            "op_id": df["op_id"],
            "op_type": df["op_type"],
            "suc": df["suc"],
        }

        for p in bst_chip.types_set():
            data[p.type] = df[p.type]

        # communication
        for p1, p2 in bst_chip.get_type_combinations():
            data[str([p1.type, p2.type])] = []

        # i->suc
        for i in range(len(df["op_id"])):
            ainstance = df.loc[i]
            write_f = int(ainstance["write"])
            comm_cost = []
            for suc in ast.literal_eval(ainstance["suc"]):
                read_t = int(df[df["op_id"] == suc]["read"])
                comm_cost.append(write_f + read_t)

            for p1, p2 in bst_chip.get_type_combinations():
                data[str([p1.type, p2.type])].append(str(comm_cost))
                pass

        df_comm = pd.DataFrame(data)
        df_comm.to_csv("inception_v1_with_comm.csv")
        pass

    # preprocess_comm()
