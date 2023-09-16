"""
Emulate the execution process, and record the total execution time.

Input:
  Graph structure with costs of 
  Chip configuration
  Execution order
  Processor dispatching
"""
from .cost_graph import GraphCost, DispatchedGraph,DispatchResult
from .processor import Processor, Chip
import numpy as np
import logging

class OpExecTime(object):

    def __init__(self):
        self.start = 0
        self.end = 0


class ExecTime(object):

    def __init__(self, exec_order):
        self.compute_time = {op: OpExecTime() for op in exec_order}

    def get_compute_st(self, op):
        return self.compute_time[op].start

    def get_compute_ed(self, op):
        return self.compute_time[op].end

    def set_compute_st(self, op, t):
        self.compute_time[op].start = t

    def set_compute_ed(self, op, t):
        self.compute_time[op].end = t

    def do_compute(self, op, cost):
        self.set_compute_ed(op, self.get_compute_st(op) + cost)

    def get_total_time(self):
        return max([e.end for k, e in self.compute_time.items()])

def async_emulation(graph_cost: GraphCost, dispatch : DispatchResult, chip: Chip):

    def validate():
        assert isinstance(graph, GraphCost)
        assert isinstance(chip, Chip)

    graph = DispatchedGraph(graph_cost)
    graph.dispatch_results = dispatch

    exec_order = graph.dispatch_results.get_exec_order_vec()
    print("exec order ", exec_order)
    exec_time = ExecTime(exec_order)

    def run():
        avail_proc = {op: {p_id: 0 for p_id in chip.ids()} for op in exec_order}

        for i,op in enumerate(exec_order):
            assigned_p_id = graph.get_dispatch(op)

            # current op cost
            op_cost = graph.get_dispatched_compute_cost(op)
            if op_cost <= 0:
                logging.fatal(f"Fatal: {op} cost is {op_cost}")
                exit(-1)

            # first node, end_time = cost_time
            if graph.is_entry(op):
                # add read time
                read_time = graph.get_read_cost(op, graph.get_dispatch(op))
                exec_time.do_compute(op, op_cost + read_time)
                continue

            # max(prev compute finish time + communication cost)
            array = []
            for p in graph.prevs(op):
                t = exec_time.get_compute_ed(p) + graph.get_dispatched_comm_cost(p, op)
                array.append(t)

            max_prev_finished_time = np.max( array)

            st = max(
                [avail_proc[op][assigned_p_id], max_prev_finished_time]
            )

            exec_time.set_compute_st(op, st)
            exec_time.do_compute(op, op_cost)
            # add write for
            if graph.is_exit(op):
                write_time = graph.get_write_cost(op, graph.get_dispatch(op))
                exec_time.do_compute(op, write_time)

            if i + 1 < len(exec_order):
                for p_id in chip.ids():
                    if p_id == assigned_p_id:
                        # update end time
                        avail_proc[exec_order[i + 1]][p_id] = exec_time.get_compute_ed(op)
                    else:
                        avail_proc[exec_order[i + 1]][p_id] = avail_proc[exec_order[i]][p_id]

    validate()
    run()
    return exec_time


def sequence_emulation(graph, chip, exec_order, proc_dispatch):
    pass


def ideal_emulation(graph, chip, exec_order, proc_dispatch):
    pass

def pipeline_emulation(graph, split_point):


    pass



if __name__ == "__main__":
    pass
