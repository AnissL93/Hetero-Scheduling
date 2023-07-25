"""
Emulate the execution process, and record the total execution time.

Input:
  Graph structure with costs of 
  Chip configuration
  Execution order
  Processor dispatching
"""
from .cost_graph import GraphCost, OpCost
from .processor import Processor, Chip
import numpy as np

class OpExecTime(object):

    def __init__(self):
        self.start = 0
        self.end = 0

class ExecTime(object):

    def __init__(self, exec_order):
        self.exec_time = {op : OpExecTime() for op in exec_order}

    def get_st(self, op):
        return self.exec_time[op].start

    def get_ed(self, op):
        return self.exec_time[op].end

    def set_st(self, op, t):
        self.exec_time[op].start = t

    def set_ed(self, op, t):
        self.exec_time[op].end = t

    def get_total_time(self):
        return max([e.end for k,e in self.exec_time.items()])

def async_emulation(graph: GraphCost, chip : Chip, exec_order, proc_dispatch):
    def validate():
        assert isinstance(proc_dispatch, dict)
        assert isinstance(graph, GraphCost)
        assert isinstance(chip, Chip)

        assert proc_dispatch.keys()
        assert len(proc_dispatch.keys()) == len(exec_order)


    exec_time = ExecTime(exec_order)

    def run():
        avail_proc = {op : {p_id : 0 for p_id in chip.ids()} for op in exec_order}

        for i, op in enumerate(exec_order):
            assigned_p = proc_dispatch[op]

            # current op cost
            op_cost = graph.get_op_cost_one_device(op, chip.get_processor_by_id(assigned_p).type)
            if op_cost <=0:
                print(f"Fatal: {op} cost is {op_cost}")
                exit(-1)
            
            # first node, end_time = cost_time
            if graph.is_entry(op):
                exec_time.set_ed(op, exec_time.get_st(op) + op_cost)
                continue

            max_prev_finished_time = np.max(
                np.array([exec_time.get_ed(p) for p in graph.prevs(op)])
            )

            st = np.max(
                np.array([avail_proc[op][assigned_p], max_prev_finished_time])
            )

            exec_time.set_st(op, st)
            exec_time.set_ed(op, st + op_cost)

            if i+1 < len(exec_order):
                for p_id in chip.ids():
                    if p_id == assigned_p:
                        # update end time
                        avail_proc[exec_order[i+1]][p_id] = exec_time.get_ed(op)
                    else:
                        avail_proc[exec_order[i+1]][p_id] = avail_proc[exec_order[i]][p_id]


    validate()
    run()
    return exec_time

def sequence_emulation(graph, chip, exec_order, proc_dispatch):

    pass

def ideal_emulation(graph, chip, exec_order, proc_dispatch):
    pass

if __name__ == "__main__":
    pass
    
