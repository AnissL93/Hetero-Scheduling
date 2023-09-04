import os
import sys
import ast
import numpy as np
import pandas as pd

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_dir, "third_party/Partitioning-Algorithm"))

import stg.InceptionV1 as v1
import stg.InceptionV3 as v3
import stg.InceptionV4 as v4
import stg.InceptionResnetV2 as v2


def get_speed_from_MBS(x):
    return x


def get_speed_from_GBS(x):
    return x * 1000


parameters = {
    "Cortex-A53": {
        "libmemcpy": get_speed_from_MBS(2215.3),
        "read": get_speed_from_MBS(3467.3),  # load from A53
        "write": get_speed_from_MBS(7443.1),
        "cache-size": 256 * 1e3
    },
    "Cortex-A73": {
        "libmemcpy": get_speed_from_MBS(4933.4),
        "read": get_speed_from_MBS(9541.7),  # load from A53
        "write": get_speed_from_MBS(9657.8),
        "cache-size": 1e6
    },
    "GPU": {
        "map": get_speed_from_GBS(7864.19),
        "unmap": get_speed_from_GBS(5778.92),
        "read": get_speed_from_GBS(7.58),
        "write": get_speed_from_GBS(7.58),
        "cache-size": 256 * 1e3
    }
}

class Processor:
    """
    Get time in us (1e-6)
    """
    def __init__(self, n):
        self.name = n

    def read(self, size):
        return size / parameters[self.name]["read"]

    def write(self, size):
        return size / parameters[self.name]["write"]

    def copy(self, size):
        return size / parameters[self.name]["libmemcpy"]

    def to_same_device(self, size):
        if size > parameters[self.name]["cache-size"]:
            return self.read(size) + self.write(size)
        else:
            return 0


class CPU(Processor):
    def __init__(self, n):
        super().__init__(n)
        pass

    def copy(self, size):
        return size / parameters[self.name]["libmemcpy"]

    def to_cpu(self, size, other_cpu):
        if other_cpu.name == self.name:
            return self.to_same_device(size)
        else:
            return self.write(size) + other_cpu.read(size)

    def to_gpu(self, size, gpu):
        return GPU.map(size) + GPU.unmap(size) + self.write(size) + self.copy(size) + gpu.read(size)

class GPU(Processor):

    def __init__(self, n):
        super().__init__(n)
        pass

    def map(size):
        return size / parameters["GPU"]["map"]

    def unmap(size):
        return size / parameters["GPU"]["unmap"]

    def to_cpu(self, size, cpu : Processor):
        return GPU.map(size) + GPU.unmap(size) + self.write(size) + cpu.read(size)

p = {"cpu_b":CPU("Cortex-A73"), "cpu_s" : CPU("Cortex-A53"), "gpu" : GPU("GPU")}

def transfer(size, p1, p2):
    if p1.name == p2.name:
        return p1.to_same_device(size)
    if p1.name == "GPU":
        return p1.to_cpu(size, p2)
    if p2.name == "GPU":
        return p1.to_gpu(size, p2)
    else:
        return p1.to_cpu(size, p2)

def create_csv(net):
    """
    op_id,op_type,suc,cpu_b,cpu_s,gpu
    """
    df = pd.DataFrame()
    op_id = list(net.model_dag.keys())
    suc = []
    cpu_b = []
    cpu_s = []
    gpu = []
    data = {}
    data_size = []

    comm_costs = {(f, t) : [] for f in p.keys() for t in p.keys()}
    
    for i in op_id:
        cpu_b.append(net.comp_cost_matrix_f32[i][0])
        cpu_s.append(net.comp_cost_matrix_f32[i][1])
        gpu.append(net.comp_cost_matrix_f32[i][2])
        suc.append(str(list(net.model_dag[i])))
        suc_ds = str([net.comm_data_matrix[i][s] * 4 for s in net.model_dag[i]])
        data_size.append(suc_ds)

        for k in comm_costs.keys():
            f = k[0]
            t = k[1]
            c = str([transfer(net.comm_data_matrix[i][s] * 4, p[f], p[t]) for s in net.model_dag[i]])
            comm_costs[k].append(c)

    data = {
        "op_id": op_id,
        "op_type": op_id,
        "suc": suc,
        "cpu_b": cpu_b,
        "cpu_s": cpu_s,
        "gpu": gpu,
        "data_size": data_size
    }
    data.update(comm_costs)


    return pd.DataFrame(data)


create_csv(v1).to_csv("InceptionV1.csv")
create_csv(v3).to_csv("InceptionV3.csv")
create_csv(v4).to_csv("InceptionV4.csv")
create_csv(v2).to_csv("InceptionResnetV2.csv")


print(transfer(3211264, p["cpu_b"], p["cpu_s"]))