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
from schedule.device import *


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