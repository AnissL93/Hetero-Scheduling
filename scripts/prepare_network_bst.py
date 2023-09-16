import numpy as np
import pandas as pd
import os
import pathlib
import ast

procs = ["maca", "cv_dsp"]

def get_comm_data(path, file_name):
    df = pd.read_csv(path+ "/" + file_name)
    comm = {str((p1, p2)) : [] for p1 in procs for p2 in procs}
    for i in range(len(df)):
        instance = df.loc[i]
        for f in procs:
            for t in procs:
                key = str((f, t))
                write = instance["write"]
                suc_comms = []
                for suc in ast.literal_eval(instance["suc"]):
                    suc_instance = df.loc[df["op_id"] == suc]
                    suc_instance = dict(suc_instance)
                    read = int(suc_instance["read"])
                    # print(f"read {read}")
                    # print(f"{suc} read is {read}")
                    suc_comms.append(write + read)

                # print(f"suc_comm {str(suc_comms)}")
                comm[key].append(list(suc_comms))
                # print(f"{comm[key]} is {suc_comms}")

    print(comm)
    for k in comm.keys():
        df[str(k)] = comm[k]

    new_file = pathlib.Path(file_name).stem + "_comm.csv"
    new_path = path + "/" + new_file
    print("Save csv to ", new_path)
    df.to_csv(new_path)

    print(df)

p = "./data/net_perf/bst"
get_comm_data(p, "inception_resnet_v2_detail.csv")
# for name in os.listdir(p):
#     if "detail" in name and "comm" not in name and "resnet" in :
#         print(name)
#         get_comm_data(p, name)
