import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re

import os
import sys

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_dir)

from schedule.cost_graph import *
from schedule.solver import *
from schedule.processor import *
from schedule.emulator import async_emulation
import argparse
import pathlib
import logging

# python gen_figure.py 

network_path="/home/huiying/projects/Hetero-Scheduling/data/net_perf/bst_comm/"
ilp_result_path="/home/huiying/projects/Hetero-Scheduling/results/bst/bst/"
baseline_result_path="/home/huiying/projects/Hetero-Scheduling/results/bst/baseline/"
results = {
      "chip" : "bst",
      "network_path": [
           "bev_conv_loaded_fix_sim_detail_comm.csv",
           "bevformer_with_shape_detail_comm.csv",
           "inception_resnet_v2_detail_comm.csv",
           "inception_v1_detail_comm.csv",
           "inception_v3_detail_comm.csv",
           "inception_v4_detail_comm.csv"
      ],
      "ilp_dispatch": [
          "bst-bev_conv_loaded_fix_sim_detail_comm-2023-09-05-15:47:56.dispatch.csv",
          "bst-bevformer_with_shape_detail_comm-2023-09-05-15:47:56.dispatch.csv",
          "bst-inception_resnet_v2_detail_comm-2023-09-05-15:47:56.dispatch.csv",
          "bst-inception_v1_detail_comm-2023-09-05-15:47:56.dispatch.csv",
          "bst-inception_v3_detail_comm-2023-09-05-15:47:56.dispatch.csv",
          "bst-inception_v4_detail_comm-2023-09-05-15:47:56.dispatch.csv",
      ],
      "baseline_dispatch" : [
          "bst-bev_conv_loaded_fix_sim_detail_comm-2023-09-12-10:35:40.dispatch.csv",
          "bst-bevformer_with_shape_detail_comm-2023-09-12-10:35:40.dispatch.csv",
          "bst-inception_resnet_v2_detail_comm-2023-09-12-10:35:40.dispatch.csv",
          "bst-inception_v1_detail_comm-2023-09-12-10:35:40.dispatch.csv",
          "bst-inception_v3_detail_comm-2023-09-12-10:35:40.dispatch.csv",
          "bst-inception_v4_detail_comm-2023-09-12-10:35:40.dispatch.csv"
      ]
    }


def run_network_scheduling(model, dispatch, chip):
    df_graph = pd.read_csv(model)
    df_dispatch = pd.read_csv(dispatch)
    graph = DispatchedGraph(GraphCost(df_graph, chip), df_dispatch)
    exec_time = async_emulation(graph, chip)
    return graph, exec_time

def get_data()
