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
from schedule.plot import *
import argparse
import pathlib
import logging
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd

# python gen_figure.py

def pareto_multi_benchmark(file_list : list, pdf):

    fig, axis = plt.subplots(2,3)
    dfs = []
    for f in file_list:
        df = pd.read_csv(f)
        dfs.append(df)

    plot_pareto(axis[0,0], dfs[0])
    plot_pareto(axis[0,1], dfs[1])
    plot_pareto(axis[0,2], dfs[2])
    plot_pareto(axis[0,0], dfs[3])
    plot_pareto(axis[1,1], dfs[4])
    plot_pareto(axis[1,2], dfs[4])

    plt.tight_layout()

    plt.savefig(pdf)





