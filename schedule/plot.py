import logging

import matplotlib.pyplot as plt
import pandas as pd
from paretoset import paretoset
import ast
import numpy as np

def get_color(k):
    if k:
        return "blue"
    else:
        return "orange"

def scale_0_1(x):
    return (x-np.min(x)) / np.ptp(x)


def plot_pareto(ax, df : pd.DataFrame):
    logging.info("Plot pareto font for ")
    logging.info(df)

    perf = df["perf"]

    latency = []
    thr = []

    for i in perf:
        if isinstance(i, str):
            ii = ast.literal_eval(i)
        elif isinstance(i, tuple):
            ii = i

        latency.append(ii[0])
        thr.append(1./ ii[1])

    latency = scale_0_1(latency)
    thr = scale_0_1(thr)

    # get pareto
    data = pd.DataFrame({"latency" : latency, "throughput" : thr})
    mask = paretoset(data, sense=["min", "min"])
    pareto_front = data[mask]

    print(pareto_front)

    color = [get_color(x) for x in mask]

    ax.scatter(data["latency"],  data["throughput"], c = color, alpha=0.5)
    # make line of parato front
    # ax.plot(pareto_front["latency"],  pareto_front["throughput"])
    ax.set_xlabel("Latency")
    ax.set_ylabel("1 / Throughput")
    ax.grid()


