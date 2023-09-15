import matplotlib.pyplot as plt
import pandas as pd
from paretoset import paretoset
import ast
import numpy as np

def scale_0_1(x):
    return (x-np.min(x)) / np.ptp(x)

def plot_scatter(df : pd.DataFrame, pdf_file):
    perf = df["perf"]

    latency = [ ast.literal_eval(i)[0] for i in perf]
    thr = [1. / ast.literal_eval(i)[1] for i in perf]
    # latency = scale_0_1(latency)
    # thr = scale_0_1(thr)
    data = pd.DataFrame({"latency" : latency, "throughput" : thr})
    mask = paretoset(data, sense=["min", "min"])
    pareto_front = data[mask]
    print(pareto_front)

    fig, ax = plt.subplots()
    def get_color(k):
        if k:
            return "blue"
        else:
            return "orange"

    color = [get_color(x) for x in mask]

    print(color)
    ax.scatter(latency, thr, c = color, alpha=0.5)
    
    fig.tight_layout()
    plt.savefig(pdf_file)
    
