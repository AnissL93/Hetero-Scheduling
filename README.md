
Current situation:
- Modeled only computation costs, tested on bst for all benchmarks, see results/bst

Issue: 
- Wether to use M or 2*M in the constraints?

# ILP-based Heterogeneous scheduling solver and execution time estimation

## Dependency
    - gurobi (use this as the MIP solver)
    - graphviz

## Installation

```bash
pip install -r requirements.txt
```

Set Variables

 HETERO_SCHEDULE_HOME


## Scheduling and the estimate

```bash
python scripts/solve_and_run_network.py --model data/net_perf/arm/InceptionV3_block.csv --chip khadas
```


## Estimate the network with a given dispatching strategy  

```bash
python scripts/run_network.py --model data/net_perf/arm/InceptionV3.csv --dispatch inceptionv3_dispatch.csv --chip khadas

```

## Data Format

The network and performance data are stored in .csv format.


