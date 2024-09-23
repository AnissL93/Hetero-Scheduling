from sacred.observers import MongoObserver
from sacred import Experiment
from pprint import pprint

def print_config(log):
    start_time = log["start_time"]
    stop_time = log["stop_time"]
    dur = stop_time - start_time
    config = log["config"]

    print("=============================")
    print(config)
    print(f"{start_time=}")
    print(f"{stop_time=}")
    print(f"duration: {dur.total_seconds()} sec")

ex = Experiment("Hetero-sched")
db = MongoObserver()
ex.observers.append(db)

res = list(db.runs.find())

for  val in res:
    print_config(val)