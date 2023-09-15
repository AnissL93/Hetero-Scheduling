from sacred.observers import MongoObserver
from sacred import Experiment
from pprint import pprint

ex = Experiment("Hetero-sched")
db = MongoObserver()
ex.observers.append(db)

res = list(db.runs.find())
pprint(res[-1])
