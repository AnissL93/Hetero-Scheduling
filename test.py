# """https://graphviz.org/Gallery/directed/cluster.html"""

# import pymongo
# import sacred
from sacred.observers import MongoObserver
# from sacred.observers import TinyDbReader
# from sacred.observers import TinyDbObserve


from sacred import Experiment
from pprint import pprint


import numpy as np

np.array([1, 2, 3]) + np.array([4, 5, 6])

db = MongoObserver()
ex = Experiment("configs")
ex.observers.append(db)

res = list(db.runs.find())
pprint(res[-1])

@ex.config
def my_config():
    foo = 42
    bar = 'baz'

@ex.capture
def some_function(a, foo, bar=10):
    print(a, foo, bar)
    ex.add_artifact("ff.txt")

@ex.automain
def my_main():
    some_function(1, 2, 3)     #  1  2   3
    some_function(1)           #  1  42  'baz'
    some_function(1, bar=12)   #  1  42  12
    some_function(2)            #  TypeError: missing value for 'a'


#!/usr/bin/env python
# coding=utf-8
""" A very configurable Hello World. Yay! """


# @ex.named_config
# def rude():
#     """A rude named config"""
#     recipient = "bastard"
#     message = "Fuck off you {}!".format(recipient)


# @ex.config
# def cfg():
#     recipient = "world"
#     message = "Hello {}!".format(recipient)

# @ex.config
# def cfg2():
#     recipient = "worlddddddddddd"
#     message = "Hello {}!".format(recipient)


# @ex.automain
# def main(message):
#     print(__name__)
#     print(message)

import numpy as np
