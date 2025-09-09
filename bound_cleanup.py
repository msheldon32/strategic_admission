# glue script to remove and replace existing outputs

from experiment import *

import pickle

for run_no in range(100):
    with open(f"exp_out/bound_0/run_{run_no}", "rb") as f:
        exp_run = pickle.load(f)


    with open(f"exp_out/bound_0/run_{run_no}", "wb") as f:
        pickle.dump(exp_run.summarize(), f)
