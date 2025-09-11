from experiment import *

import matplotlib.pyplot as plt
import pickle

def get_denom(self):
    pass

def analyze(folder, n_runs):
    for run_no in range(n_runs):
        with open(f"{folder}/run_{run_no}", "rb") as f:
            run_data = pickle.load(f)
        plt.plot(run_data["acrl"]["regret"][:1000], "b")
        plt.plot(run_data["ablation"]["regret"][:1000], "r")
        plt.plot(run_data["ucrl"]["regret"][:1000], "g")
        plt.show()


if __name__ == "__main__":
    analyze("exp_out/bound_25_25_states", 50)
