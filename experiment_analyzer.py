from experiment import *

import matplotlib.pyplot as plt
import pickle

def get_denom(self):
    pass

def analyze(folder, n_runs):
    avg_regret = None
    abl_regret = None
    ucrl_regret = None
    gain = None

    for run_no in range(n_runs):
        with open(f"{folder}/run_{run_no}", "rb") as f:
            run_data = pickle.load(f)
            run_data["acrl"]["regret"] = run_data["acrl"]["regret"]
            run_data["ablation"]["regret"] = run_data["ablation"]["regret"]
        print(f"total reward: {run_data['acrl']['reward'][-1]}")
        print(f"total time: {run_data['acrl']['time'][-1]}")
        print(f"ideal gain: {run_data['acrl']['ideal_gain']}")
        def normalize(l):
            return [x/run_data["acrl"]["ideal_gain"] for x in l]
        if avg_regret is None:
            avg_regret = [[x,1] for x in normalize(run_data["acrl"]["regret"])]
            abl_regret = [[x,1] for x in normalize(run_data["ablation"]["regret"])]
            ucrl_regret = [[x,1] for x in normalize(run_data["ucrl"]["regret"])]
            gain = [[x/y,1] for x, y in zip(normalize(run_data["acrl"]["reward"]), run_data["acrl"]["time"])]
        else:
            for i, x in enumerate(normalize(run_data["acrl"]["regret"])):
                avg_regret[i][0] += x
                avg_regret[i][1] += 1
            for i, x in enumerate(normalize(run_data["ablation"]["regret"])):
                abl_regret[i][0] += x
                abl_regret[i][1] += 1
            for i, x in enumerate(normalize(run_data["ucrl"]["regret"])):
                ucrl_regret[i][0] += x
                ucrl_regret[i][1] += 1
            i = 0
            for x,y in zip(normalize(run_data["ucrl"]["reward"]), run_data["acrl"]["time"]):
                gain[i][0] += (x/y)
                gain[i][1] += 1
                i += 1
        #plt.plot(normalize(run_data["acrl"]["regret"]), "b")
        #plt.plot(normalize(run_data["ablation"]["regret"]), "r")
        #plt.plot(normalize(run_data["ucrl"]["regret"]), "g")
        #plt.show()
    avg_regret = [x/n for x, n in avg_regret]
    abl_regret = [x/n for x, n in abl_regret]
    ucrl_regret = [x/n for x, n in ucrl_regret]
    gain = [x/n for x, n in gain]
    plt.plot(avg_regret, "b")
    plt.plot(abl_regret, "r")
    #plt.plot(ucrl_regret, "g")
    plt.show()

    plt.plot(gain, "b")
    plt.show()


if __name__ == "__main__":
    analyze("exp_out/bound_5_5_states", 50)
