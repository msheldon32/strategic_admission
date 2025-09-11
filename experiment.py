import model
import simulator
import agent
import observer

import numpy as np
import pickle

class ExperimentRun:
    def __init__(self, model_bounds, rng, max_step_count):
        self.model = model.generate_model(model_bounds, model.RewardGenerator(rng), rng)
        self.model_bounds = model_bounds
        self.rng = rng
        self.max_step_count = max_step_count

        self.ideal_agent = agent.KnownPOAgent(self.model)
        self.agent = agent.ACRLAgent(self.model_bounds, self.model.state_rewards)
        self.ablation_agent = agent.AblationACRLAgent(self.model_bounds, self.model.state_rewards)
        self.ucrl_agent = agent.UCRLAgent(self.model_bounds, self.model.state_rewards)

        self.acrl_observer = observer.Observer(self.model)
        self.acrl_simulator = simulator.Simulator(self.model, self.agent, self.acrl_observer, self.rng)
        
        self.ablation_observer = observer.Observer(self.model)
        self.ablation_simulator = simulator.Simulator(self.model, self.ablation_agent, self.ablation_observer, self.rng)

        self.ucrl_observer = observer.Observer(self.model)
        self.ucrl_simulator = simulator.Simulator(self.model, self.ucrl_agent, self.ucrl_observer, self.rng)

    def run(self):
        for i in range(self.max_step_count):
            self.acrl_simulator.step()
            self.ablation_simulator.step()
            self.ucrl_simulator.step()

    def summarize(self):
        ideal_gain = self.ideal_agent.get_estimated_gain()
        timestep = 10000
        return {
                "acrl": self.acrl_observer.summarize(ideal_gain, timestep=timestep),
                "ablation": self.ablation_observer.summarize(ideal_gain, timestep=timestep),
                "ucrl": self.ucrl_observer.summarize(ideal_gain, timestep=timestep)
            }

class Experiment:
    def __init__(self, all_bounds, runs_per_bound, rng, max_step_count):
        self.all_bounds = all_bounds
        self.runs_per_bound = runs_per_bound
        self.max_step_count = max_step_count
        self.rng = rng

        #self.runs = [[ExperimentRun(bound, self.rng, self.max_step_count) for i in range(self.runs_per_bound)] for bound in self.all_bounds]
        self.failed_runs = []

    def run(self):
        for i, bound in enumerate(self.all_bounds):
            lstate = bound.capacities[0]
            rstate = bound.capacities[1]
            for run in range(self.runs_per_bound):
                exp_run = ExperimentRun(bound, self.rng, self.max_step_count)
                try:
                    exp_run.run()
                except Exception as e:
                    self.failed_runs.append([run, str(e)])
                    continue
                with open(f"exp_out/bound_{lstate}_{rstate}_states/run_{run}", "wb") as f:
                    pickle.dump(exp_run.summarize(), f)

if __name__ == "__main__":
    # schedule: 
    # seed 1000: 6 types, 11 states
    # seed 2000: 6 types, 21 states
    # seed 3000: 6 types, 51 states
    # seed 4000: 6 types, 101 states
    rng = np.random.default_rng(seed=3000)
    bounds = [
            model.ModelBounds([3,3],[5,5]),
            #model.ModelBounds([3,3],[10,10]),
            #model.ModelBounds([3,3],[25,25]),
            #model.ModelBounds([3,3],[50,50]),
            ]

    experiment = Experiment(bounds, 50, rng, 2000000)

    experiment.run()
