import random
import math

import numpy as np

from agent import KnownPOAgent, DeterministicAgent, ACRLAgent
from model import generate_model, ModelBounds, RewardGenerator
from observer import Observer
from policy import Policy

class Simulator:
    def __init__(self, model, agent, observer, rng: np.random._generator.Generator):
        self.model = model
        self.agent = agent
        self.observer = observer
        self.rng = rng

        self.state = self.model.capacities[1] # note that this is the 0-indexed definition, we begin at 0
        self.t = 0
        self.n = 0


    def step(self):
        # generate an exponential RV based on the total rate
        total_rate = self.model.get_total_rate(self.state)
        time_elapsed = self.rng.exponential(scale=1/total_rate)
        #time_elapsed = -math.log(random.random())/total_rate
        self.t += time_elapsed

        transition_probs = self.model.get_transition_probs(self.state)
        transition_type = self.rng.choice(self.model.transition_labels, p=transition_probs)

        accept = self.agent.act(self.state, transition_type)

        reward = self.model.get_reward(self.state, transition_type, accept, time_elapsed)
        #reward = self.model.get_mean_reward(self.state, self.agent.policy.limiting_types[self.state])*time_elapsed
        mean_reward = self.model.get_mean_reward(self.state, self.agent.policy.limiting_types[self.state])
        #print(f"mean reward: {mean_reward}")
        #print(f"state: {self.state}")
        #print(f"state vector: {self.model.get_reward_vector(self.agent.policy)}")
        self.observer.step_mean_rewards.append(mean_reward)

        next_state = self.model.get_next_state(self.state, transition_type, accept)

        self.agent.observe(self.state, next_state, transition_type, reward, time_elapsed)
        self.observer.observe(self.state, next_state, transition_type, reward, time_elapsed)

        if accept:
            pass
            #print(f"accepting transition({transition_type}): {self.model.get_job_population(self.state)}->{self.model.get_job_population(next_state)}")

        self.state = next_state

if __name__ == "__main__":
    rng = np.random.default_rng()
    model_bounds = ModelBounds([5,5],[-10,10])
    model = generate_model(model_bounds, RewardGenerator(rng), rng)

    #agent = KnownPOAgent(model)
    agent = ACRLAgent(model_bounds, model.state_rewards)
    #agent = DeterministicAgent(model, Policy.full_rejection_policy(model))
    #agent = DeterministicAgent(model, Policy.full_acceptance_policy(model))
    observer = Observer(model)
    simulator = Simulator(model, agent, observer, rng)

    for i in range(100000):
        """if i != 0 and i % 100000 == 0:
            print("i: ", i)
            print("Simulated gain: ", observer.get_gain())
            print("Estimated gain: ", agent.get_estimated_gain())
            print("Avg state rewards: ", observer.get_avg_reward_state())
            print("Mean state rewards: ", model.get_reward_vector(agent.policy))
            print("state: ", simulator.state)
            print("Empirical rates: ", observer.get_empirical_rates())
            print("Actual rates: ", [model.get_transition_rates(s, agent.policy.get_limiting_types(s)) for s in range(model.n_states)])
            print("Actual probabilities: ", model.get_steady_state_probs(agent.policy))
            print("Empirical probabilities: ", observer.get_empirical_probs())
            observer.reset()"""
        simulator.step()

    print("Simulated gain: ", observer.get_gain())
    print("Estimated gain: ", agent.get_estimated_gain())
