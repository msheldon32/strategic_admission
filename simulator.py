import random
import math

import numpy as np
import matplotlib.pyplot as plt

from agent import KnownPOAgent, DeterministicAgent, ACRLAgent, ClassicalACRLAgent, UCRLAgent, AblationACRLAgent
from model import generate_model, ModelBounds, RewardGenerator
from observer import Observer
from policy import Policy

class Simulator:
    def __init__(self, model, agent, observer, rng: np.random._generator.Generator):
        self.model = model
        self.agent = agent
        self.observer = observer
        self.rng = rng

        self.state = 0#self.model.capacities[1] # note that this is the 0-indexed definition, we begin at 0 in the standard one
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
        #mean_reward = self.model.get_mean_reward(self.state, self.agent.policy.limiting_types[self.state])
        #print(f"mean reward: {mean_reward}")
        #print(f"state: {self.state}")
        #print(f"state vector: {self.model.get_reward_vector(self.agent.policy)}")
        #self.observer.step_mean_rewards.append(mean_reward)

        next_state = self.model.get_next_state(self.state, transition_type, accept)

        action_no = 1 if accept else 0

        self.agent.observe(self.state, next_state, action_no, transition_type, reward, time_elapsed)
        self.observer.observe(self.state, next_state, transition_type, reward, time_elapsed)

        if accept:
            pass
            #print(f"accepting transition({transition_type}): {self.model.get_job_population(self.state)}->{self.model.get_job_population(next_state)}")

        self.state = next_state

if __name__ == "__main__":
    input("Bounds are now flat")
    rng = np.random.default_rng()
    model_bounds = ModelBounds([3,3],[25,25])
    #model_bounds.customer_ub = 4
    #model_bounds.server_ub = 4
    #model_bounds.abandonment_ub = 4
    model = generate_model(model_bounds, RewardGenerator(rng), rng)

    ideal_agent = KnownPOAgent(model)
    agent = ACRLAgent(model_bounds, model.state_rewards)
    agent_ablation = AblationACRLAgent(model_bounds, model.state_rewards)
    ucrl_agent = UCRLAgent(model_bounds, model.state_rewards)
    observer = Observer(model)
    ablation_observer = Observer(model)
    ideal_observer = Observer(model)
    ucrl_observer = Observer(model)
    simulator = Simulator(model, agent, observer, rng)
    simulator2 = Simulator(model, ideal_agent, ideal_observer, rng)
    simulator_ucrl = Simulator(model, ucrl_agent, ucrl_observer, rng)
    simulator_ablation = Simulator(model, agent_ablation, ablation_observer, rng)

    fap = Policy.full_acceptance_policy(model)

    for i in range(1000000000):
        if i == 100:
            initial_value = observer.get_past_n_gain(100)
        if i != 0 and i % 10000 == 0 and i > 1000:
            print(f"steps before episode: {agent.exploration.steps_before_episode}")
            #agent.parameter_estimator.print_with_confidence(agent.initial_confidence_param/agent.exploration.steps_before_episode)
            print("Trailing gain (learning): ", observer.get_past_n_gain(10000))
            print("Trailing gain (ablation): ", ablation_observer.get_past_n_gain(10000))
            print("Baseline gain (naive admission): ", model.get_gain_bias(fap)[1])
            print("Optimistic gain (learning): ", agent.get_estimated_gain())
            print("Trailing gain (ideal): ", ideal_observer.get_past_n_gain(10000))
            print("Ideal gain (ideal): ", ideal_agent.get_estimated_gain())
            print("Trailing gain (ucrl): ", ucrl_observer.get_past_n_gain(10000))
            #print(f"True model:")
            #print(model)
            #print(f"Optimistic model:")
            #print(agent.model)
            #print(agent.parameter_estimator.transition_counts)
        if i % 100000 == 0 and i != 0:
            #print(f"True model:")
            #print(model)
            #print(f"Optimistic model:")
            #print(agent.model)
            #confidence_param = agent.get_confidence_param()
            #agent.parameter_estimator.print_with_confidence(confidence_param)
            #raise Exception("stop")
            #ucrl_agent.print()
            observer.plot_regret(ideal_agent.get_estimated_gain(), "b")
            #ucrl_observer.plot_regret(ideal_agent.get_estimated_gain(), "r")
            ablation_observer.plot_regret(ideal_agent.get_estimated_gain(), "r")
            plt.show()
            #observer.plot_total_reward(ideal_agent.get_estimated_gain())

        simulator.step()
        simulator2.step()
        simulator_ucrl.step()
        simulator_ablation.step()

    print("Simulated gain: ", observer.get_gain())
    print("Estimated gain: ", agent.get_estimated_gain())
