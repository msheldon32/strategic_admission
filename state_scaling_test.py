import random
import math

import numpy as np
import matplotlib.pyplot as plt

from agent import KnownPOAgent, DeterministicAgent, ACRLAgent, ClassicalACRLAgent, UCRLAgent
from model import generate_model, ModelBounds, RewardGenerator, StateRewards, Model
from observer import Observer
from policy import Policy

from simulator import Simulator

if __name__ == "__main__":
    rng = np.random.default_rng()
    all_bounds = [ModelBounds([5,5],[2**x,2**x]) for x in range(6)]
    #model_bounds.customer_ub = 4
    #model_bounds.server_ub = 4
    #model_bounds.abandonment_ub = 4
    model = generate_model(all_bounds[-1], RewardGenerator(rng), rng)
    state_rewards = model.state_rewards
    models = []

    for model_idx in range(6):
        # truncate the model accordingly
        lcap = 32
        side = 2**model_idx
        state_lb = lcap - side
        state_ub = lcap + side+1
        new_state_rewards = StateRewards(state_rewards.customer_rewards[state_lb:state_ub],state_rewards.server_rewards[state_lb:state_ub],state_rewards.abandonment_rewards[state_lb:state_ub],state_rewards.holding_rewards[state_lb:state_ub],)
        new_model = Model([side, side], model.customer_rates[state_lb:state_ub], model.server_rates[state_lb:state_ub], model.abandonment_rates[state_lb:state_ub], new_state_rewards)
        models.append(new_model)

    agents = []
    observers = []
    simulators = []
    ideal_agents = []
    for model, model_bounds in zip(models, all_bounds):
        #agent = ACRLAgent(model_bounds, model.state_rewards)
        agents.append(ACRLAgent(model_bounds, model.state_rewards))
        observers.append(Observer(model))
        simulators.append(Simulator(model,agents[-1], observers[-1], rng))
        ideal_agents.append(KnownPOAgent(model))

    for i in range(1000000000):
        if i % 1000000 == 0 and i != 0:
            #print(f"True model:")
            #print(model)
            #print(f"Optimistic model:")
            #print(agent.model)
            #confidence_param = agent.get_confidence_param()
            #agent.parameter_estimator.print_with_confidence(confidence_param)
            #raise Exception("stop")
            #ucrl_agent.print()
            for i, observer in enumerate(observers):
                observer.plot_regret(ideal_agents[i].get_estimated_gain(), "b")
            plt.show()
            #observer.plot_total_reward(ideal_agent.get_estimated_gain())

        for simulator in simulators:
            simulator.step()

    print("Simulated gain: ", observer.get_gain())
    print("Estimated gain: ", agent.get_estimated_gain())
