import math
import copy
import random

import model

import numpy as np
from scipy.stats import chi2

import policy

class ParameterEstimator:
    def __init__(self, model_bounds):
        self.model_bounds = model_bounds

        self.change_counts = [[[0 for k in range(3)] for j in range(model_bounds.n_actions)] for i in range(model_bounds.n_states)]
        self.rewards = [[[] for j in range(model_bounds.n_actions)] for i in range(model_bounds.n_states)]

        self.max_reward = 1
        self.min_reward = -1

    def get_change_idx(self, state, next_state):
        return (next_state-state)+1

    def observe(self, state, next_state, action, time_elapsed, reward):
        change_idx = self.get_change_idx(state, next_state)
        self.transition_counts[state][action][change_idx] += 1
        clipped_reward = max(min(reward, self.max_reward), self.min_reward)
        self.rewards[state][action].append(clipped_reward)

    def change_prob_estimate(self, state, action):
        ct = sum(self.change_counts[state][action])

        if total_n_transitions == 0:
            if state == 0:
                return [0, 0.5, 0.5]
            elif state == sum(self.model_bounds.capacities):
                return [0.5, 0.5, 0]
            return [1/3, 1/3, 1/3]

        return [x/ct for x in self.change_counts[state][action]]

    def change_prob_epsilon(self, state, action, confidence_param):
        ct = sum(self.change_counts[state][action])
        if ct== 0:
            return 2

        inner_term = ((14*self.model_bounds.n_states)/ct)*math.log(2*(self.model_bounds.n_actions)/confidence_param)
        return math.sqrt(inner_term)

    def reward_estimate(self, state, action):
        ct = len(self.rewards[state][action])
        total = sum(self.rewards[state][action])

        return total/ct

    def reward_epsilon(self, state, action, confidence_param):
        ct = len(self.rewards[state][action])

        inner_term = (7/(2*ct))*math.log((2*self.model_bounds.n_actions*self.model_bounds.n_states)/confidence_param)
        return math.sqrt(inner_term)

    def reward_ub(self, state, action, confidence_param):
        return self.reward_estimate(state, action) + self.reward_epsilon(state, action, confidence_param)

def get_eva_next_u(probs, epsilon, u):
    sorted_states = sorted([(x, i) for i, x in enumerate(u)])
    sorted_states = [x[1] for x in sorted_actions]

    prob_estimate = copy.deepcopy(probs)
    prob_estimate[sorted_states[-1]] += (epsilon/2)
    total_prob = sum(prob_estimate)

    for state in sorted_states:
        if total_prob <= 1:
            break
        outside_prob = total_prob-prob_estimate[state]
        new_prob = max(0, 1-outside_prob)
        total_prob -= (prob_estimate[state]-new_prob)
        prob_estimate[state] = new_prob

    return prob_estimate

def get_eva_policy(parameter_estimator, model_bounds, confidence_param, n_steps):
    n_actions = model_bounds.n_actions
    n_states = model_bounds.n_states
    rewards = [[parameter_estimator.reward_ub(state, action, confidence_param) for action in range(n_actions)] for state in range(n_states)]
    prob_estimates = [[parameter_estimator.change_prob_estimate(state, action) for action in range(n_actions)] for state in range(n_states)]
    prob_epsilon = [[parameter_estimator.change_prob_epsilon(state, action) for action in range(n_actions)] for state in range(n_states)]
    values = [0 for x in range(model_bounds.n_states)]
    state_action_mapping = [0 for x in range(model_bounds.n_states)]

    while True:
        new_values = [float("-inf") for x in range(model_bounds.n_states)]

        for state in n_states:
            for action in n_actions:
                adjacent_probs = copy.deepcopy(prob_estimates[state][action])
                if state == 0:
                    adjacent_probs = adjacent_probs[1:]
                    adjacent_u = values[:2]
                elif state == n_states-1:
                    adjacent_probs = adjacent_probs[:-1]
                    adjacent_u = values[-2:]
                else:
                    adjacent_u = values[(state-1):(state+2)]
                next_u = get_eva_next_u(adjacent_probs, prob_epsilon[state][action], adjacent_u)
                u_candidate = rewards[state][action] + next_u
                if u_candidate > new_values[state][action]:
                    state_action_mapping[state] = action
                    new_values[state][action] = u_candidate

        # check for convergence and update values
        max_change = max([x-y for x,y in zip(new_values, values)])
        min_change = min([x-y for x,y in zip(new_values, values)])

        values = new_values

        if (max_change-min_change) < math.pow(n_steps,-0.5):
            break

    return state_action_mapping
