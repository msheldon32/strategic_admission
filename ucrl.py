# implementation of UCRL2

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

        self.n_states = model_bounds.n_states*model_bounds.n_transitions
        self.n_actions = 2

        self.change_counts = [[[0 for k in range(self.n_states)] for j in range(self.n_actions)] for i in range(self.n_states)]
        self.rewards = [[[] for j in range(self.n_actions)] for i in range(self.n_states)]

        self.max_reward = 2
        self.min_reward = -2

    def is_minimal_state(self, state):
        # we use the convention (state)*(n_transitions) + transition_no
        return state < self.model_bounds.n_transitions

    def is_maximal_state(self, state):
        return state >= (self.model_bounds.n_states-1)*self.model_bounds.n_transitions

    def observe(self, state, next_state, action, time_elapsed, reward):
        self.transition_counts[state][action][next_state] += 1
        clipped_reward = max(min(reward, self.max_reward), self.min_reward)
        self.rewards[state][action].append(clipped_reward)

    def change_prob_estimate(self, state, action):
        ct = sum(self.change_counts[state][action])

        if ct == 0:
            return [(1/self.n_states) for x in range(self.n_states)]

        return [x/ct for x in self.change_counts[state][action]]

    def change_prob_epsilon(self, state, action, confidence_param):
        ct = sum(self.change_counts[state][action])
        if ct== 0:
            return 2

        inner_term = ((14*self.n_states)/ct)*math.log(2*(self.n_actions)/confidence_param)
        return math.sqrt(inner_term)

    def reward_estimate(self, state, action):
        ct = len(self.rewards[state][action])
        if ct == 0:
            return 1
        total = sum(self.rewards[state][action])

        return total/ct

    def reward_epsilon(self, state, action, confidence_param):
        ct = len(self.rewards[state][action])

        inner_term = (7/(2*max(ct,1)))*math.log((2*self.n_actions*self.n_states)/confidence_param)
        return 2*math.sqrt(inner_term) # added in the 2 for reward scaling

    def reward_ub(self, state, action, confidence_param):
        return self.reward_estimate(state, action) + self.reward_epsilon(state, action, confidence_param)

class Exploration:
    def __init__(self, model_bounds):
        self.model_bounds = model_bounds
        self.n_states = model_bounds.n_states*model_bounds.n_transitions
        self.n_actions = 2
        self.sa_visit_counts = [[0 for j in range(self.n_actions)] for i in range(self.n_states)]
        self.sa_visit_counts_in_episode = [[0 for j in range(self.n_actions)] for i in range(self.n_states)]
        self.steps_before_episode = 1
        self.n_episodes = 0

    def observe(self, state, action):
        self.sa_visit_counts[state][action] += 1
        self.sa_visit_counts_in_episode[state][action] += 1

        return (2*self.sa_visit_counts_in_episode[state][action]) >= self.sa_visit_counts[state][action]
    
    def new_episode(self):
        self.sa_visit_counts_in_episode = [[0 for j in range(self.n_actions)] for i in range(self.n_states)]

        self.steps_before_episode = sum([sum(x) for x in self.sa_visit_counts])
        self.n_episodes += 1


def get_eva_next_u(probs, epsilon, u):
    sorted_states = sorted([(x, i) for i, x in enumerate(u)])
    sorted_states = [x[1] for x in sorted_states]

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

    return sum([x*y for x,y in zip(prob_estimate, u)])

def get_eva_policy(parameter_estimator, model_bounds, confidence_param, n_steps):
    n_actions = 2
    n_states = model_bounds.n_states*model_bounds.n_transitions
    rewards = [[parameter_estimator.reward_ub(state, action, confidence_param) for action in range(n_actions)] for state in range(n_states)]
    prob_estimates = [[parameter_estimator.change_prob_estimate(state, action) for action in range(n_actions)] for state in range(n_states)]
    prob_epsilon = [[parameter_estimator.change_prob_epsilon(state, action, confidence_param) for action in range(n_actions)] for state in range(n_states)]
    values = [0 for x in range(n_states)]
    state_action_mapping = [0 for x in range(n_states)]

    while True:
        new_values = [float("-inf") for x in range(n_states)]

        for state in range(n_states):
            for action in range(n_actions):
                adjacent_probs = copy.deepcopy(prob_estimates[state][action])
                next_u = get_eva_next_u(adjacent_probs, prob_epsilon[state][action], values)
                u_candidate = rewards[state][action] + next_u
                if u_candidate > new_values[state]:
                    state_action_mapping[state] = action
                    new_values[state] = u_candidate

        # check for convergence and update values
        max_change = max([x-y for x,y in zip(new_values, values)])
        min_change = min([x-y for x,y in zip(new_values, values)])

        values = new_values

        if (max_change-min_change) < math.pow(n_steps,-0.5):
            break

    return state_action_mapping
