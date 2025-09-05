import math
import copy
import random

import model

import numpy as np
from scipy.stats import chi2

from ac import *

def generate_extended_model_ablation(model_bounds, parameter_estimator, state_rewards, confidence_param):
    failed = False

    extended_bounds = model_bounds.get_extended_bounds()
    extended_rewards = state_rewards.get_extended_rewards()

    # starting again from scratch!
    positive_rate_bounds = [parameter_estimator.transition_rate_bounds(state, confidence_param, True) for state in range(model_bounds.n_states)]
    negative_rate_bounds = [parameter_estimator.transition_rate_bounds(state, confidence_param, False) for state in range(model_bounds.n_states)]
    positive_transitions = [parameter_estimator.transition_prob_estimate(state, confidence_param, True) for state in range(model_bounds.n_states)]
    negative_transitions = [parameter_estimator.transition_prob_estimate(state, confidence_param, False) for state in range(model_bounds.n_states)]
    positive_epsilons = [parameter_estimator.transition_prob_epsilon(state, confidence_param, True) for state in range(model_bounds.n_states)]
    negative_epsilons = [parameter_estimator.transition_prob_epsilon(state, confidence_param, False) for state in range(model_bounds.n_states)]
    
    # eta and gamma
    abandonments = [0 for i in range(model_bounds.n_states)]
    if model_bounds.capacities[0] > 0:
        for state in range(model_bounds.capacities[0]-1,-1,-1):
            eta_prob = max(0, positive_transitions[state][0]-(positive_epsilons[state]/2))
            naive_eta = eta_prob*positive_rate_bounds[state][0]
            abandonments[state] = max(naive_eta, model_bounds.rate_lb)
            max_eta = abandonments[state]

    if model_bounds.capacities[1] > 0:
        for state in range(model_bounds.capacities[0]+1, model_bounds.n_states):
            gamma_prob = max(0, negative_transitions[state][0]-(negative_epsilons[state]/2))
            naive_gamma = gamma_prob*negative_rate_bounds[state][0]
            abandonments[state] = max(naive_gamma, model_bounds.rate_lb)
            max_gamma = abandonments[state]

    # total positive and negative rates
    total_positive_rates = [0 for i in range(model_bounds.n_states)]
    total_negative_rates = [0 for i in range(model_bounds.n_states)]

    for state in range(0, model_bounds.n_states):
        naive_rate = positive_rate_bounds[state][1]
        total_positive_rates[state] = naive_rate
        min_positive_rate = total_positive_rates[state]

    for state in range(model_bounds.n_states-1, -1, -1):
        naive_rate = negative_rate_bounds[state][1]
        total_negative_rates[state] = naive_rate
        min_negative_rate = total_negative_rates[state]


    # handling positive probabilities
    positive_transition_params = copy.deepcopy(positive_transitions)
    negative_transition_params = copy.deepcopy(negative_transitions)

    for state in range(model_bounds.n_states):
        excess = positive_epsilons[state]/2
        for customer_idx in extended_rewards.customer_order[state]:
            if excess <= 0:
                break

            if customer_idx == model_bounds.n_classes[0]:
                if state >= model_bounds.capacities[1]:
                    continue
                # this is the abandonment class
                min_val = abandonments[state]/total_positive_rates[state]
                new_prob = max(positive_transition_params[state][0]-excess, min_val)
                excess -= positive_transition_params[state][0]-new_prob
                positive_transition_params[state][0] = new_prob
                continue
            new_prob = max(positive_transition_params[state][customer_idx+1]-excess,0)
            excess -= positive_transition_params[state][customer_idx+1]-new_prob
            positive_transition_params[state][customer_idx+1] = new_prob
        best_type = extended_rewards.customer_order[state][-1]
        
        if best_type == model_bounds.n_classes[0]:
            if state >= model_bounds.capacities[1]:
                best_type = extended_rewards.customer_order[state][-2]
                positive_transition_params[state][best_type+1] += max((positive_epsilons[state]/2)-excess,0)
            else:
                positive_transition_params[state][0] += max((positive_epsilons[state]/2)-excess,0)
        else:
            positive_transition_params[state][best_type+1] += max((positive_epsilons[state]/2)-excess,0)
    

    for state in range(model_bounds.n_states):
        excess = negative_epsilons[state]/2
        for server_idx in extended_rewards.server_order[state]:
            if excess <= 0:
                break

            if server_idx == model_bounds.n_classes[1]:
                if state <= model_bounds.capacities[1]:
                    continue
                # this is the abandonment class
                min_val = abandonments[state]/total_negative_rates[state]
                new_prob = max(negative_transition_params[state][0]-excess, min_val)
                excess -= negative_transition_params[state][0]-new_prob
                negative_transition_params[state][0] = new_prob
                continue
            new_prob = max(negative_transition_params[state][server_idx+1]-excess,0)
            excess -= negative_transition_params[state][server_idx+1]-new_prob
            negative_transition_params[state][server_idx+1] = new_prob
        best_type = extended_rewards.server_order[state][-1]
        
        if best_type == model_bounds.n_classes[0]:
            if state <= model_bounds.capacities[1]:
                best_type = extended_rewards.server_order[state][-2]
                negative_transition_params[state][best_type+1] += max((negative_epsilons[state]/2)-excess,0)
            else:
                negative_transition_params[state][0] += max((negative_epsilons[state]/2)-excess,0)
        else:
            negative_transition_params[state][best_type+1] += max((negative_epsilons[state]/2)-excess,0)

    customer_rates = []
    server_rates = []

    for state in range(model_bounds.n_states):
        state_rates = []
        positive_rate = total_positive_rates[state]

        for customer_idx in range(model_bounds.n_classes[0]):
            state_rates.append(positive_transition_params[state][customer_idx+1]*positive_rate)

        if abs(sum(positive_transition_params[state])-1) > 0.01:
            failed = True
        if positive_rate < 0:
            failed = True
        if any([x < 0 for x in positive_transition_params[state]]):
            failed = True

        if state < model_bounds.capacities[1]:
            min_val = abandonments[state]
            excess_abandonments = positive_transition_params[state][0]*positive_rate
            state_rates.append(excess_abandonments-min_val)
        else:
            state_rates.append(0)

        customer_rates.append(state_rates)

    for state in range(model_bounds.n_states):
        state_rates = []
        negative_rate = total_negative_rates[state]

        for server_idx in range(model_bounds.n_classes[1]):
            state_rates.append(negative_transition_params[state][server_idx+1]*negative_rate)

        if abs(sum(negative_transition_params[state])-1) > 0.01:
            failed = True
        if negative_rate < 0:
            failed = True
        if any([x < 0 for x in negative_transition_params[state]]):
            failed = True

        if state > model_bounds.capacities[1]:
            min_val = abandonments[state]
            excess_abandonments = negative_transition_params[state][0]*positive_rate
            state_rates.append(excess_abandonments-min_val)
        else:
            state_rates.append(0)

        server_rates.append(state_rates)

    return model.Model(extended_bounds.capacities, customer_rates, server_rates, abandonments, extended_rewards), failed
