import math
import copy
import random

import model

import numpy as np
from scipy.stats import chi2

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

    if model_bounds.capacities[1] > 0:
        for state in range(model_bounds.capacities[0]+1, model_bounds.n_states):
            gamma_prob = max(0, negative_transitions[state][0]-(negative_epsilons[state]/2))
            naive_gamma = gamma_prob*negative_rate_bounds[state][0]
            abandonments[state] = max(naive_gamma, model_bounds.rate_lb)

    # total positive and negative rates
    total_positive_rates = [0 for i in range(model_bounds.n_states)]
    total_negative_rates = [0 for i in range(model_bounds.n_states)]

    for state in range(0, model_bounds.n_states):
        naive_rate = positive_rate_bounds[state][1]
        total_positive_rates[state] = naive_rate

    for state in range(model_bounds.n_states-1, -1, -1):
        naive_rate = negative_rate_bounds[state][1]
        total_negative_rates[state] = naive_rate


    # handling positive probabilities
    positive_transition_params = copy.deepcopy(positive_transitions)
    negative_transition_params = copy.deepcopy(negative_transitions)

    for state in range(model_bounds.n_states):
        excess = positive_epsilons[state]/2
        supp = positive_epsilons[state]/2

        # forward pass: remove excess
        for customer_idx in extended_rewards.customer_order[state]:
            if excess <= 0:
                break

            min_val = 0

            transition_idx = customer_idx + 1

            if customer_idx == model_bounds.n_classes[0]:
                if state >= model_bounds.capacities[1]:
                    continue

                min_val = abandonments[state]/total_positive_rates[state]
            
                transition_idx = 0

            new_prob = max(positive_transition_params[state][transition_idx]-excess, min_val)
            excess -= positive_transition_params[state][transition_idx]-new_prob
            positive_transition_params[state][transition_idx] = new_prob

        supp -= excess
        # backward pass: accumulate
        for customer_idx in extended_rewards.customer_order[state][::-1]:
            if supp <= 0:
                break

            max_val = 1

            transition_idx = customer_idx + 1

            if customer_idx == model_bounds.n_classes[0]:
                if state >= model_bounds.capacities[1]:
                    continue
            
                transition_idx = 0

            new_prob = min(positive_transition_params[state][transition_idx]+supp, max_val)
            supp -= new_prob-positive_transition_params[state][transition_idx]
            positive_transition_params[state][transition_idx] = new_prob

    for state in range(model_bounds.n_states):
        excess = negative_epsilons[state]/2
        supp = negative_epsilons[state]/2

        # forward pass: remove excess
        for server_idx in extended_rewards.server_order[state]:
            if excess <= 0:
                break

            min_val = 0

            transition_idx = server_idx + 1

            if server_idx == model_bounds.n_classes[1]:
                if state <= model_bounds.capacities[1]:
                    continue

                min_val = abandonments[state]/total_negative_rates[state]
            
                transition_idx = 0

            new_prob = max(negative_transition_params[state][transition_idx]-excess, min_val)
            excess -= negative_transition_params[state][transition_idx]-new_prob
            negative_transition_params[state][transition_idx] = new_prob

        supp -= excess
        # backward pass: accumulate
        for server_idx in extended_rewards.server_order[state][::-1]:
            if supp <= 0:
                break

            max_val = 1

            transition_idx = server_idx + 1

            if server_idx == model_bounds.n_classes[1]:
                if state <= model_bounds.capacities[1]:
                    continue
            
                transition_idx = 0

            new_prob = min(negative_transition_params[state][transition_idx]+supp, max_val)
            supp -= new_prob-negative_transition_params[state][transition_idx]
            negative_transition_params[state][transition_idx] = new_prob

    customer_rates = []
    server_rates = []

    for state in range(model_bounds.n_states):
        state_rates = []
        positive_rate = total_positive_rates[state]

        for customer_idx in range(model_bounds.n_classes[0]):
            state_rates.append(positive_transition_params[state][customer_idx+1]*positive_rate)

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

        if state > model_bounds.capacities[1]:
            min_val = abandonments[state]
            excess_abandonments = negative_transition_params[state][0]*negative_rate
            state_rates.append(excess_abandonments-min_val)
        else:
            state_rates.append(0)

        server_rates.append(state_rates)

        #if any([x < -0.01 for x in state_rates]):
        #    print(f"state: {state}")
        #    print(f"negative rate: {negative_rate}")
        #    print(f"transition probs: {negative_transition_params[state]}")
        #    print(f"abandonment rate: {abandonments[state]}")
        #    print(f"excess abandonments: {excess_abandonments}")
        #    print(f"state rates: {state_rates}")
        #    raise Exception("stop")


    out_model = model.Model(extended_bounds.capacities, customer_rates, server_rates, abandonments, extended_rewards)

    return out_model, out_model.is_valid()
