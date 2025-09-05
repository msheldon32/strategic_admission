import math
import copy
import random

import model

import numpy as np
from scipy.stats import chi2

class ParameterEstimator:
    def __init__(self, model_bounds):
        self.model_bounds = model_bounds

        self.sojourn_times = [[] for i in range(model_bounds.n_states)]
        self.transition_counts = [[0 for j in range(model_bounds.n_transitions)] for i in range(model_bounds.n_states)]

        self.positive_sojourn_times = [[] for i in range(model_bounds.n_states)]
        self.negative_sojourn_times = [[] for i in range(model_bounds.n_states)]

        self.positive_clock = 0
        self.negative_clock = 0

    def observe(self, state, transition_type, time_elapsed):
        self.transition_counts[state][self.model_bounds.get_transition_idx(transition_type)] += 1
        self.sojourn_times[state].append(time_elapsed)

        self.positive_clock += time_elapsed
        self.negative_clock += time_elapsed

        is_positive = (transition_type > 0) or (transition_type == 0 and state < self.model_bounds.capacities[1])

        if is_positive:
            self.positive_sojourn_times[state].append(self.positive_clock)
            self.positive_clock = 0
        else:
            self.negative_sojourn_times[state].append(self.negative_clock)
            self.negative_clock = 0

    def get_count(self, state, is_positive):
        if is_positive:
            return len(self.positive_sojourn_times[state])
        return len(self.negative_sojourn_times[state])

    def get_naive_rate_bounds(self, state, is_positive):
        lbound = self.model_bounds.rate_lb
        rbound = self.model_bounds.customer_ub if is_positive else self.model_bounds.server_ub

        if state < self.model_bounds.capacities[1] and is_positive:
            rbound += self.model_bounds.abandonment_ub
        elif state > self.model_bounds.capacities[1] and (not is_positive):
            rbound += self.model_bounds.abandonment_ub

        return [lbound, rbound]

    def sojourn_time_estimate(self, state, confidence_param, is_positive):
        acc = 0
        min_rate = self.model_bounds.rate_lb

        times = self.positive_sojourn_times[state] if is_positive else self.negative_sojourn_times[state]

        for i, stime in enumerate(times):
            truncation = math.sqrt(2*(i+1)/(math.pow(min_rate,2)*math.log(2*self.model_bounds.n_states/confidence_param)))
            if stime <= truncation:
                acc += stime
        return acc/len(self.sojourn_times[state])

    def sojourn_time_epsilon(self, state, confidence_param, is_positive):
        ct = self.get_count(state, is_positive)

        inner_term = (2/max(1, ct))*math.log(2*self.model_bounds.n_states/confidence_param)

        min_rate = self.model_bounds.rate_lb

        return (4/min_rate)*math.sqrt(inner_term)

    def transition_rate_bounds(self, state, confidence_param, is_positive):
        ct = self.get_count(state, is_positive)

        if ct == 0:
            return self.get_naive_rate_bounds(state, is_positive)

        st = self.sojourn_time_estimate(state, confidence_param, is_positive)
        ste = self.sojourn_time_epsilon(state, confidence_param, is_positive)

        min_rate, max_rate = self.get_naive_rate_bounds(state, is_positive)

        stime_lb = min(max(st-ste, 1/max_rate), 1/min_rate)
        stime_ub = min(max(st+ste, 1/max_rate), 1/min_rate)

        return [1/stime_ub, 1/stime_lb]

    def consider_abandonments(self, state, is_positive):
        zero_state = self.model_bounds.capacities[1]
        return (is_positive and state < zero_state) or ((not is_positive) and state > zero_state)

    def transition_prob_estimate(self, state, confidence_param, is_positive):
        # here, we just consider half of the transition space corresponding to the respective side
        if is_positive:
            transition_counts = self.transition_counts[state][self.model_bounds.n_classes[1]:]
            if state >= self.model_bounds.capacities[1]:
                transition_counts[0] = 0
        else:
            transition_counts = self.transition_counts[state][:self.model_bounds.n_classes[1]+1][::-1]
            if state <= self.model_bounds.capacities[1]:
                transition_counts[0] = 0
        total_n_transitions = sum(transition_counts)

        if total_n_transitions == 0:
            # naive case
            if self.consider_abandonments(state, is_positive):
                return [(1/len(transition_counts)) for x in transition_counts]
            if len(transition_counts) == 1:
                return [1]
            naive_prob = 1/(len(transition_counts)-1)
            out_vec = [naive_prob for x in transition_counts]
            out_vec[0] = 0
            return out_vec

        return [x/total_n_transitions for x in transition_counts]

    def transition_prob_epsilon(self, state, confidence_param, is_positive):
        ct = self.get_count(state, is_positive)
        if ct== 0:
            return 2
        transition_count = (self.model_bounds.n_classes[0]+1) if is_positive else (self.model_bounds.n_classes[1]+1)
        inner_term = (2*transition_count/ct)*math.log(2*self.model_bounds.n_states/confidence_param)
        return (math.sqrt(inner_term))

    def print_with_confidence(self, confidence_param):
        transition_probs_pos = [self.transition_prob_estimate(state, confidence_param, True) for state in range(self.model_bounds.n_states)]
        transition_epsilon_pos = [self.transition_prob_epsilon(state, confidence_param, True) for state in range(self.model_bounds.n_states)]
        transition_probs_neg = [self.transition_prob_estimate(state, confidence_param, False) for state in range(self.model_bounds.n_states)]
        transition_epsilon_neg = [self.transition_prob_epsilon(state, confidence_param, False) for state in range(self.model_bounds.n_states)]
        pos_bounds = [self.transition_rate_bounds(state, confidence_param, True) for state in range(self.model_bounds.n_states)]
        neg_bounds = [self.transition_rate_bounds(state, confidence_param, False) for state in range(self.model_bounds.n_states)]
        print("---------------------------------------")
        print("Transition probabilities")
        print("Positive:")
        print([f"{x} +- {e}" for x,e in zip(transition_probs_pos, transition_epsilon_pos)])
        print("Negative:")
        print([f"{x} +- {e}" for x,e in zip(transition_probs_neg, transition_epsilon_neg)])
        print("---------------------------------------")
        print("Transition rates")
        print("Positive:")
        print(pos_bounds)
        print("Negative:")
        print(neg_bounds)



class Exploration:
    def __init__(self, model_bounds: model.ModelBounds):
        self.model_bounds = model_bounds
        self.state_visit_counts = [0 for i in range(self.model_bounds.n_states)]
        self.state_visit_counts_in_episode = [0 for i in range(self.model_bounds.n_states)]
        self.steps_before_episode = 1
        self.n_episodes = 0

    def observe(self, state: int) -> bool:
        self.state_visit_counts[state] += 1
        self.state_visit_counts_in_episode[state] += 1

        return (2*self.state_visit_counts_in_episode[state]) >= self.state_visit_counts[state]

    def new_episode(self):
        self.state_visit_counts_in_episode = [0 for i in range(self.model_bounds.n_states)]

        self.steps_before_episode = sum(self.state_visit_counts)
        self.n_episodes += 1

def generate_extended_model(model_bounds, parameter_estimator, state_rewards, confidence_param):
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
    max_eta = 0
    if model_bounds.capacities[0] > 0:
        for state in range(model_bounds.capacities[0]-1,-1,-1):
            eta_prob = max(0, positive_transitions[state][0]-(positive_epsilons[state]/2))
            naive_eta = eta_prob*positive_rate_bounds[state][0]
            abandonments[state] = max(max_eta, naive_eta, model_bounds.rate_lb)
            max_eta = abandonments[state]

    max_gamma = 0
    if model_bounds.capacities[1] > 0:
        for state in range(model_bounds.capacities[0]+1, model_bounds.n_states):
            gamma_prob = max(0, negative_transitions[state][0]-(negative_epsilons[state]/2))
            naive_gamma = gamma_prob*negative_rate_bounds[state][0]
            abandonments[state] = max(max_gamma, naive_gamma, model_bounds.rate_lb)
            max_gamma = abandonments[state]

    # total positive and negative rates
    total_positive_rates = [0 for i in range(model_bounds.n_states)]
    total_negative_rates = [0 for i in range(model_bounds.n_states)]

    min_positive_rate = float("inf")
    for state in range(0, model_bounds.n_states):
        naive_rate = positive_rate_bounds[state][1]
        total_positive_rates[state] = min(naive_rate, min_positive_rate)
        min_positive_rate = total_positive_rates[state]

    min_negative_rate = float("inf")
    for state in range(model_bounds.n_states-1, -1, -1):
        naive_rate = negative_rate_bounds[state][1]
        total_negative_rates[state] = min(naive_rate, min_negative_rate)
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
