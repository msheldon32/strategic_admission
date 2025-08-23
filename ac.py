import math
import copy

import model

import numpy as np
from scipy.stats import chi2

class ParameterEstimator:
    def __init__(self, model_bounds):
        self.model_bounds = model_bounds

        self.sojourn_times = [[] for i in range(model_bounds.n_states)]
        self.transition_counts = [[0 for j in range(model_bounds.n_transitions)] for i in range(model_bounds.n_states)]

    def observe(self, state, transition_type, time_elapsed):
        self.transition_counts[state][self.model_bounds.get_transition_idx(transition_type)] += 1
        self.sojourn_times[state].append(time_elapsed)

    def sojourn_time_estimate(self, state, confidence_param):
        if len(self.sojourn_times[state]) == 0:
            return ((1/self.model_bounds.total_rate_lb(state))+(1/self.model_bounds.total_rate_ub(state)))/2
        acc = 0
        min_rate = self.model_bounds.total_rate_lb(state)
        for i, stime in enumerate(self.sojourn_times[state]):
            truncation = math.sqrt(2*(i+1)/(math.pow(min_rate,2)*math.log(1/confidence_param)))
            if stime <= truncation:
                acc += stime
        return acc/len(self.sojourn_times[state])

    def sojourn_time_epsilon(self, state, confidence_param):
        ct = len(self.sojourn_times[state])

        if ct == 0:
            return ((1/self.model_bounds.total_rate_lb(state))-(1/self.model_bounds.total_rate_ub(state)))/2

        inner_term = (14/max(1, ct))*math.log(2*self.model_bounds.n_states/confidence_param)

        min_rate = self.model_bounds.total_rate_lb(state)
        #max_rate = self.model_bounds.get_maximum_rate()

        return (4/min_rate)*math.sqrt(inner_term)

    def transition_rate_bounds(self, state, confidence_param):
        st = self.sojourn_time_estimate(state, confidence_param)
        ste = self.sojourn_time_epsilon(state, confidence_param)
        min_rate = self.model_bounds.total_rate_lb(state)
        max_rate = self.model_bounds.total_rate_ub(state)

        stime_lb = max(st-ste, 1/max_rate)
        stime_ub = min(st+ste, 1/min_rate)

        return [1/stime_ub, 1/stime_lb]

    def transition_rate_estimate_loose(self, state, confidence_param):
        if len(self.sojourn_times[state]) == 0:
            return (self.model_bounds.total_rate_lb(state)+self.model_bounds.total_rate_ub(state))/2
        st = self.sojourn_time_estimate(state, confidence_param)

        if st == 0:
            return (self.model_bounds.total_rate_lb(state)+self.model_bounds.total_rate_ub(state))/2

        return 1/st
        
    def transition_rate_epsilon_loose(self, state, confidence_param):
        ct = len(self.sojourn_times[state])

        if ct == 0:
            return (self.model_bounds.total_rate_ub(state)-self.model_bounds.total_rate_lb(state))/2

        inner_term = (14/max(1, ct))*math.log(2*self.model_bounds.n_states/confidence_param)

        min_rate = self.model_bounds.get_minimum_rate()
        max_rate = self.model_bounds.get_maximum_rate()

        return (4*((max_rate**2)/min_rate)*math.sqrt(inner_term))

    def transition_prob_estimate(self, state, confidence_param):
        transition_counts = copy.deepcopy(self.transition_counts[state])
        total_n_transitions = sum(transition_counts)

        if total_n_transitions == 0:
            return [(1/len(transition_counts)) for x in transition_counts]

        return [x/total_n_transitions for x in transition_counts]

    def transition_prob_epsilon(self, state, confidence_param):
        total_n_transitions = sum(self.transition_counts[state])
        if total_n_transitions == 0:
            return 1
        inner_term = (14*self.model_bounds.n_transitions/max(1,total_n_transitions))*math.log(2*self.model_bounds.n_states/confidence_param)
        return (math.sqrt(inner_term))
        #return 0

    def print_with_confidence(self, confidence_param):
        transition_rates = [self.transition_rate_estimate(state, confidence_param) for state in range(self.model_bounds.n_states)]
        transition_epsilon = [self.transition_rate_epsilon(state, confidence_param) for state in range(self.model_bounds.n_states)]

        print([f"{x} +- {e}" for x,e in zip(transition_rates, transition_epsilon)])


class ClassicalParameterEstimator:
    def __init__(self, model_bounds):
        # use chi2 bounds instead of truncated mean, this is for practical applications with worse regret guarantees.
        self.model_bounds = model_bounds

        self.sojourn_times = [[] for i in range(model_bounds.n_states)]
        self.transition_counts = [[0 for j in range(model_bounds.n_transitions)] for i in range(model_bounds.n_states)]

    def observe(self, state, transition_type, time_elapsed):
        self.transition_counts[state][self.model_bounds.get_transition_idx(transition_type)] += 1
        self.sojourn_times[state].append(time_elapsed)

    def sojourn_time_estimate(self, state, confidence_param):
        return sum(self.sojourn_times[state])/len(self.sojourn_times[state])

    def transition_rate_estimate(self, state, confidence_param):
        ci = self.transition_rate_ci(state, confidence_param)

        return (ci[1]+ci[0])/2

    def transition_rate_epsilon(self, state, confidence_param):
        ci = self.transition_rate_ci(state, confidence_param)

        return (ci[1]-ci[0])/2
        
    def transition_rate_ci(self, state, confidence_param):
        n_transitions = len(self.sojourn_times[state])
        total_time = sum(self.sojourn_times[state])

        if n_transitions == 0:
            return (self.model_bounds.total_rate_lb(state), self.model_bounds.total_rate_ub(state))
        #alpha = 1-confidence_param
        lower = chi2.ppf(confidence_param/2, 2*n_transitions) / (2*total_time)
        upper = chi2.ppf(1 - confidence_param/2, 2*n_transitions) / (2*total_time)
        print(f"lower: {lower}, upper: {upper}")
        return lower, upper

    def transition_prob_estimate(self, state, confidence_param):
        transition_counts = copy.deepcopy(self.transition_counts[state])
        total_n_transitions = sum(transition_counts)

        if total_n_transitions == 0:
            return [(1/len(transition_counts)) for x in transition_counts]

        return [x/total_n_transitions for x in transition_counts]

    def transition_prob_epsilon(self, state, confidence_param):
        total_n_transitions = sum(self.transition_counts[state])
        if total_n_transitions == 0:
            return 1
        inner_term = (14*self.model_bounds.n_transitions/max(1,total_n_transitions))*math.log(2*self.model_bounds.n_states/confidence_param)
        print(f"state: {state}, prob half length: {math.sqrt(inner_term)}")
        return math.sqrt(inner_term)

    def print_with_confidence(self, confidence_param):
        transition_rates = [self.transition_rate_estimate(state, confidence_param) for state in range(self.model_bounds.n_states)]
        transition_epsilon = [self.transition_rate_epsilon(state, confidence_param) for state in range(self.model_bounds.n_states)]

        print([f"{x} +- {e}" for x,e in zip(transition_rates, transition_epsilon)])

class MockParameterEstimator:
    def __init__(self, rates, epsilons, transition_probs, transition_epsilons):
        self.rates = rates
        self.epsilons = epsilons
        self.transition_probs = transition_probs
        self.transition_epsilons = transition_epsilons

    def sojourn_time_estimate(self, state, confidence_param):
        return 1/self.rates[state]

    def transition_rate_estimate(self, state, confidence_param):
        return self.rates[state]
        
    def transition_rate_epsilon(self, state, confidence_param):
        return self.transition_epsilons[state]

    def transition_prob_estimate(self, state, confidence_param):
        return self.transition_probs[state]

    def transition_prob_epsilon(self, state, confidence_param):
        return self.transition_epsilons[state]


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
    transition_rate_bounds = [parameter_estimator.transition_rate_bounds(state, confidence_param) for state in range(model_bounds.n_states)]
    transition_probs = [parameter_estimator.transition_prob_estimate(state, confidence_param) for state in range(model_bounds.n_states)]
    transition_prob_epsilon = [parameter_estimator.transition_prob_epsilon(state, confidence_param) for state in range(model_bounds.n_states)]

    abandonments = [0 for state in range(model_bounds.n_states)]
    total_customer_arrivals = [0 for state in range(model_bounds.n_states)]
    total_server_arrivals = [0 for state in range(model_bounds.n_states)]
    customer_rates = [[0]*(model_bounds.n_classes[0]+1) for state in range(model_bounds.n_states)]
    server_rates = [[0]*(model_bounds.n_classes[1]+1) for state in range(model_bounds.n_states)]

    ext_model_bounds = model_bounds.get_extended_bounds()
    ext_state_rewards = state_rewards.get_extended_rewards()

    # generate eta
    max_eta = 0
    for state in range(model_bounds.capacities[1]+1, model_bounds.n_states):
        transition_idx = model_bounds.get_transition_idx(0)
        naive_eta = max(transition_probs[state][transition_idx]-(0.5*transition_prob_epsilon[state]),0) * (transition_rate_bounds[state][0])
        eta = max(naive_eta, max_eta, model_bounds.rate_lb)
        abandonments[state] = eta
        max_eta = eta

    # generate gamma
    max_gamma = 0
    for state in range(model_bounds.capacities[1]-1,-1,-1):
        transition_idx = model_bounds.get_transition_idx(0)
        naive_gamma = max(transition_probs[state][transition_idx]-(0.5*transition_prob_epsilon[state]),0) * (transition_rate_bounds[state][0])
        gamma = max(naive_gamma, max_gamma, model_bounds.rate_lb)
        abandonments[state] = gamma
        max_gamma = gamma

    # DRY
    get_state_iterator = lambda side: range(0, model_bounds.n_states-1) if side == "customer" else range(model_bounds.n_states-1,0,-1)
    consider_abandonments = lambda state, side: (state < model_bounds.capacities[1] and side == "customer") or (state > model_bounds.capacities[1] and side == "server")
    get_start_transition_customer = lambda state: model_bounds.get_transition_idx(0) if consider_abandonments(state, "customer") else model_bounds.get_transition_idx(1)
    get_end_transition_customer = lambda state: model_bounds.get_transition_idx(model_bounds.n_classes[0])
    get_start_transition_server = lambda state: 0
    get_end_transition_server = lambda state: model_bounds.get_transition_idx(0) if consider_abandonments(state, "server") else model_bounds.n_classes[1]-1
    get_start_transition  = lambda state, side: get_start_transition_customer(state) if side == "customer" else get_start_transition_server(state)
    get_end_transition  = lambda state, side: get_end_transition_customer(state) if side == "customer" else get_end_transition_server(state)

    order_transitions = lambda x, side: x if side == "customer" else x[::-1]

    min_total_increase_rate = float("inf")
    for state in get_state_iterator("customer"):
        start_transition = get_start_transition(state, "customer")
        end_transition = get_end_transition(state, "customer")

        cum_prob = sum(transition_probs[state][start_transition:end_transition+1])

        ci_ub = min(cum_prob+(0.5*transition_prob_epsilon[state]),1)*(transition_rate_bounds[state][1])
        total_rate_ub = model_bounds.customer_ub+model_bounds.abandonment_ub

        total_increase_rate = min(ci_ub, total_rate_ub, min_total_increase_rate)
        min_total_increase_rate = total_increase_rate

        total_customer_arrivals[state] = total_increase_rate
        if consider_abandonments(state, "customer"):
            total_customer_arrivals[state] -= abandonments[state]
    min_total_decrease_rate = float("inf")
    for state in get_state_iterator("server"):
        start_transition = get_start_transition(state, "server")
        end_transition = get_end_transition(state, "server")

        cum_prob = sum(transition_probs[state][start_transition:end_transition+1])

        ci_ub = min(cum_prob+(0.5*transition_prob_epsilon[state]),1)*(transition_rate_bounds[state][1])
        total_rate_ub = model_bounds.server_ub+model_bounds.abandonment_ub

        total_decrease_rate = min(ci_ub, total_rate_ub, min_total_decrease_rate)
        min_total_decrease_rate = total_decrease_rate

        total_server_arrivals[state] = total_decrease_rate
        if consider_abandonments(state, "server"):
            total_server_arrivals[state] -= abandonments[state]
    
    for state in get_state_iterator("customer"):
        max_total_rate = model_bounds.customer_ub+model_bounds.server_ub+model_bounds.abandonment_ub
        min_rate = model_bounds.rate_lb
        inflation_factor = 2*(max_total_rate/min_rate)
        # customers
        arrival_excess = (0.5*inflation_factor)*transition_prob_epsilon[state]

        ordered_customers = ext_state_rewards.customer_order[state]

        start_transition = get_start_transition(state, "customer")
        end_transition = get_end_transition(state, "customer")
        cum_prob = sum(transition_probs[state][start_transition:end_transition+1])
        if cum_prob == 0:
            n_transitions = (end_transition+1)-start_transition
            conditional_probs = [1/n_transitions for x in range(n_transitions)]
        else:
            conditional_probs = order_transitions([x / cum_prob for x in transition_probs[state][start_transition:end_transition+1]], "customer")

        if consider_abandonments(state, "customer"):
            # reorder everything to be at the end.
            conditional_probs = conditional_probs[1:] + [conditional_probs[0]]
        else:
            conditional_probs.append(0)

        ext_cond_probs = copy.copy(conditional_probs)
        maximal_class = ext_state_rewards.customer_order[state][-1]
        if (not consider_abandonments(state, "customer")) and maximal_class == len(conditional_probs)-1:
            maximal_class = ext_state_rewards.customer_order[state][-2]
        ext_cond_probs[maximal_class] += arrival_excess
        burned_prob = 0
        for customer_type in ext_state_rewards.customer_order[state]:
            min_prob = 0

            if customer_type == len(conditional_probs)-1:
                if consider_abandonments(state, "customer"):
                    min_prob = (abandonments[state]/(total_customer_arrivals[state]+abandonments[state]))
                else:
                    continue
            prob_burn = min(arrival_excess - burned_prob, ext_cond_probs[customer_type] - min_prob)
            ext_cond_probs[customer_type] -= prob_burn
            burned_prob += prob_burn

            if customer_type == len(conditional_probs)-1:
                ext_cond_probs[customer_type] -= (abandonments[state]/(total_customer_arrivals[state]+abandonments[state]))
        total_rate = total_customer_arrivals[state] + (abandonments[state] if consider_abandonments(state, "customer") else 0)
        customer_rates[state] = [total_rate * p for p in ext_cond_probs]

    
    for state in get_state_iterator("server"):
        max_total_rate = model_bounds.customer_ub+model_bounds.server_ub+model_bounds.abandonment_ub
        min_rate = model_bounds.rate_lb
        inflation_factor = 2*(max_total_rate/min_rate)
        # servers
        arrival_excess = (0.5*inflation_factor)*transition_prob_epsilon[state]
        print(f"state: {state}")
        print(f"Transition prob epsilon: {transition_prob_epsilon[state]}")
        print(f"arrival_excess: {arrival_excess}")
        #raise Exception("stop")

        ordered_servers = ext_state_rewards.server_order[state]

        start_transition = get_start_transition(state, "server")
        end_transition = get_end_transition(state, "server")

        cum_prob = sum(transition_probs[state][start_transition:end_transition+1])
        if cum_prob == 0:
            n_transitions = (end_transition+1)-start_transition
            conditional_probs = [1/n_transitions for x in range(n_transitions)]
        else:
            conditional_probs = order_transitions([x / cum_prob for x in transition_probs[state][start_transition:end_transition+1]], "server")

        if consider_abandonments(state, "server"):
            # reorder everything to be at the end.
            conditional_probs = conditional_probs[:-1] + [conditional_probs[-1]]
        else:
            conditional_probs.append(0)

        ext_cond_probs = copy.copy(conditional_probs)
        maximal_class = ext_state_rewards.server_order[state][-1]
        if (not consider_abandonments(state, "server")) and maximal_class == len(conditional_probs)-1:
            maximal_class = ext_state_rewards.server_order[state][-2]
        ext_cond_probs[maximal_class] += arrival_excess
        burned_prob = 0
        for server_type in ext_state_rewards.server_order[state]:
            min_prob = 0

            if server_type == len(conditional_probs)-1:
                if consider_abandonments(state, "server"):
                    min_prob = (abandonments[state]/(total_server_arrivals[state]+abandonments[state]))
                else:
                    continue
            prob_burn = min(arrival_excess - burned_prob, ext_cond_probs[server_type] - min_prob)
            ext_cond_probs[server_type] -= prob_burn
            burned_prob += prob_burn

            if server_type == len(conditional_probs)-1:
                ext_cond_probs[server_type] -= (abandonments[state]/(total_server_arrivals[state]+abandonments[state]))
        total_rate = total_server_arrivals[state] + (abandonments[state] if consider_abandonments(state, "server") else 0)
        server_rates[state] = [total_rate * p for p in ext_cond_probs]

    return model.Model(ext_model_bounds.capacities, customer_rates, server_rates, abandonments, ext_state_rewards)
