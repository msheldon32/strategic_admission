import random
import copy
import math

import numpy as np

import policy

class Model:
    def __init__(self, capacities, customer_rates, server_rates, abandonment_rates, state_rewards):
        self.capacities = capacities
        self.customer_rates = customer_rates
        self.server_rates = server_rates
        self.abandonment_rates = abandonment_rates
        self.state_rewards = state_rewards

        self.customer_rates[-1] = [0 for x in self.customer_rates[0]]
        self.server_rates[0] = [0 for x in self.server_rates[0]]

        self.n_customer_types = len(self.customer_rates[0])
        self.n_server_types = len(self.server_rates[0])

        self.n_states = sum(self.capacities) + 1

        self.aggregate_customer_rates = [sum(self.customer_rates[state]) for state in range(self.n_states)]
        self.aggregate_server_rates = [sum(self.server_rates[state]) for state in range(self.n_states)]

        self.transition_labels = [i for i in range(-self.n_server_types, self.n_customer_types+1)]
        self.transition_rates = [self.server_rates[state][::-1] + [self.abandonment_rates[state]] + self.customer_rates[state] for state in range(0, self.n_states)]

    def __str__(self):
        return f"Aggregate customer rates: {self.aggregate_customer_rates}\naggregate server rates: {self.aggregate_server_rates}\nabandonment_rates: {self.abandonment_rates}\ncustomer rates: {self.customer_rates}\nserver rates: {self.server_rates}\nrewards: {self.state_rewards}"

    def get_customer_rate(self, state):
        return self.aggregate_customer_rates[state]
    
    def get_server_rate(self, state):
        return self.aggregate_server_rates[state]
    
    def get_abandonment_rate(self, state):
        return self.abandonment_rates[state]
    
    def get_total_rate(self, state):
        return self.get_customer_rate(state) + self.get_server_rate(state) + self.get_abandonment_rate(state)

    def get_job_population(self, state):
        return state - self.capacities[1]

    def get_transition_probs(self, state):
        rates = self.transition_rates[state]
        total_rate = sum(rates)

        if total_rate == 0:
            raise Exception("Invalid Model: there should be at least one valid transition")

        return [x/total_rate for x in self.transition_rates[state]]
    
    def get_reward(self, state, transition_type, accept, time_elapsed):
        if transition_type == 0 and not accept:
            raise Exception("Invalid Agent: cannot reject an abandonment")
        if not accept:
            return time_elapsed*self.state_rewards.holding_rewards[state]
        transition_index = transition_type + self.n_server_types
        return self.state_rewards.transition_rewards[state][transition_index] + time_elapsed*self.state_rewards.holding_rewards[state]
    
    def get_next_state(self, state, transition_type, accept):
        if transition_type == 0 and not accept:
            raise Exception("Invalid Agent: cannot reject an abandonment")
        if not accept:
            return state

        if transition_type > 0:
            return state + 1

        if transition_type < 0:
            return state - 1

        if state > self.capacities[1] and transition_type == 0:
            return state - 1

        if transition_type < self.capacities[1] and transition_type == 0:
            return state + 1
        
        return state

    def get_accepted_customer_types(self, state, limiting_type):
        if limiting_type == -1:
            return []

        threshold = self.state_rewards.customer_rewards[state][limiting_type]
        return [i for i in range(self.n_customer_types) if self.state_rewards.customer_rewards[state][i] >= threshold]

    def get_accepted_server_types(self, state, limiting_type):
        if limiting_type == -1:
            return []
        threshold = self.state_rewards.server_rewards[state][limiting_type]
        return [i for i in range(self.n_server_types) if self.state_rewards.server_rewards[state][i] >= threshold]

    def get_customer_acceptance_rate(self, state, limiting_type):
        types = self.get_accepted_customer_types(state, limiting_type)
        accept_rate = sum([rate for i, rate in enumerate(self.customer_rates[state]) if i in types])
        return accept_rate

    def get_server_acceptance_rate(self, state, limiting_type):
        types = self.get_accepted_server_types(state, limiting_type)
        accept_rate = sum([rate for i, rate in enumerate(self.server_rates[state]) if i in types])
        return accept_rate

    def get_customer_acceptance_reward(self, state, limiting_type):
        types = self.get_accepted_customer_types(state, limiting_type)
        accept_rates = [(rate if i in types else 0) for i, rate in enumerate(self.customer_rates[state])]
        accept_reward = sum([rate*reward for rate, reward in zip(accept_rates, self.state_rewards.customer_rewards[state])])

        return accept_reward

    def get_server_acceptance_reward(self, state, limiting_type):
        types = self.get_accepted_server_types(state, limiting_type)
        accept_rates = [(rate if i in types else 0) for i, rate in enumerate(self.server_rates[state])]
        accept_reward = sum([rate*reward for rate, reward in zip(accept_rates, self.state_rewards.server_rewards[state])])

        return accept_reward

    def get_sojourn_rate(self, state, limiting_types):
        customer_accept_rate = self.get_customer_acceptance_rate(state, limiting_types[0])
        server_accept_rate = self.get_server_acceptance_rate(state, limiting_types[1])

        return self.abandonment_rates[state] + customer_accept_rate + server_accept_rate

    def get_transition_rates(self, state, limiting_types):
        rate_down = self.get_server_acceptance_rate(state, limiting_types[1])
        rate_up   = self.get_customer_acceptance_rate(state, limiting_types[0])

        if state < self.capacities[1]:
            rate_up += self.abandonment_rates[state]
        if state > self.capacities[1]:
            rate_down += self.abandonment_rates[state]
        if rate_up < 0:
            print(f"invalid rate_up at {state}")
            print(self.customer_rates[state])
            print(f"abandonments: {self.abandonment_rates[state]}")
        if rate_down < 0:
            print(f"invalid rate_down at {state}")
            print(self.server_rates[state])
            print(f"abandonments: {self.abandonment_rates[state]}")
        return [rate_down, rate_up]

    def get_mean_reward(self, state, limiting_types):
        customer_reward = self.get_customer_acceptance_reward(state, limiting_types[0])
        server_reward = self.get_server_acceptance_reward(state, limiting_types[1])
        abandonment_reward = self.abandonment_rates[state] * self.state_rewards.abandonment_rewards[state]

        return customer_reward + server_reward + abandonment_reward + self.state_rewards.holding_rewards[state]

    def get_generator_matrix(self, policy):
        generator = np.zeros((self.n_states, self.n_states))

        for state in range(self.n_states):
            transitions = self.get_transition_rates(state, policy.get_limiting_types(state))
            if transitions[0] < 0:
                raise Exception(f"invalid generator: {transitions[0]}")
            total_rate = 0
            if state > 0:
                generator[state, state-1] = transitions[0]
                total_rate += transitions[0]
            if state < self.n_states-1:
                generator[state, state+1] = transitions[1]
                total_rate += transitions[1]

            generator[state, state] = -total_rate
        return generator

    def get_reward_vector(self, policy):
        vector = np.zeros(self.n_states)

        for state in range(self.n_states):
            vector[state] = self.get_mean_reward(state, policy.get_limiting_types(state))

        return vector

    def get_steady_state_probs(self, policy):
        generator = self.get_generator_matrix(policy)

        A = generator.T
        A[-1, :] = np.ones(self.n_states)
        b = np.zeros(self.n_states)
        b[-1] = 1

        return np.linalg.solve(A,b)

    def get_gain_bias(self, policy):
        generator = self.get_generator_matrix(policy)
        reward_vector = self.get_reward_vector(policy)

        """print(generator)
        print("-----------------")
        print(self.customer_rates)
        print(self.customer_rewards)
        print("-----------------")
        print(self.server_rates)
        print(self.server_rewards)
        print("-----------------")
        print(self.abandonment_rates)
        print(policy.limiting_types)"""

        #raise Exception("stop")

        # create a submatrix, add rows of -1 to the rhs to represent the gain, and another row at the end to set one bias value to 0
        gain_coef_vector = -np.ones((self.n_states,1))
        #gain_coef_vector = np.ones((self.n_states,1))
        lhs_rb_vector = np.zeros((1, self.n_states))
        lhs_rb_vector[0,0] = 1
        lhs_matrix = np.block([[generator, gain_coef_vector],[lhs_rb_vector, np.zeros((1,1))]])
        rhs_vector = -np.concatenate([reward_vector, np.zeros(1)])

        # solve Zh = g-r, or Ax = -r
        gain_bias_vector = np.linalg.solve(lhs_matrix, rhs_vector)

        #raise Exception("this doesn't look right, bottom value should be 0")

        # extract bias vector and gain
        return gain_bias_vector[:-1], gain_bias_vector[-1]
    
    def get_maximal_action(self, state, bias, gain):
        max_bias = float("-inf")
        max_limit_types = [-1,-1]

        for limit_customer_type in range(-1, self.n_customer_types):
            for limit_server_type in range(-1, self.n_server_types):
                new_bias = 0
                unnorm_bias = 0
                limit_types = [limit_customer_type, limit_server_type]
                reward = self.get_mean_reward(state, limit_types)
                transition_rates = self.get_transition_rates(state, limit_types)

                if state > 0:
                    unnorm_bias += transition_rates[0] * bias[state-1]
                if state < self.n_states-1:
                    unnorm_bias += transition_rates[1] * bias[state+1]
                
                # the problem is below. Basically we will 0 out the bias.
                unnorm_bias += (reward-gain)
                if sum(transition_rates) == 0:
                    if reward > gain:
                        new_bias = float("inf")
                    else:
                        new_bias = float("-inf")
                else:
                    new_bias = unnorm_bias / sum(transition_rates)
                
                if new_bias > max_bias:
                    max_limit_types = limit_types
                    max_bias = new_bias
        #print(f"max_bias: {max_bias} at {state}")
        #print(f"holding rewards: {self.state_rewards.holding_rewards}")
        return max_limit_types


class ModelBounds:
    def __init__(self, n_classes, capacities):
        self.rate_lb = 1
        self.customer_ub = 5
        self.server_ub = 5
        self.abandonment_ub = 2

        self.n_classes = n_classes
        self.capacities = capacities
        self.transition_labels = [i for i in range(-self.n_classes[1], self.n_classes[0]+1)]

    def get_extended_bounds(self):
        new_bounds = ModelBounds([x+1 for x in self.n_classes], self.capacities)
        new_bounds.rate_lb = self.rate_lb
        new_bounds.customer_ub = self.customer_ub
        new_bounds.server_ub = self.server_ub
        new_bounds.abandonment_ub = self.abandonment_ub

        return new_bounds

    @property
    def n_states(self):
        return sum(self.capacities)+1

    @property
    def n_transitions(self):
        return sum(self.n_classes)+1

    def total_rate_ub(self,state):
        if state == 0:
            return self.abandonment_ub + self.customer_ub
        elif state == self.n_states-1:
            return self.abandonment_ub + self.server_ub
        return self.abandonment_ub + self.customer_ub + self.server_ub

    def total_rate_lb(self,state):
        if state == 0:
            return self.rate_lb*2
        elif state == self.n_states-1:
            return self.rate_lb*2
        return self.rate_lb*3
    
    def positive_rate_ub(self):
        return self.abandonment_ub + self.customer_ub
    
    def negative_rate_ub(self):
        return self.abandonment_ub + self.server_ub

    def get_transition_idx(self, transition_type):
        return self.transition_labels.index(transition_type)

    def get_maximum_rate(self):
        return self.customer_ub+self.server_ub+self.abandonment_ub

    def get_minimum_rate(self):
        return self.rate_lb

class RewardGenerator:
    def __init__(self, rng):
        self.rng = rng
    
    def generate_customer_rewards(self,bounds):
        return [[self.rng.uniform(-1,1) for stype in range(bounds.n_classes[0])] for i in range(bounds.n_states)]
        #return [[0 for stype in range(bounds.n_classes[1])] for i in range(bounds.n_states)]

    def generate_server_rewards(self,bounds):
        return [[self.rng.uniform(-1,1) for stype in range(bounds.n_classes[1])] for i in range(bounds.n_states)]
        #return [[0 for stype in range(bounds.n_classes[1])] for i in range(bounds.n_states)]

    def generate_abandonment_rewards(self,bounds):
        return [self.rng.uniform(-0.5,0) for i in range(bounds.n_states)]
        #return [0 for i in range(bounds.n_states)]

    def generate_holding_rewards(self,bounds):
        return [abs(state-bounds.capacities[1])/100 for state in range(bounds.n_states)]
        #return [0 for i in range(bounds.n_states)]

class StateRewards:
    def __init__(self, customer_rewards, server_rewards, abandonment_rewards, holding_rewards):
        self.customer_rewards = customer_rewards
        self.server_rewards = server_rewards
        self.abandonment_rewards = abandonment_rewards
        self.holding_rewards = holding_rewards

        self.n_customer_types = len(self.customer_rewards[0])
        self.n_server_types = len(self.server_rewards[0])

        self.n_states = len(self.customer_rewards)

        self.transition_rewards = [self.server_rewards[state][::-1] + [self.abandonment_rewards[state]] + self.customer_rewards[state] for state in range(0, self.n_states)]

        # customer/server orderings
        self.customer_order = [sorted([(x,i) for i,x in enumerate(y)]) for y in self.customer_rewards]
        self.customer_order = [[x[1] for x in y] for y in self.customer_order]
        self.server_order = [sorted([(x,i) for i,x in enumerate(y)]) for y in self.server_rewards]
        self.server_order = [[x[1] for x in y] for y in self.server_order]

    def get_extended_rewards(self):
        new_customer_rewards = [a + [b] for a,b in zip(self.customer_rewards, self.abandonment_rewards)]
        new_server_rewards = [a + [b] for a,b in zip(self.server_rewards, self.abandonment_rewards)]
        extended_rewards = StateRewards(new_customer_rewards, new_server_rewards, copy.deepcopy(self.abandonment_rewards), copy.deepcopy(self.holding_rewards))
        return extended_rewards

    def __str__(self):
        return "\n" +f"customer_rewards: {self.customer_rewards}"+f"server_rewards: {self.server_rewards}"+f"abandonment_rewards: {self.abandonment_rewards}"+f"holding_rewards: {self.holding_rewards}"


def generate_model(bounds: ModelBounds, reward_generator: RewardGenerator, rng: np.random._generator.Generator):
    n_states = bounds.n_states
    def generate_arrival_rates(ct, total_rate):
        probs = list(rng.dirichlet(np.ones(ct)))
        rates = [total_rate * prob for prob in probs]
        return rates
    
    customer_lb = (bounds.rate_lb+bounds.customer_ub)/2
    server_lb   = (bounds.rate_lb+bounds.server_ub)/2
    total_customer_rates = sorted([rng.uniform(customer_lb, bounds.customer_ub) for i in range(n_states)], reverse=True)
    total_server_rates = sorted([rng.uniform(server_lb,bounds.server_ub) for i in range(n_states)])

    customer_rates = [generate_arrival_rates(bounds.n_classes[0], rate) for rate in total_customer_rates]
    server_rates = [generate_arrival_rates(bounds.n_classes[1], rate) for rate in total_server_rates]

    # create abandonments
    customer_abandonments = sorted([rng.uniform(bounds.rate_lb, bounds.abandonment_ub) for i in range(bounds.capacities[0])])
    server_abandonments = sorted([rng.uniform(bounds.rate_lb, bounds.abandonment_ub) for i in range(bounds.capacities[1])], reverse=True)
    abandonment_rates = server_abandonments + [0] + customer_abandonments

    customer_rewards = reward_generator.generate_customer_rewards(bounds)
    server_rewards = reward_generator.generate_server_rewards(bounds)
    abandonment_rewards = reward_generator.generate_abandonment_rewards(bounds)
    holding_rewards = reward_generator.generate_holding_rewards(bounds)

    state_rewards = StateRewards(customer_rewards, server_rewards, abandonment_rewards, holding_rewards)
    return Model(bounds.capacities, customer_rates, server_rates, abandonment_rates, state_rewards)

if __name__ == "__main__":
    rng = np.random.default_rng(seed=2)
    model = generate_model(ModelBounds(), RewardGenerator(rng), rng)
    
    policy = policy.Policy.full_acceptance_policy(model)
    
    gains = []

    while True:
        bias, gain = model.get_gain_bias(policy)
        gains.append(gain)
        new_policy = policy.get_improved_policy()
        if policy == new_policy:
            break
        policy = new_policy
    print(gains)
