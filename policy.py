import utils

class Policy:
    def __init__(self, model):
        self.model = model
        self.n_states = model.n_states
        self.n_server_types = model.n_server_types
        self.n_customer_types = model.n_customer_types

        self.limiting_types = []

    def get_limiting_types(self, state):
        return self.limiting_types[state]

    def get_improved_policy(self):
        bias, gain = self.model.get_gain_bias(self)

        new_policy = Policy(self.model)
        new_policy.limiting_types = []

        for state in range(self.n_states):
            new_policy.limiting_types.append(self.model.get_maximal_action(state, bias,gain))

        return new_policy

    def act(self, state, transition_type):
        if transition_type == 0:
            return True
        if transition_type < 0:
            server_type = (-transition_type)-1
            limiting_type = self.limiting_types[state][1]
            if limiting_type == -1:
                return False
            return self.model.server_rewards[state][server_type] >= self.model.server_rewards[state][limiting_type]
        if transition_type > 0:
            customer_type = transition_type-1
            limiting_type = self.limiting_types[state][0]
            if limiting_type == -1:
                return False
            return self.model.customer_rewards[state][customer_type] >= self.model.customer_rewards[state][limiting_type]

    def __eq__(self, other):
        for x, y in zip(self.limiting_types, other.limiting_types):
            if x[0] != y[0] or x[1] != y[1]:
                return False
        return True

    @staticmethod
    def full_acceptance_policy(model):
        out_policy = Policy(model)

        out_policy.limiting_types = []
        for state in range(model.n_states):
            limiting_customer_type = utils.argmin(model.customer_rewards[state])
            limiting_server_type = utils.argmin(model.server_rewards[state])
            out_policy.limiting_types.append([limiting_customer_type, limiting_server_type])

        return out_policy

    @staticmethod
    def full_rejection_policy(model):
        out_policy = Policy(model)

        out_policy.limiting_types = [[-1,-1] for state in range(model.n_states)]

        return out_policy
