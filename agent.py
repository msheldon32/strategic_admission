import policy

class Agent:
    def __init__(self, model):
        self.model = model

    def act(self, state, transition_type):
        if transition_type == 0:
            return True
        return random.choice([True, False])

    def observe(self,state, next_state, transition_type, reward, time_elapsed):
        pass

class DeterministicAgent(Agent):
    def __init__(self, model, policy):
        super().__init__(model)
        self.policy = policy

    def act(self, state, transition_type):
        if transition_type == 0:
            return True
        
        return self.policy.act(state, transition_type)
    
    def get_estimated_gain(self):
        return self.model.get_gain_bias(self.policy)[1]

class KnownPOAgent(Agent):
    def __init__(self, model):
        super().__init__(model)
        self.policy = policy.Policy.full_acceptance_policy(model)

        while True:
            new_policy = self.policy.get_improved_policy()
            if self.policy == new_policy:
                break
            self.policy = new_policy

    def act(self, state, transition_type):
        if transition_type == 0:
            return True
        
        return self.policy.act(state, transition_type)
    
    def get_estimated_gain(self):
        return self.model.get_gain_bias(self.policy)[1]
