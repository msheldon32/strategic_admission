import math
import random

import policy
import model
import ac

class Agent:
    def __init__(self):
        pass

    def act(self, state, transition_type):
        if transition_type == 0:
            return True
        return random.choice([True, False])

    def observe(self,state, next_state, transition_type, reward, time_elapsed):
        pass

class DeterministicAgent(Agent):
    def __init__(self, model, policy):
        super().__init__()
        self.model = model
        self.policy = policy

    def act(self, state, transition_type):
        if transition_type == 0:
            return True
        
        return self.policy.act(state, transition_type)
    
    def get_estimated_gain(self):
        return self.model.get_gain_bias(self.policy)[1]

class KnownPOAgent(Agent):
    def __init__(self, model):
        super().__init__()
        self.model = model
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

class ACRLAgent(Agent):
    def __init__(self, model_bounds, state_rewards):
        # make sure there's +1 classes for handling abandonments
        super().__init__()
        self.parameter_estimator = ac.ParameterEstimator(model_bounds)
        self.state_rewards = state_rewards
        self.model_bounds = model_bounds
        self.exploration = ac.Exploration(model_bounds)


        self.initial_confidence_param = 0.9999
        self.model = ac.generate_extended_model(model_bounds, self.parameter_estimator, self.state_rewards, self.initial_confidence_param)
        self.policy = policy.Policy.full_acceptance_policy(self.model)
        self.update_policy()
        n_policies = 1

    def update_policy(self):
        while True:
            new_policy = self.policy.get_improved_policy()
            if self.policy == new_policy:
                return
            self.policy = new_policy
        self.n_policies += 1
        if self.n_policies >= 10:
            raise Exception("stop - several rounds of policies")

    def act(self, state, transition_type):
        if transition_type == 0:
            return True
        
        return self.policy.act(state, transition_type)

    def observe(self,state, next_state, transition_type, reward, time_elapsed):
        self.parameter_estimator.observe(state, transition_type, time_elapsed)
        if self.exploration.observe(state):
            self.exploration.new_episode()
            self.model = ac.generate_extended_model(self.model_bounds, self.parameter_estimator, self.state_rewards, self.initial_confidence_param/math.sqrt(self.exploration.steps_before_episode))
            self.policy.model = self.model
            self.update_policy()
            #if self.exploration.n_episodes == 20:
            #    raise Exception("stop")
    
    def get_estimated_gain(self):
        return self.model.get_gain_bias(self.policy)[1]
