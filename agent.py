import math
import random

import policy
import model
import ac
import ac_ablation

import ucrl

class Agent:
    def __init__(self):
        pass

    def act(self, state, transition_type):
        if transition_type == 0:
            return True
        return random.choice([True, False])

    def observe(self,state, next_state, action, transition_type, reward, time_elapsed):
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

        for i in range(10000):
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


        self.initial_confidence_param = 0.5
        self.model, failed = ac.generate_extended_model(model_bounds, self.parameter_estimator, self.state_rewards, self.initial_confidence_param)
        self.policy = policy.Policy.full_acceptance_policy(self.model)
        self.n_policies = 1
        self.update_policy()

    def update_policy(self):
        for i in range(1000):
            new_policy = self.policy.get_improved_policy()
            if self.policy == new_policy:
                return
            self.policy = new_policy
        self.n_policies += 1
        #if self.n_policies >= 10:
        #    raise Exception("stop - several rounds of policies")

    def get_confidence_param(self):
        return self.initial_confidence_param/self.exploration.steps_before_episode

    def act(self, state, transition_type):
        if transition_type == 0:
            return True
        
        return self.policy.act(state, transition_type)

    def observe(self,state, next_state, action, transition_type, reward, time_elapsed):
        self.parameter_estimator.observe(state, transition_type, time_elapsed)
        if self.exploration.observe(state):
            self.exploration.new_episode()
            model, failed = ac.generate_extended_model(self.model_bounds, self.parameter_estimator, self.state_rewards, self.initial_confidence_param/self.exploration.steps_before_episode)
            if not failed:
                self.model = model
                self.policy.model = self.model
                self.update_policy()
                #if self.exploration.n_episodes == 20:
                #    raise Exception("stop")
    
    def get_estimated_gain(self):
        return self.model.get_gain_bias(self.policy)[1]

class AblationACRLAgent(Agent):
    def __init__(self, model_bounds, state_rewards):
        # make sure there's +1 classes for handling abandonments
        super().__init__()
        self.parameter_estimator = ac.ParameterEstimator(model_bounds)
        self.state_rewards = state_rewards
        self.model_bounds = model_bounds
        self.exploration = ac.Exploration(model_bounds)


        self.initial_confidence_param = 0.5
        self.model, failed = ac_ablation.generate_extended_model_ablation(model_bounds, self.parameter_estimator, self.state_rewards, self.initial_confidence_param)
        self.policy = policy.Policy.full_acceptance_policy(self.model)
        self.n_policies = 1
        self.update_policy()

    def update_policy(self):
        for i in range(1000):
            new_policy = self.policy.get_improved_policy()
            if self.policy == new_policy:
                return
            self.policy = new_policy
        self.n_policies += 1
        #if self.n_policies >= 10:
        #    raise Exception("stop - several rounds of policies")

    def get_confidence_param(self):
        return self.initial_confidence_param/self.exploration.steps_before_episode

    def act(self, state, transition_type):
        if transition_type == 0:
            return True
        
        return self.policy.act(state, transition_type)

    def observe(self,state, next_state, action, transition_type, reward, time_elapsed):
        self.parameter_estimator.observe(state, transition_type, time_elapsed)
        if self.exploration.observe(state):
            self.exploration.new_episode()
            model, failed = ac_ablation.generate_extended_model_ablation(self.model_bounds, self.parameter_estimator, self.state_rewards, self.initial_confidence_param/self.exploration.steps_before_episode)
            if not failed:
                self.model = model
                self.policy.model = self.model
                self.update_policy()
                #if self.exploration.n_episodes == 20:
                #    raise Exception("stop")
    
    def get_estimated_gain(self):
        return self.model.get_gain_bias(self.policy)[1]

class ClassicalACRLAgent(Agent):
    def __init__(self, model_bounds, state_rewards):
        # make sure there's +1 classes for handling abandonments
        super().__init__()
        self.parameter_estimator = ac.ClassicalParameterEstimator(model_bounds)
        self.state_rewards = state_rewards
        self.model_bounds = model_bounds
        self.exploration = ac.Exploration(model_bounds)


        self.initial_confidence_param = 0.5
        self.model = ac.generate_extended_model(model_bounds, self.parameter_estimator, self.state_rewards, self.initial_confidence_param)
        self.policy = policy.Policy.full_acceptance_policy(self.model)
        self.update_policy()
        self.n_policies = 1

    def update_policy(self):
        for i in range(1000):
            new_policy = self.policy.get_improved_policy()
            if self.policy == new_policy:
                return
            self.policy = new_policy
        self.policy = self.policy.clean()
        self.n_policies += 1
        #if self.n_policies >= 10:
        #    raise Exception("stop - several rounds of policies")

    def act(self, state, transition_type):
        if transition_type == 0:
            return True
        
        return self.policy.act(state, transition_type)

    def observe(self,state, next_state, action, transition_type, reward, time_elapsed):
        self.parameter_estimator.observe(state, transition_type, time_elapsed)
        if self.exploration.observe(state):
            self.exploration.new_episode()
            model, failed = ac.generate_extended_model(self.model_bounds, self.parameter_estimator, self.state_rewards, self.initial_confidence_param/(self.exploration.steps_before_episode))
            if not failed:
                self.model = model
                self.policy.model = self.model
                self.update_policy()
                #if self.exploration.n_episodes == 20:
                #    raise Exception("stop")
    
    def get_estimated_gain(self):
        return self.model.get_gain_bias(self.policy)[1]

class AlternateUCRLAgent(Agent):
    def __init__(self, model_bounds, state_rewards):
        super().__init__()
        self.parameter_estimator = ucrl.ParameterEstimator(model_bounds)
        self.state_rewards = state_rewards
        self.model_bounds = model_bounds
        self.exploration = ucrl.Exploration(model_bounds)

        self.initial_confidence_param = 0.5
        
        self.policy = [1 for x in range(self.model_bounds.n_states * self.model_bounds.n_transitions)]
        self.update_policy()
        
        self.last_observation = []

    def update_policy(self):
        confidence_param = self.initial_confidence_param/self.exploration.steps_before_episode

        self.policy = ucrl.get_eva_policy(self.parameter_estimator, self.model_bounds, confidence_param, self.exploration.steps_before_episode)

    def state_conversion(self, state, transition_type):
        return (state*self.model_bounds.n_transitions)+(transition_type+self.model_bounds.n_classes[1])

    def act(self, state, transition_type):
        new_state = self.state_conversion(state, transition_type)
        if len(self.last_observation) > 0:
            self.analyze_transition(new_state)

        if transition_type == 0:
            return True

        return self.policy[new_state] == 1

    def analyze_transition(self, new_state):
        state, action, time_elapsed, reward = self.last_observation
        self.parameter_estimator.observe(state, new_state, action, time_elapsed, reward)
        if self.exploration.observe(state, action):
            self.exploration.new_episode()
            self.update_policy()

    def observe(self, state, next_state, action, transition_type, reward, time_elapsed):
        # we don't *really* know the next state here
        conv_state = self.state_conversion(state, transition_type)
        self.last_observation = [conv_state, action, time_elapsed, reward]

    def print(self):
        confidence_param = self.initial_confidence_param/self.exploration.steps_before_episode
        print("-------------------------------------------")
        print("Policy:")
        print(self.policy)
        print("-------------------------------------------")
        print("Beliefs:")
        print(self.parameter_estimator.print(confidence_param))

class UCRLAgent(Agent):
    def __init__(self, model_bounds, state_rewards):
        super().__init__()
        self.state_rewards = state_rewards
        self.model_bounds = model_bounds
        self.exploration = ucrl.Exploration(model_bounds, model_bounds.n_states, model_bounds.n_actions)
        self.parameter_estimator = ucrl.ParameterEstimator(model_bounds, model_bounds.n_states, model_bounds.n_actions)

        self.initial_confidence_param = 0.5
        
        self.policy = [1 for x in range(self.model_bounds.n_states)]
        self.update_policy()
        
        self.last_observation = []

        # enumerate the action space
        self.expanded_as = []
        for customer_type in range(-1, self.model_bounds.n_classes[0]):
            for server_type in range(-1, self.model_bounds.n_classes[1]):
                self.expanded_as.append([customer_type, server_type])

    def update_policy(self):
        confidence_param = self.initial_confidence_param/self.exploration.steps_before_episode

        self.policy = ucrl.get_eva_policy(self.parameter_estimator, self.model_bounds, confidence_param, self.exploration.steps_before_episode)

    def action_conversion(self, action):
        return self.expanded_as[action]

    def accept_customer(self, state, action, customer_type):
        limiting_customer, limiting_server = self.action_conversion(action)

        if limiting_customer == -1:
            return False 
        return self.state_rewards.customer_rewards[state][customer_type] >= self.state_rewards.customer_rewards[state][limiting_customer]

    def accept_server(self, state, action, server_type):
        limiting_customer, limiting_server = self.action_conversion(action)

        if limiting_server == -1:
            return False 
        return self.state_rewards.server_rewards[state][server_type] >= self.state_rewards.server_rewards[state][limiting_server]

    def act(self, state, transition_type):
        action = self.policy[state]
        if transition_type == 0:
            return True
        elif transition_type < 0:
            return self.accept_server(state, action, (-transition_type)-1)
        return self.accept_customer(state, action, transition_type-1)

    def observe(self, state, next_state, accepted, transition_type, reward, time_elapsed):
        action = self.policy[state]
        self.parameter_estimator.observe(state, next_state, action, time_elapsed, reward)
        if self.exploration.observe(state, action):
            self.exploration.new_episode()
            self.update_policy()

    def print(self):
        confidence_param = self.initial_confidence_param/self.exploration.steps_before_episode
        print("-------------------------------------------")
        print("Policy:")
        print(self.policy)
        print("-------------------------------------------")
        print("Beliefs:")
        print(self.parameter_estimator.print(confidence_param))
