import collections

class Observer:
    def __init__(self, model):
        self.transitions = []
        self.states = []

        self.total_time = 0
        self.total_reward = 0
        self.step_rewards = []
        self.step_times = []
        self.step_mean_rewards = []
        self.step_states = []

        self.transitions = []

        self.model = model

    def observe(self, state, next_state, transition_type, reward, time_elapsed):
        self.total_time += time_elapsed
        self.total_reward += reward

        self.step_rewards.append(reward)
        self.step_times.append(time_elapsed)
        self.step_states.append(state)

        self.transitions.append((state, next_state))

    def get_gain(self):
        return self.total_reward/self.total_time

    def get_avg_reward_state(self):
        reward_dict = collections.defaultdict(lambda: [0,0])

        for state, time, reward in zip(self.step_states, self.step_times, self.step_rewards):
            reward_dict[state][0] += time
            reward_dict[state][1] += reward

        reward_vec = []

        for state in range(self.model.n_states):
            if reward_dict[state][0] == 0:
                reward_vec.append(0)
            else:
                reward_vec.append(reward_dict[state][1]/reward_dict[state][0])
        return reward_vec

    def get_total_time_in_state(self):
        time_dict = collections.defaultdict(lambda: 0)

        for state, time in zip(self.step_states, self.step_times):
            time_dict[state] += time

        time_vec = []

        for state in range(self.model.n_states):
            time_vec.append(time_dict[state])
        return time_vec

    def get_empirical_rates(self):
        transition_dict = collections.defaultdict(lambda: 0)

        time_vec = self.get_total_time_in_state()

        for ttup in self.transitions:
            transition_dict[ttup] += 1

        rate_vec = []
        for state in range(self.model.n_states):
            rate_down = transition_dict[(state, state-1)]/time_vec[state]
            rate_up = transition_dict[(state, state+1)]/time_vec[state]
            rate_vec.append([rate_down, rate_up])

        return rate_vec

    def get_empirical_probs(self):
        time_vec = self.get_total_time_in_state()
        total_time = sum(time_vec)
        return [x/total_time for x in time_vec]
    
    def reset(self):
        self.transitions = []
        self.states = []

        self.total_time = 0
        self.total_reward = 0
        self.step_rewards = []
        self.step_times = []
        self.step_mean_rewards = []
        self.step_states = []

        self.transitions = []
