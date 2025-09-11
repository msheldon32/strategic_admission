import collections
import matplotlib.pyplot as plt

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

    def get_past_n_gain(self, n):
        reward_upto = sum(self.step_rewards[-n:])
        time_upto = sum(self.step_times[-n:])
        return reward_upto/time_upto

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

    def plot_regret(self, ideal_gain, color="r"):
        regret = [0]
        cum_regret = [0]

        for t, x in zip(self.step_times, self.step_rewards):
            r = (ideal_gain*t) - x
            regret.append(r)
            cum_regret.append(cum_regret[-1] + r)

        plt.plot(cum_regret, color=color)
        #plt.show()
    
    def plot_total_reward(self, ideal_gain):
        cum_reward = [0]
        cum_gain = [0]

        for t, x in zip(self.step_times, self.step_rewards):
            g = (ideal_gain*t)
            cum_gain.append(cum_gain[-1]+g)
            cum_reward.append(cum_reward[-1]+x)
        
        plt.plot(cum_reward)
        plt.plot(cum_gain)
        plt.show()

    def summarize(self, ideal_gain, timestep=10000):
        cum_reward = [0]
        cum_regret = [0]
        cum_time = [0]
        for t, x in zip(self.step_times, self.step_rewards):
            g = (ideal_gain*t)
            cum_regret.append(cum_regret[-1]+(g- x))
            cum_reward.append(cum_reward[-1]+x)
            cum_time.append(cum_time[-1]+t)
        
        cum_reward_tstep = []
        cum_regret_tstep = []
        cum_time_tstep = []
        t = 0

        while (t+1) < len(cum_reward):
            cum_reward_tstep.append(cum_reward[t+1])
            cum_regret_tstep.append(cum_regret[t+1])
            cum_time_tstep.append(cum_time[t+1])
            t += timestep

        return {
                "regret": cum_regret_tstep,
                "reward": cum_reward_tstep,
                "time": cum_time_tstep,
                "ideal_gain": ideal_gain
            }
