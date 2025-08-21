from ac import *
from model import *

if __name__ == "__main__":
    rates = [3,2.4,3]
    epsilons = [0.2,0.2,0.2]
    transition_probs = [[0.2,0.4,0.4],[0.5,0,0.5],[0.4,0.4,0.2]]
    transition_epsilons = [0.1,0.1,0.1]
    pe = MockParameterEstimator(rates, epsilons, transition_probs, transition_epsilons)

    customer_rewards = [[0],[0.5],[0.5]]
    server_rewards = [[0.5],[0.5],[0]]
    abandonment_rewards = [1,0,-1]
    holding_rewards = [0,0,0]

    state_rewards = StateRewards(customer_rewards, server_rewards, abandonment_rewards,holding_rewards)

    model_bounds = ModelBounds(n_classes=[1,1], capacities=[1,1])

    generate_extended_model(model_bounds, pe, state_rewards, 0)
