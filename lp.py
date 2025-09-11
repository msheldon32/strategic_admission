import gurobipy as gp
from gurobipy import GRB


import policy

def get_optimal_policy(model, v_basis=None, c_basis=None):
    m = gp.Model("linear_programming")
    
    m.setParam(GRB.Param.OutputFlag, 0)
    m.setParam(GRB.Param.OptimalityTol, 1e-9)
    m.setParam(GRB.Param.FeasibilityTol, 1e-9)

    n_states = model.n_states
    n_actions = (model.n_customer_types+1)*(model.n_server_types+1)

    bias_values = m.addMVar(n_states, name="bias", lb=float("-inf"))
    gain = m.addVar(lb=float("-inf"), name="gain")
    slack = m.addMVar((n_states, n_actions), name="slack", lb=0)

    action_mapping = []

    for i in range(-1, model.n_customer_types):
        for j in range(-1, model.n_server_types):
            action_mapping.append((i,j))

    # bias must be greater than the bias for each action
    for state in range(n_states):
        for action_no, action in enumerate(action_mapping):
            rate_down, rate_up = model.get_transition_rates(state, action)
            reward = model.get_mean_reward(state, action)
            adjacent_bias = (rate_up*bias_values[state+1] if state < n_states-1 else 0) + (rate_down*bias_values[state-1] if state > 0 else 0)

            unnorm_bias = reward - gain + adjacent_bias

            m.addConstr((rate_up+rate_down)*bias_values[state] == unnorm_bias + slack[state, action_no])

    m.addConstr(bias_values[model.capacities[1]] == 0)
    m.setObjective(gain, GRB.MINIMIZE)
    m.update()
    if v_basis is not None:
        m.setAttr("VBasis", m.getVars(), v_basis)
        m.setAttr("CBasis", m.getConstrs(), c_basis)
    m.optimize()
    
    out_policy = policy.Policy(model)

    for state in range(n_states):
        out_policy.limiting_types.append((-1,-1))
        min_slack = float("inf")
        for action_no, action in enumerate(action_mapping):
            if slack.X[state, action_no] < min_slack:
                min_slack = slack.X[state,action_no]
                out_policy.limiting_types[-1] = action
    v_basis = m.getAttr("VBasis", m.getVars())
    c_basis = m.getAttr("CBasis", m.getConstrs())
    return out_policy, gain.X, v_basis, c_basis

