def sarsa_update(transitions, initial_states, alpha, gamma, max_steps):
    """
    Perform SARSA updates on the given environment transitions.
    Args:
        transitions (dict): mapping (state, action) -> (reward, next_state)
        initial_states (list): list of starting states to simulate episodes from
        alpha (float): learning rate
        gamma (float): discount factor
        max_steps (int): maximum steps allowed per episode
    Returns:
        dict: final Q-table as a dictionary {(state, action): value}
    """
    # Initialize Q-table
    Q = {}
    for (state, action), (reward, next_state) in transitions.items():
        Q[(state, action)] = 0
    actions = set(action for (state, action) in transitions.keys())


    def get_action_greedy_(state):
        return max((transitions.get((state, a), (-float('inf'), None)), a) for a in actions)[1]
    for init_state in initial_states:
        state = init_state
        action = get_action_greedy_(state)
        for step in range(max_steps):
            if state == "terminal":
                break
            reward, next_state = transitions.get((state, action), (0, "terminal"))
            next_action = get_action_greedy_(next_state)
            Q[(state, action)] += alpha * (reward + gamma * Q.get((next_state, next_action), 0) - Q[(state, action)])
            state = next_state
            action = next_action

    return Q
