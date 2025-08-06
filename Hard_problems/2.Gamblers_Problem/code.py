def gambler_value_iteration(ph, theta=1e-9):
    """
    Computes the optimal value function and policy for the Gambler's Problem.
    Args:
      ph: probability of heads
      theta: convergence threshold
    Returns:
      V: list of values for all states 0..100
      policy: list of optimal stakes for all states 0..100
    """
    V = [0.0] * 101  # Initialize value function
    policy = [0] * 101  # Initialize policy
    V[100] = 1.0  # Terminal state value:
    def _update():
        delta = 0.0
        for s in range(1, 100):
            best_value = float('-inf')
            best_stake = 0
            for stake in range(1, min(s, 100 - s) + 1):
                expected_value = (ph * V[s + stake]) + ((1 - ph) * V[s - stake])
                if expected_value > best_value:
                    best_value = expected_value
                    best_stake = stake
            delta = max(delta, abs(V[s] - best_value))
            V[s] = best_value
            policy[s] = best_stake
        return delta
    while _update() > theta:
        pass
    return V, policy    
