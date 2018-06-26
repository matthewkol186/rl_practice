import numpy as np
import random
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy
from collections import defaultdict

# monte carlo policy evaluation in windy gridworld

SMALL_ENOUGH = 10e-4
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')

def play_episode(policy, grid):
    grid.set_state(random.choice(list(grid.actions.keys()))) # start at random state
    start_state = grid.current_state()
    states = []
    rewards = []
    states.append(start_state)
    rewards.append(0)
    while not grid.game_over():
        action = policy[grid.current_state()]
        probs = [ 0.5 if a == action else 0.5/3 for a in ALL_POSSIBLE_ACTIONS]
        reward = grid.move(np.random.choice(ALL_POSSIBLE_ACTIONS, p = probs)) # randomness
        states.append(grid.current_state())
        rewards.append(reward)
    
    returns = returns_from_rewards(states, rewards)
    return states[:-1], returns[:-1] # the last return is always 0, disregard this!!

def returns_from_rewards(states, rewards):
    returns = []
    G = 0
    for s, reward in zip(reversed(states), reversed(rewards)):
        returns.append(G)
        G = reward + GAMMA * G
    returns.reverse()
    return returns

def first_visit_monte_carlo_prediction(policy, N, grid):
    V = defaultdict(float)
    times_seen = defaultdict(int)

    for i in range(N):
        states, returns = play_episode(policy, grid)
        states_seen = set()
        for s, g in zip(states, returns):
            if s not in states_seen:
                V[s] = times_seen[s] / (times_seen[s] + 1) * V[s] + 1.0 / (times_seen[s] + 1) * g # moving average
                times_seen[s] += 1
                states_seen.add(s)
    return V

if __name__ == '__main__':
    # iterative policy evaluation
    # given a policy, let's find its value function V(s)
    grid = standard_grid()

    ### fixed policy shiznits ###
    policy = {
        (2,0): 'U',
        (1,0): 'U',
        (0,0): 'R',
        (0,1): 'R',
        (0,2): 'R',
        (1,2): 'U',
        (2,1): 'L',
        (2,2): 'U',
        (2,3): 'L',
    }

    print_policy(policy, grid)

    V = first_visit_monte_carlo_prediction(policy, 5000, grid)
    
    print("Values for fixed policy:")
    print_values(V, grid)

