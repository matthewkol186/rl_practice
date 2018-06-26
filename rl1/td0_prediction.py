import numpy as np
import random
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy
from collections import defaultdict

# monte carlo policy evaluation in windy gridworld

SMALL_ENOUGH = 10e-4
GAMMA = 0.9
ALPHA = 0.1
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')

def random_action(a, eps = 0.1):
    p = np.random.random()
    if p < (1 - eps):
        return a
    else:
        return np.random.choice(ALL_POSSIBLE_ACTIONS)

def play_episode(policy, grid):
    grid.set_state((2,0)) # start at random state
    start_state = grid.current_state()
    states = []
    rewards = []
    states.append(start_state)
    rewards.append(0)
    action = random_action(policy[start_state])
    seen_states = set()
    while not grid.game_over():
        reward = grid.move(action)
        states.append(grid.current_state())
        seen_states.add(grid.current_state())
        rewards.append(reward)
        action = random_action(policy[grid.current_state()]) if not grid.game_over() else None
        
    return states, rewards # the last return is always 0, disregard this!!

def first_visit_td0_prediction(policy, N, grid):
    V = defaultdict(float)
    times_seen = defaultdict(int)

    for i in range(N):
        states, rewards = play_episode(policy, grid)
        states_seen = set()
        for i in range(len(states)):
            s = states[i]
            r = rewards[i + 1] if i+1<len(states) else 0
            if s not in states_seen:
                V[s] = V[s] + ALPHA * (r + GAMMA * V[states[i+1] if i+1<len(states) else 'blorp'] - V[s]) # moving average
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
        (1,2): 'R',
        (2,1): 'R',
        (2,2): 'R',
        (2,3): 'U',
    }

    print_policy(policy, grid)

    V = first_visit_td0_prediction(policy, 5000, grid)
    
    print("Values for fixed policy:")
    print_values(V, grid)

