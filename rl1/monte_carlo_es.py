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
    actions = []
    rewards = []
    states.append(start_state)
    rewards.append(0)
    action = np.random.choice(ALL_POSSIBLE_ACTIONS)
    seen_states = set()
    while not grid.game_over():
        actions.append(action)
        reward = grid.move(action)
        states.append(grid.current_state())
        if grid.current_state() in seen_states:
            # print('JEEPERS')
            # print(states)
            # print(actions)
            rewards.append(-100)
            break
        seen_states.add(grid.current_state())
        rewards.append(reward)
        action = policy[grid.current_state()] if grid.current_state() in policy else None

    returns = returns_from_rewards(states, rewards)
    return states[:-1], actions, returns[:-1] # the last return is always 0, disregard this!!

def returns_from_rewards(states, rewards):
    returns = []
    G = 0
    for s, reward in zip(reversed(states), reversed(rewards)):
        returns.append(G)
        G = reward + GAMMA * G
    returns.reverse()
    return returns

def first_visit_monte_carlo_iteration(policy, grid):
    Q = defaultdict(float)
    times_seen = defaultdict(int)
    biggest_change = 0

    for i in range(2000):
        print(i)
        states, actions, returns = play_episode(policy, grid)
        states_seen = set()
        for s, a, g in zip(states, actions, returns):
            if (s, a) not in states_seen:
                old_q = Q[(s,a)]
                Q[(s, a)] = times_seen[(s, a)] / (times_seen[(s, a)] + 1) * Q[(s, a)] + 1.0 / (times_seen[(s, a)] + 1) * g # moving average
                times_seen[(s, a)] += 1
                states_seen.add((s, a))
        
        for s in grid.actions.keys():
            best_action = None
            max_Q = float('-inf')
            for a in ALL_POSSIBLE_ACTIONS:
                if Q[(s, a)] > max_Q:
                    best_action = a 
                    max_Q = Q[(s, a)]
            policy[s] = best_action
        
    return policy, Q

if __name__ == '__main__':
    # iterative policy evaluation
    # given a policy, let's find its value function V(s)
    grid = negative_grid(step_cost=-0.9)

    print("Rewards:")
    print_values(grid.rewards, grid)

    # state -> action
    # we'll randomly choose an action and update as we learn

    policy = {}
    states = grid.all_states()
    for s in grid.actions.keys():
        policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)

    print("Random policy:")
    print_policy(policy, grid)

    policy, Q = first_visit_monte_carlo_iteration(policy, grid)

    V = {}
    for s in grid.actions.keys():
        max_Q = float('-inf')
        for a in ALL_POSSIBLE_ACTIONS:
            if Q[(s, a)] > max_Q:
                max_Q = Q[(s, a)]
        V[s] = max_Q

    print("Learned policy:")
    print_policy(policy, grid)

    print("Learned value function:")
    print_values(V, grid)

