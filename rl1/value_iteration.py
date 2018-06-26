import numpy as np
import matplotlib.pyplot as pyplot
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy

SMALL_ENOUGH = 1e-3
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U','D', 'L', 'R')

# this is deterministic
# all p(s', r|s, a) = 1 or 0

if __name__ == '__main__':
    grid = negative_grid(step_cost=-0.1)
    
    # print rewards
    print("rewards:")
    print_values(grid.rewards, grid)

    # state -> action
    # we'll randomly choose an action and update as we learn

    policy = {}
    states = grid.all_states()
    for s in grid.actions.keys():
        policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)

    # initial policy
    print("initial policy:")
    print_policy(policy, grid)

    # initialize V(s)
    V = {}
    for s in states:
        if s in grid.actions:
            V[s] = np.random.random()
        else:
            V[s] = 0
    
    # value iteration
    while True:
        biggest_change = 0
        for s in states:
            old_v = V[s]
            new_v = 0
            action_v = []
            if s in grid.actions:
                for a in ALL_POSSIBLE_ACTIONS:
                    grid.set_state(s)
                    reward = grid.move(a)
                    cur_val = reward + GAMMA * V[grid.current_state()]
                    action_v.append(cur_val)
                V[s] = np.max(action_v)
                biggest_change = max(biggest_change, abs(V[s] - old_v))
        
        if biggest_change < SMALL_ENOUGH:
            break
    
    for s in policy.keys():
        action_v = []
        for a in ALL_POSSIBLE_ACTIONS:
            grid.set_state(s)
            reward = grid.move(a)
            cur_val = reward + GAMMA * V[grid.current_state()]
            action_v.append(cur_val)
        best_action = ALL_POSSIBLE_ACTIONS[np.argmax(action_v)]
        policy[s] = best_action

    print("Values for value iteration:")
    print_values(V, grid)
    print("\n\n")

    print("Policy for value iteration:")
    print_policy(policy, grid)
    print("\n\n")
