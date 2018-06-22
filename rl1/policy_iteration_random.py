import numpy as np
import matplotlib.pyplot as pyplot
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy

SMALL_ENOUGH = 10e-4
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U','D', 'L', 'R')

# this is deterministic
# all p(s', r|s, a) = 1 or 0

if __name__ == '__main__':
    # reward of -0.1 for every non-terminal state
    grid = negative_grid(step_cost=-1.0)
    
    # print rewards
    print("rewards:")
    print_values(grid.rewards, grid)

    # state -> action
    # we'll randomly choose an action and update as we learn

    policy = {}
    states = grid.all_states()
    for s in grid.actions.keys():
        policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)

    # initialize V(s)
    V = {}
    for s in states:
        V[s] = 0
    
    # repeat until policy does not change
    # repeat til convergence
    policy_changed = True
    while policy_changed:
        while True:
            biggest_change = 0
            for s in states:
                if s in policy:
                    old_v = V[s]
                    V[s] = 0
                    for action in ALL_POSSIBLE_ACTIONS:
                        p = 0.5 if action == policy[s] else 0.5/3
                        grid.set_state(s)
                        reward = grid.move(action)
                        V[s] += p * (reward + GAMMA * V[grid.current_state()])
                    biggest_change = np.max([biggest_change, abs(V[s] - old_v)])
            
            if biggest_change < SMALL_ENOUGH:
                break
        policy_changed = False
        for s in policy.keys():
            cur_action = policy[s]
            actions = grid.actions[s]
            action_v = []
            for a in actions:
                grid.set_state(s)
                reward = grid.move(a)
                action_v.append(reward + V[grid.current_state()])
            best_action = actions[np.argmax(action_v)]
            policy_changed = policy_changed or best_action != cur_action
            policy[s] = best_action

    print("Values for policy iteration:")
    print_values(V, grid)
    print("\n\n")

    print("Policy for policy iteration:")
    print_policy(policy, grid)
    print("\n\n")
