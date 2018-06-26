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
    # grid = negative_grid(step_cost=-1.0)
    grid = standard_grid()
    
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
    
    # repeat until policy does not change
    # repeat til convergence
    policy_changed = True
    while policy_changed:

        # policy evaluation step
        while True:
            biggest_change = 0
            for s in states:
                old_v = V[s]
                new_v = 0
                if s in policy:
                    for action in ALL_POSSIBLE_ACTIONS:
                        p = 0.5 if action == policy[s] else 0.5/3
                        grid.set_state(s)
                        reward = grid.move(action)
                        new_v += p * (reward + GAMMA * V[grid.current_state()])
                    V[s] = new_v
                    biggest_change = max(biggest_change, abs(V[s] - old_v))
            
            if biggest_change < SMALL_ENOUGH:
                break
        
        policy_changed = False
        for s in policy.keys():
            cur_action = policy[s]
            action_v = []
            for a in ALL_POSSIBLE_ACTIONS:
                cur_val = 0
                for action in ALL_POSSIBLE_ACTIONS:
                    p = 0.5 if action == a else 0.5/3
                    grid.set_state(s)
                    reward = grid.move(action)
                    cur_val += p * (reward + GAMMA * V[grid.current_state()])
                action_v.append(cur_val)
            best_action = ALL_POSSIBLE_ACTIONS[np.argmax(action_v)]
            policy_changed = policy_changed or best_action != cur_action
            policy[s] = best_action

    print("Values for policy iteration:")
    print_values(V, grid)
    print("\n\n")

    print("Policy for policy iteration:")
    print_policy(policy, grid)
    print("\n\n")
