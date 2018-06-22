import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid

SMALL_ENOUGH = 10e-4

def print_values(V, g):
    for i in range(g.width):
        print("------------------------")
        for j in range(g.height):
            v = V.get((i,j), 0)
            if v >= 0:
                print(" %.2f|" % v, end='')
            else:
                print("%.2f|" % v, end='')
        print('')

def print_policy(P, g):
    for i in range(g.width):
        print("------------------------")
        for j in range(g.height):
            a = P.get((i,j), ' ')
            print("  %s  |" % a, end='')
        print("")

if __name__ == '__main__':
    # iterative policy evaluation
    # given a policy, let's find its value function V(s)
    grid = standard_grid()
    states = grid.all_states()

    V = {}
    for s in states:
        V[s] = 0
    gamma = 1.0
    # repeat til convergence
    while True:
        biggest_change = 0
        for s in states:
            if not grid.is_terminal(s):
                old_v = V[s]
                V[s] = 0
                for action in grid.actions[s]:
                    grid.set_state(s)
                    reward = grid.move(action)
                    V[s] += 1.0 / len(grid.actions[s]) * (reward + gamma * V[grid.current_state()])
                biggest_change = np.max([biggest_change, abs(V[s] - old_v)])
        
        if biggest_change < SMALL_ENOUGH:
            break
    print("Values for uniformly random actions:")
    print_values(V, grid)
    print("\n\n")

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

    # initialize V(s) = 0
    V = {}
    for s in states:
        V[s] = 0
    
    # let's see how V(s) changes as we get further away from the reward
    gamma = 0.9 # discount factor

    while True:
        biggest_change = 0
        for s in states:
            old_v = V[s]

            if s in policy:
                a = policy[s]
                grid.set_state(s)
                r = grid.move(a)
                V[s] = r + gamma * V[grid.current_state()]
                biggest_change = max(biggest_change, abs(old_v - V[s]))

        if biggest_change < SMALL_ENOUGH:
            break
    
    print("Values for fixed policy:")
    print_values(V, grid)

