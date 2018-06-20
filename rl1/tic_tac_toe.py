### Tic Tac Toe
### Matthew Sun
import random
import numpy as np
import json

EPS = 0.1 # using epsilon greedy algorithm

PLAYER1 = 0
PLAYER2 = 1
DRAW = 2
ALPHA = 0.5
EMPTY_TOKEN = ' '

class Player(object):
    def __init__(self, letter='X', player=PLAYER1):
        self.state_history = []
        self.letter = letter
        self.player = player
        self.values = {}

    def take_action(self, env, DEBUG=False):
        grid = env.get_state()
        if grid not in self.values:
            self.values[grid] = 0.5
        
        actions = [i for i, ltr in enumerate(grid) if ltr == EMPTY_TOKEN] # all empty spots
        if DEBUG:
            print('Possible actions to be considered: ', actions)
        vals = []
        for i in actions:
            new_grid = grid[0:i] + self.letter + grid[i+1:]
            if new_grid not in self.values:
                self.values[new_grid] = 0.5
            vals.append(self.values[new_grid])
        
        action = random.choice(actions) if random.random() < EPS else actions[np.argmax(vals)]
        move_loc = (action // 3, action % 3)
        if DEBUG:
            print('Player', str(self.player + 1), 'to place a letter', self.letter, 'at location', move_loc)
        env.register_move(move_loc, self.letter, self.player)
    
    def update_state_history(self, state):
        self.state_history.append(state)

    def update(self, env, DEBUG=False):
        reversed_history = list(reversed(self.state_history))
        if env.outcome == self.player:
            self.values[reversed_history[0]] = 1.0
        else:
            self.values[reversed_history[0]] = 0.0
        
        if DEBUG:
            print('States of game', reversed_history)
        for i in range(1, len(reversed_history)):
            if reversed_history[i] not in self.values:
                self.values[reversed_history[i]] = 0.5
            self.values[reversed_history[i]] = self.values[reversed_history[i]] + ALPHA * (self.values[reversed_history[i - 1]] - self.values[reversed_history[i]])
            if DEBUG:
                print('Value for state', reversed_history[i], self.values[reversed_history[i]])
        
class Environment:
    def __init__(self):
        self.state = [  [EMPTY_TOKEN, EMPTY_TOKEN, EMPTY_TOKEN], 
                        [EMPTY_TOKEN, EMPTY_TOKEN, EMPTY_TOKEN],
                        [EMPTY_TOKEN, EMPTY_TOKEN, EMPTY_TOKEN] ] # all spots begin empty
        self.outcome = -1
        self.last_move_loc = None
        self.last_move_player = None
    
    def get_state(self):
        return ''.join([''.join(x) for x in self.state]) # flatten into string

    def register_move(self, loc, letter, player):
        self.state[loc[0]][loc[1]] = letter
        self.last_move_loc = loc
        self.last_move_player = player

    def generate_candidates(self, loc):
        x_dirs = [0, -1, 1]
        y_dirs = [0, -1, 1]
        candidates = []

        for (x_dir, y_dir) in [(x, y) for x in x_dirs for y in y_dirs]:
            if x_dir == 0 and y_dir == 0: 
                continue
            triple = []
            for i in range(3):
                new_x = loc[0] + x_dir * i
                new_y = loc[1] + y_dir * i
                if new_x < 0 or new_x > 2 or new_y < 0 or new_y > 2: # hard code size of board
                    break # break if out of bounds
                triple.append((loc[0] + x_dir * i, loc[1] + y_dir * i)) # move in the correct directions

            if len(triple) == 3:
                candidates.append(triple)
        
        return candidates

    def game_over(self):
        # check all possible indices around the last move location
        if self.last_move_loc is None:
            return False
        to_check = self.generate_candidates(self.last_move_loc)
        for t in to_check:
            l1 = self.state[t[0][0]][t[0][1]]
            l2 = self.state[t[1][0]][t[1][1]]
            l3 = self.state[t[2][0]][t[2][1]]
            if l1 == l2 and l2 == l3:
                self.outcome = self.last_move_player
                return True
        
        try: 
            if self.get_state().index(EMPTY_TOKEN) != -1: # board is full and draw!
                return False
        except ValueError: # if no empty space was found
            self.outcome = DRAW
            return True
        
        return False
    
    def reset(self):
        self.state = [  [EMPTY_TOKEN, EMPTY_TOKEN, EMPTY_TOKEN], 
                        [EMPTY_TOKEN, EMPTY_TOKEN, EMPTY_TOKEN],
                        [EMPTY_TOKEN, EMPTY_TOKEN, EMPTY_TOKEN] ] # all spots begin empty
        self.outcome = -1
        self.last_move_loc = None
        self.last_move_player = None
    
    def draw_board(self):
        print('\n--------------\n'.join(['  |  '.join(row) for row in self.state]))
        print('##################')

def play_game(p1, p2, env, draw=True):
    current_player = None

    while not env.game_over():
        if current_player == p1:
            current_player = p2
        else:
            current_player = p1

        # draw board!
        if draw:
            env.draw_board()

        # current player makes a move
        current_player.take_action(env, DEBUG=draw)

        # update state histories
        state = env.get_state()
        p1.update_state_history(state)
        p2.update_state_history(state)

    if draw:
        env.draw_board()

    p1.update(env, DEBUG=draw)
    p2.update(env, DEBUG=draw)

player1 = Player(letter='X', player=PLAYER1)
player2 = Player(letter='O', player=PLAYER2)
environment = Environment()

for i in range(100):
    play_game(player1, player2, environment, draw=False if i % 10 != 0 else True)
    environment.reset()

print(json.dumps(player1.values, indent=1))
print(json.dumps(player2.values, indent=1))