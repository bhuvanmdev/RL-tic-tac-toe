import torch
import random
import numpy as np


class Q_learning:
    def __init__(self, letter, rand=False,alpha=0.25 , gamma=0.95):
        self.alpha = alpha
        self.gamma = gamma
        self.q = {}
        self.cur_board = None
        self.cur_move = None
        self.letter = letter
        self.epsilon = 200 # make it 0 during infernece
        self.rand = rand

    def convert_board_to_num_array(self, board):
        return tuple([1 if spot == 'X' else 0 if spot == ' ' else -1 for spot in board])

    def get_q(self, state, action):
        return self.q.get((state, action),0)

    def get_action(self, game ):
        state = self.convert_board_to_num_array(game.board)
        if self.rand == None:
            action = int(input("Enter the position: "))
        elif random.randint(1,200) < self.epsilon - game.x_wins-game.tie   or self.rand == True:
            action = random.choice(game.available_moves())
        else:
            q_values = np.array([self.get_q(state, action) for action in game.available_moves()])
            action = game.available_moves()[np.argmax(q_values)]
        self.cur_board = state
        self.cur_move = action
        return action

    def update(self, game ):
        reward = 0
        if self.rand or self.rand is None:return
        if game.current_winner == self.letter:
            reward = 1
            # game.x_wins += 1 if self.letter == 'X' else 0
            # game.o_wins += 1 if self.letter == 'O' else 0
        elif game.current_winner == 'T':
            reward = -10
            # game.tie += 1
        elif game.current_winner and game.current_winner != 'T':
            reward = -10
            # game.x_wins += 0 if self.letter == 'X' else 1
            # game.o_wins += 0 if self.letter == 'O' else 1
        new_state = self.convert_board_to_num_array(game.board.copy())
        if game.available_moves().__len__() != 0:
            q_values = np.array([self.get_q(new_state, action) for action in game.available_moves()])
            # print(q_values,game.available_moves())
            new_q = reward + self.gamma * np.max(q_values)  - self.get_q(self.cur_board, self.cur_move)
        else:
            new_q = reward
        self.q[(self.cur_board, self.cur_move)] = self.get_q(self.cur_board, self.cur_move) + self.alpha * new_q

    def save_q(self, filename):
        torch.save(self.q, filename)

    def load_q(self, filename):
        self.q = torch.load(filename)

    def print_q(self):
        for key in self.q:
            print(key, self.q[key])