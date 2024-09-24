import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import random
import numpy as np

BATCH_SIZE = 128


class Agent:
    def __init__(self, letter,alpha,gamma,episilon):
        self.letter = letter
        self.cur_board = None
        self.cur_move = None
        self.epsilon = episilon # for 200 games there is randomness
        self.alpha = alpha
        self.gamma = gamma

    def convert_board_to_num_array(self, board):
        return tuple([1 if spot == 'X' else 0 if spot == ' ' else -1 for spot in board])
    
    def get_action(self, *args,**kwargs):pass

    def update(self, *args,**kwargs):pass

class Q_learning(Agent):
    def __init__(self, letter,alpha=0.25 , gamma=0.95,episilon=200):
        super().__init__(letter,alpha,gamma,episilon)
        self.q = {}

    def get_q(self, state, action):
        return self.q.get((state, action),0)

    def get_action(self, game ):
        state = self.convert_board_to_num_array(game.board)
        if random.randint(1,200) < self.epsilon - game.x_wins-game.tie:
            action = random.choice(game.available_moves())
        else:
            q_values = np.array([self.get_q(state, action) for action in game.available_moves()])
            action = game.available_moves()[np.argmax(q_values)]
        self.cur_board = state
        self.cur_move = action
        return action

    def update(self, game ):
        reward = 0
        if game.current_winner == self.letter:
            reward = 1
        elif game.current_winner == 'T':
            reward = -10
        elif game.current_winner and game.current_winner != 'T':
            reward = -10
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

class DQNAgent(Agent):
    def __init__(self, letter,model,lr=0.001 , gamma=0.95,episilon=200):
        super().__init__(letter,lr,gamma,episilon)
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()# the problem is regression like, so we use MSE
        self.memory = deque(maxlen=1000)
        self.epsilon = 200

    def get_action(self, game):
        self.cur_board = self.convert_board_to_num_array(game.board)
        if random.randint(1,200) < self.epsilon:
            action = random.choice(game.available_moves())
            cur_move = [0]*9
            cur_move[action] = 1
            self.cur_move = cur_move
        else:
            self.model.eval()
            q_values = self.model(self.list_to_tensor(self.cur_board))
            action = torch.argmax(q_values).item()
            self.cur_move = q_values.tolist()
        return action

    def train(self, x, q_modified):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(x)
        loss = self.criterion(output, q_modified)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def list_to_tensor(self, list):
        return torch.tensor(list).float()

    def update(self,game):
        cur_board = self.list_to_tensor(self.cur_board)
        next_board = self.list_to_tensor(self.convert_board_to_num_array(game.board))
        cur_action = self.list_to_tensor(self.cur_move)
        if game.current_winner == self.letter:
            reward = 1.0
        elif game.current_winner != None or game.current_winner == 'T':
            reward = -1.0
        else:
            reward = 0.0
        
        q_modified = cur_action.clone().detach() # to be trained on
        if not game.is_empty_squares():
            q = reward
        else:
            self.model.eval()
            q = reward + self.gamma * torch.max(self.model(next_board)).item() # found using the next state's best action.
        q_modified[torch.argmax(q_modified)] = q # give all legal moves the same q
        self.train(cur_board, q_modified)
        self.store(cur_board.tolist(), q_modified.tolist())
        if game.current_winner:
            self.experience_replay()


    def store(self, state, q_modified):
        self.memory.append((state, q_modified))

    def experience_replay(self):
        if len(self.memory) <= BATCH_SIZE:
            states, q_modified = zip(*self.memory)
        else:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
            states, q_modified = zip(*mini_sample)
        states = torch.tensor(states).float()
        q_modified = torch.tensor(q_modified).float()
        return self.train(states, q_modified)

class RandomAgent(Agent):
    def __init__(self, letter):
        super().__init__(letter,0,0,0)

    def get_action(self, game):
        return random.choice(game.available_moves())