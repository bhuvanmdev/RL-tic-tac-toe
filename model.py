import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
import numpy as np
from game import TicTacToe
random.seed(42)
torch.manual_seed(42)

BATCH_SIZE = 128

class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(9, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 9)

    def forward(self, inp):
        # inp_clone = inp.clone().detach()
        x = F.relu(self.fc1(inp))
        x = F.relu(self.fc2(x))
        # print(F.relu(self.fc3(x))*torch.where(inp==0.0,1.0,0.0))
        x = self.fc3(x)
        x[(inp != 0.0).nonzero(as_tuple=True)] = float('-inf')
        # print(x,(inp_clone != 0.0).nonzero(as_tuple=True),inp)
        x = F.softmax(x,dim=-1)
        return x

    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        self.load_state_dict(torch.load(path))

class DQNAgent:
    def __init__(self, letter):
        self.model = DNN()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        self.memory = deque(maxlen=1000)
        self.cur_board = None
        self.cur_move = None
        self.letter = letter

    def convert_board_to_num_array(self, board):
        return tuple([1 if spot == 'X' else 0 if spot == ' ' else -1 for spot in board])

    def get_action(self, game):
        self.model.eval()
        self.cur_board = self.convert_board_to_num_array(game.board)
        q_values = self.model(self.list_to_tensor(self.cur_board))
        action = torch.argmax(q_values).item()
        # print(q_values)
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
        if game.current_winner == 'X':
            reward = 1.0
        elif game.current_winner == 'O' or game.current_winner == 'T':
            reward = -1.0
        else:
            reward = 0.0
        
        q_modified = cur_action.clone().detach() # to be trained on
        if not game.is_empty_squares():
            q = reward
        else:
            self.model.eval()
            q = reward + 0.9 * torch.max(self.model(next_board)).item() # found using the next state's best action.
        q_modified[torch.argmax(q_modified)] = q # give all legal moves the same q
        self.train(cur_board, q_modified)
        self.store(cur_board.tolist(), q_modified.tolist())
        if game.current_winner:
            self.experience_replay()

        ## todo

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
        
            

        # batch dat
if __name__ == '__main__':
    game = TicTacToe(3)
    # plt.ion()
    X_agent = DQNAgent('X')
    # X_agent.load_q(r"E:\re_inforcement_learning\x.pth")
    O_agent = DQNAgent('O')
    # O_agent.rand = True # make it none to user playable and True for random agent
    x_score = []    
    o_score = []
    tie_score = []
    episodes = 2_00
    for i in range(episodes):
        first = [X_agent, O_agent][(i+0)%2] #uncomment this line to make the game random
        # first = X_agent #uncomment this line to make the first player deterministic
        second = X_agent if first == O_agent else O_agent
        game.reset()
        # print(i,"first",first.letter)
        temp = 0
        while True:
            while True:
                actionfirst = first.get_action(game)
                if game.make_move(actionfirst, first.letter):break
                else:print(0000000000000000) 
                # print(1111111111111111)
            if temp:
                second.update(game)
            else:
                temp = 1 
            if game.current_winner is not None:
                game.x_wins += 1 if game.current_winner == 'X' else 0
                game.o_wins += 1 if game.current_winner == 'O' else 0
                game.tie += 1 if game.current_winner == 'T' else 0
                first.update(game)
                break
            # game.print_board()
            # print(789)
            
            while True:    
                secondaction = second.get_action(game)
                # print(secondaction)
                if game.make_move(secondaction, second.letter):break
                else:print(123456789)
                # print(222222222222222222)
            # print(333333333333333333)
            first.update(game)
            if game.current_winner is not None:
                game.x_wins += 1 if game.current_winner == 'X' else 0
                game.o_wins += 1 if game.current_winner == 'O' else 0
                game.tie += 1 if game.current_winner == 'T' else 0
                second.update(game)
                break
            # game.print_board()

        if i%100 == 0:
            print(f"i={i}",game.x_wins/(game.o_wins+game.tie+game.x_wins),game.o_wins/(game.o_wins+game.tie+game.x_wins),game.tie/(game.o_wins+game.tie+game.x_wins))
else:pass
    # print(DNN()(torch.tensor([-1,1,0,0,0,-1,1,0,1]).float()))
    # print(torch.tensor([1,1,1,1,1,1,1,1,1]).float()*torch.tensor([0,1,1,1,0,1,1,1,1]).float())