import random
import time
import numpy as np
import torch
from matplotlib import pyplot as plt
from IPython import display

random.seed(42)
class TicTacToe:
    def __init__(self,size=3):
        self.board = [' ' for _ in range(size*size)]
        self.current_winner = None
        self.x_wins = 0
        self.o_wins = 0
        self.tie = 0
        self.prev_game= []
        self.size = size

    def print_board(self,board=None):
        if board is None:
            board = self.board
        for row in [board[i*self.size:(i+1)*self.size] for i in range(self.size)]:
            print('| ' + ' | '.join(row) + ' |')
        print('\n')

    def available_moves(self):
        return [i for i, spot in enumerate(self.board) if spot == ' ']

    def is_empty_squares(self):
        return self.available_moves().__len__() != 0

    def count_empty_squares(self):
        return self.board.count(' ')

    def make_move(self, square, letter):
        if self.board[square] == ' ':
            self.board[square] = letter
            if self._winner(square, letter):
                self.current_winner = letter
            elif self.count_empty_squares() == 0:
                self.current_winner = 'T'
            return True
        
        return False

    def _winner(self, square, letter):
        row_ind = square // self.size
        row = self.board[row_ind*self.size:(row_ind+1)*self.size]
        if all([spot == letter for spot in row]):
            return True
        
        col_ind = square % self.size
        column = [self.board[col_ind+i*self.size] for i in range(self.size)]
        if all([spot == letter for spot in column]):
            return True
    
        if square % 2 == 0:
            diagonal1 = [self.board[i] for i in [4*x for x in range(self.size)]]
            if all([spot == letter for spot in diagonal1]):
                return True
            diagonal2 = [self.board[i] for i in [2*(x+1) for x in range(self.size)]]
            if all([spot == letter for spot in diagonal2]):
                return True
        return False
    
    def reset(self):
        self.board = [' ' for _ in range(self.size*self.size)]
        self.current_winner = None
        # self.prev_game = []
    

    def prev_game_store(self):
        self.prev_game.append(str(self.board.copy()))


class Q_learning:
    def __init__(self, letter, rand=False,alpha=0.3 , gamma=0.9):
        self.alpha = alpha
        self.gamma = gamma
        self.q = {}
        self.cur_board = None
        self.cur_move = None
        self.letter = letter
        self.epsilon = 0.9
        self.rand = rand

    def convert_board_to_num_array(self, board):
        return tuple([1 if spot == 'X' else 0 if spot == ' ' else -1 for spot in board])

    def get_q(self, state, action):
        return self.q.get((state, action),0)

    def move(self, game: TicTacToe):
        state = self.convert_board_to_num_array(game.board)
        if random.random() < self.epsilon -(10-game.size**0.5)*(game.x_wins + game.o_wins + game.tie) or self.rand:
            action = random.choice(game.available_moves())
        else:
            q_values = np.array([self.get_q(state, action) for action in game.available_moves()])
            action = game.available_moves()[np.argmax(q_values)]
        self.cur_board = state
        self.cur_move = action
        return action

    def update_q(self, game: TicTacToe):
        reward = 0
        if self.rand:return
        if game.current_winner == self.letter:
            reward = 5
            # game.x_wins += 1 if self.letter == 'X' else 0
            # game.o_wins += 1 if self.letter == 'O' else 0
        elif game.current_winner == 'T':
            reward = -2.5
            # game.tie += 1
        elif game.current_winner and game.current_winner != 'T':
            reward = -5
            # game.x_wins += 0 if self.letter == 'X' else 1
            # game.o_wins += 0 if self.letter == 'O' else 1
        new_state = self.convert_board_to_num_array(game.board.copy())
        if game.available_moves().__len__() != 0:
            q_values = np.array([self.get_q(new_state, action) for action in game.available_moves()])
            # print(q_values,game.available_moves())
            new_q = reward + self.gamma * np.max(q_values) # missed - self.get_q(self.cur_board, self.cur_move)
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


#steps to use a q-learning agent
#1. create a game object
#2. create a q-learning object
#3. play the game
#4. update the q-learning object

if __name__ == '__main__':
    game = TicTacToe(5)
    plt.ion()
    X_agent = Q_learning('X')
    O_agent = Q_learning('O')
    O_agent.rand = True
    x_score = []
    o_score = []
    tie_score = []
    record = [0,0,0]
    for i in range(50000):
        first = random.choice([X_agent, O_agent])
        second = X_agent if first == O_agent else O_agent
        game.reset()
        while True:
            while True:
                actionfirst = first.move(game)
                if game.make_move(actionfirst, first.letter):break
            second.update_q(game)
            if game.current_winner is not None:
                game.x_wins += 1 if game.current_winner == 'X' else 0
                game.o_wins += 1 if game.current_winner == 'O' else 0
                game.tie += 1 if game.current_winner == 'T' else 0
                first.update_q(game)
                break
            while True:    
                secondaction = second.move(game)
                if game.make_move(secondaction, second.letter):break
            first.update_q(game)
            if game.current_winner is not None:
                game.x_wins += 1 if game.current_winner == 'X' else 0
                game.o_wins += 1 if game.current_winner == 'O' else 0
                game.tie += 1 if game.current_winner == 'T' else 0
                second.update_q(game)
                break
        game.prev_game_store()
        record = [max(record[0],game.x_wins/(game.o_wins+game.tie+game.x_wins)),max(record[1],game.tie/(game.o_wins+game.tie+game.x_wins)),max(record[2],game.o_wins/(game.o_wins+game.tie+game.x_wins))]
        x_score.append(game.x_wins/(game.o_wins+game.tie+game.x_wins))
        o_score.append(game.o_wins/(game.o_wins+game.tie+game.x_wins))
        tie_score.append(game.tie/(game.o_wins+game.tie+game.x_wins))
        display.clear_output(wait=True)
        display.display(plt.gcf())
        plt.clf()
        plt.title('Training...')
        plt.xlabel('Number of Games')
        plt.ylabel('Score')
        plt.plot(x_score)
        plt.plot(o_score)
        plt.plot(tie_score)
        plt.legend(['X','O','Tie'])
        plt.ylim(ymin=0)
        plt.show(block=False)
        #exit when q is pressed
        if plt.waitforbuttonpress(0.000001):
            break



    print(record,set(game.prev_game).__len__())

    



