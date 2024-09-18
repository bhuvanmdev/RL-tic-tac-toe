import random
import time
import numpy as np
import torch
random.seed(42)
class TicTacToe:
    def __init__(self):
        self.board = [' ' for _ in range(9)]
        self.current_winner = None
        self.x_wins = 0
        self.o_wins = 0
        self.tie = 0
        self.prev_game= []
        self.prev_starter = ''

    def print_board(self,board=None):
        if board is None:
            board = self.board
        for row in [board[i*3:(i+1)*3] for i in range(3)]:
            print('| ' + ' | '.join(row) + ' |')
        print('\n')

    def available_moves(self):
        return [i for i, spot in enumerate(self.board) if spot == ' ']

    def is_empty_squares(self):
        return ' ' in self.board

    def count_empty_squares(self):
        return self.board.count(' ')

    def make_move(self, square, letter):
        if self.board[square] == ' ':
            self.board[square] = letter
            if self._winner(square, letter):
                self.current_winner = letter
            return True
        return False

    def _winner(self, square, letter):
        row_ind = square // 3
        row = self.board[row_ind*3:(row_ind+1)*3]
        if all([spot == letter for spot in row]):
            return True
        
        col_ind = square % 3
        column = [self.board[col_ind+i*3] for i in range(3)]
        if all([spot == letter for spot in column]):
            return True
    
        if square % 2 == 0:
            diagonal1 = [self.board[i] for i in [0, 4, 8]]
            if all([spot == letter for spot in diagonal1]):
                return True
            diagonal2 = [self.board[i] for i in [2, 4, 6]]
            if all([spot == letter for spot in diagonal2]):
                return True
        return False
    
    def reset(self):
        self.board = [' ' for _ in range(9)]
        self.current_winner = None
        self.prev_game = []
        self.prev_starter = ''
    
    def play_game(self,letter,agent=None, print_game=False,store=False):
        if store:
            self.prev_starter = letter
        while self.is_empty_squares():
            # square = input(f"{letter}'s turn: ")
            # square = move#random.choice(self.available_moves())
            if agent:
                square = agent.move(self,letter)
            else:
                square = random.choice(self.available_moves())
            if not self.make_move(square, letter):
                # print('Invalid move!')
                continue
            if print_game:
                self.print_board()
            if agent:
                agent.update(self)
            if store:
                self.prev_game.append(self.board.copy())
            if self.current_winner:
                if self.current_winner == 'X':
                    self.x_wins += 1
                elif self.current_winner == 'O':
                    self.o_wins += 1
                # print(f'{self.current_winner} wins!')
                    if store:
                        print('o wins')
                        [self.print_board(x) for x in self.prev_game]
                return letter
            letter = 'O' if letter == 'X' else 'X'
        if print_game:pass
            # print('It\'s a tie!')\
        if store:
            print('tie game')
            [self.print_board(x) for x in self.prev_game]
        self.tie += 1
        return 'T'
    
    def prev_game_store(self):
        self.perv_game.append(self.board)


class Q_learning:
    def __init__(self, epsilon=0, alpha=0.3 , gamma=0.9):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.q = {}
        self.cur_board = None
        self.cur_move = None
    def convert_board_to_num_array(self, board):
        return tuple([1 if spot == 'X' else 0 if spot == ' ' else -1 for spot in board])

    def get_q(self, state, action):
        return self.q.get((state, action),0)

    def move(self, game: TicTacToe,letter=None):
        state = self.convert_board_to_num_array(game.board)
        if random.random() < self.epsilon or letter == 'O':
            action = random.choice(game.available_moves())
        else:
            q_values = np.array([self.get_q(state, action) for action in game.available_moves()])
            action = game.available_moves()[np.argmax(q_values)]
        self.cur_board = state
        self.cur_move = action
        return action

    def update(self, game):
        reward = 0
        if game.current_winner == 'X':
            reward = 2
        elif game.current_winner == 'O':
            reward = -3
        elif not game.is_empty_squares():
            reward = 1
        next_state = self.convert_board_to_num_array(game.board.copy())
        q_values = np.array([self.get_q(next_state, action) for action in game.available_moves()])
        next_max_q = np.max(q_values) if len(q_values) > 0 else 0
        new_q = self.get_q(self.cur_board, self.cur_move) + self.alpha * (reward + self.gamma * next_max_q - self.get_q(self.cur_board, self.cur_move))
        self.q[(self.cur_board, self.cur_move)] = new_q

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
    game = TicTacToe()
    q = Q_learning()
    # q.load_q(r"E:\re_inforcement_learning\q.pt")
    for _ in range(1000):
        letter = random.choice(['X', 'O'])
        # print(f'{letter} goes first')
        game.play_game(letter,q)
        # time.sleep(1)
        game.reset()
    # q.save_q(r"E:\re_inforcement_learning\q.pt")
    print(game.x_wins, game.o_wins,game.tie)
