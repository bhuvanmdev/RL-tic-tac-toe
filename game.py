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
