import random
from matplotlib import pyplot as plt
from game import TicTacToe
from agents import DQNAgent, Q_learning
import torch
from models import CNN_Model,DNN
random.seed(42)
torch.manual_seed(42)

# more negative reward is better than more positive reward
#steps to use a q-learning agent
#1. create a game object
#2. create a q-learning object
#3. play the game
#4. update the q-learning object

if __name__ == '__main__':
    board_size = 3
    game = TicTacToe(board_size)
    plt.ion()
    X_agent = DQNAgent('X',CNN_Model(board_size))
    # X_agent.load_q(r"E:\re_inforcement_learning\x.pth")
    O_agent = DQNAgent('O',DNN(board_size))#Q_learning('O')
    # O_agent.rand = True # make it none to user playable and True for random agent
    x_score = []    
    o_score = []
    tie_score = []
    episodes = 100_000
    for i in range(episodes):
        first = [X_agent, O_agent][(i+0)%2] #uncomment this line to make the game random
        # first = X_agent #uncomment this line to make the first player deterministic
        second = X_agent if first == O_agent else O_agent
        game.reset()
        temp = 0
        while True:
            while True:
                actionfirst = first.get_action(game)
                if game.make_move(actionfirst, first.letter):break
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
            while True:    
                secondaction = second.get_action(game)
                if game.make_move(secondaction, second.letter):break
            first.update(game)
            if game.current_winner is not None:
                game.x_wins += 1 if game.current_winner == 'X' else 0
                game.o_wins += 1 if game.current_winner == 'O' else 0
                game.tie += 1 if game.current_winner == 'T' else 0
                second.update(game)
                break
        x_score.append(game.x_wins/(game.o_wins+game.tie+game.x_wins))
        o_score.append(game.o_wins/(game.o_wins+game.tie+game.x_wins))
        tie_score.append(game.tie/(game.o_wins+game.tie+game.x_wins))
        if (i+1)%100 == 0:
            print(f"i={i}",game.x_wins/(game.o_wins+game.tie+game.x_wins),game.o_wins/(game.o_wins+game.tie+game.x_wins),game.tie/(game.o_wins+game.tie+game.x_wins))


plt.title('Trained')
plt.xlabel('Number of Games')
plt.ylabel('Score %')
plt.plot(x_score)
plt.plot(o_score)
plt.plot(tie_score)
plt.legend([f'X-{(game.x_wins/(game.x_wins+game.tie+game.o_wins))*100}',f'O-{(game.o_wins/(game.o_wins+game.tie+game.x_wins))*100}',f'Tie-{(game.tie/(game.o_wins+game.tie+game.x_wins))*100}'])
plt.ylim(ymin=0)
plt.show(block=True)

if plt.waitforbuttonpress(0.000001):pass


# X_agent.save_q(r"E:\re_inforcement_learning\x.pth")
# O_agent.save_q(r"E:\re_inforcement_learning\o.pth")
# print(game.x_wins/(game.o_wins+game.tie+game.x_wins),game.o_wins/(game.o_wins+game.tie+game.x_wins),game.tie/(game.o_wins+game.tie+game.x_wins),len(X_agent.q))#,set(game.prev_game).__len__())

    



