import random
from matplotlib import pyplot as plt
from IPython import display
from game import TicTacToe
from qtagent import Q_learning
random.seed(42)

# more negative reward is better than more positive reward
#steps to use a q-learning agent
#1. create a game object
#2. create a q-learning object
#3. play the game
#4. update the q-learning object

if __name__ == '__main__':
    game = TicTacToe(3)
    # plt.ion()
    X_agent = Q_learning('X')
    # X_agent.load_q(r"E:\re_inforcement_learning\x.pth")
    O_agent = Q_learning('O')
    O_agent.rand = True # make it none to user playable and True for random agent
    x_score = []    
    o_score = []
    tie_score = []
    episodes = 5_00_000
    for i in range(episodes):
        # first = random.choice([X_agent, O_agent]) #uncomment this line to make the game random
        first = X_agent #uncomment this line to make the first player deterministic
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
            # game.print_board()
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
            # game.print_board()

        if i%10000 == 0:
            print(f"i={i}",game.x_wins/(game.o_wins+game.tie+game.x_wins),game.o_wins/(game.o_wins+game.tie+game.x_wins),game.tie/(game.o_wins+game.tie+game.x_wins))
        # game.prev_game_store()
        # record = [max(record[0],game.x_wins/(game.o_wins+game.tie+game.x_wins)),max(record[1],game.tie/(game.o_wins+game.tie+game.x_wins)),max(record[2],game.o_wins/(game.o_wins+game.tie+game.x_wins))]
        x_score.append(game.x_wins/(game.o_wins+game.tie+game.x_wins))
        o_score.append(game.o_wins/(game.o_wins+game.tie+game.x_wins))
        tie_score.append(game.tie/(game.o_wins+game.tie+game.x_wins))
        # display.clear_output(wait=True)
        # plt.clf()
    plt.title('Trained')
    plt.xlabel('Number of Games')
    plt.ylabel('Score %')
    plt.plot(x_score)
    plt.plot(o_score)
    plt.plot(tie_score)
    plt.legend([f'X-{(game.x_wins/(game.x_wins+game.tie+game.o_wins))*100}',f'O-{(game.o_wins/(game.o_wins+game.tie+game.x_wins))*100}',f'Tie-{(game.tie/(game.o_wins+game.tie+game.x_wins))*100}'])
    plt.ylim(ymin=0)
    plt.show()
        # #exit when q is pressed
        # if plt.waitforbuttonpress(0.000001):
        #     break


    # X_agent.save_q(r"E:\re_inforcement_learning\x.pth")
    # O_agent.save_q(r"E:\re_inforcement_learning\o.pth")
    # print(game.x_wins/(game.o_wins+game.tie+game.x_wins),game.o_wins/(game.o_wins+game.tie+game.x_wins),game.tie/(game.o_wins+game.tie+game.x_wins),len(X_agent.q))#,set(game.prev_game).__len__())

    



