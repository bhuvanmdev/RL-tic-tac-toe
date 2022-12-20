import random, time


def user_turn(table, first_time, chance):
    global userturn
    string_alp = "abcdefghi"
    string_num = "123456789"
    retry_try_except = True
    if first_time:
        return table
    elif not first_time:
        if chance == "X":
            print("\n")
            while retry_try_except:
                try:
                    input_h_ai = int(input("Your turn: "))
                    while input_h_ai not in range(1, 10):
                        input_h_ai = int(input("'X' turn(ONLY 1-9 IDIOT!!): "))
                    retry_try_except = False
                except ValueError:
                    print("Type only numbers from 1-9")
            repeating_chance = table.find(string_alp[input_h_ai - 1])
        elif chance == "O":
            print("\nMy turn")
            time.sleep(1)
            input_h_ai = random.choice(range(1, 10))
            repeating_chance = table.find(string_alp[input_h_ai - 1])
            while repeating_chance == -1:
                input_h_ai = random.choice(range(1, 10))
                repeating_chance = table.find(string_alp[input_h_ai - 1])
        for x in string_num:
            x = int(x)
            if x == input_h_ai:
                if repeating_chance == -1:
                    return "\nits already done\n"
                else:
                    return table.replace(string_alp[x - 1], f"{chance}")
            else:
                continue


print("a=1|b=2|c=3|\n____________\nd=4|e=5|f=6|\n____________\ng=7|h=8|i=9|\n")
retry = True
table = "a|b|c|\n______\nd|e|f|\n______\ng|h|i|"
correct_response = True
list_win_pattern = [(0, 2, 4), (14, 16, 18), (28, 30, 32),
                    (0, 14, 28), (2, 16, 30), (4, 18, 32),
                    (0, 16, 32), (4, 16, 28)]
game_over = False


while retry:
    try:
        begin = int(input("You(X) play first(1)\n\t\t\tOR\nI(O) play first(0): "))
        while begin not in (0, 1):
            begin = int(input("You(X) play first(1)\n\t\t\tOR\nI(O) play first(0): "))
            print("\npress either one or Zero")
        retry = False
    except ValueError:
        print("\npress either one or Zero")


while table.count("X")+table.count("O") != 9 and not game_over:
    repeats_x = repeats_o = 0
    if correct_response:
        table = user_turn(table=table, first_time=True, chance="69")
    if bool(begin):
        table_1 = user_turn(table=table, first_time=False, chance="X")
        if table_1 == "\nits already done\n":
            print(table_1)
            begin = 1
            correct_response = False
        else:
            table = table_1
            print(table)
            begin = 0
            correct_response = True

    elif not bool(begin):
        table = user_turn(table=table, first_time=False, chance="O")
        print(table)
        begin = 1
        correct_response = True

    for i in list_win_pattern:
        repeats_o = repeats_x = 0
        for t in i:
            if "X" == table[t]:
                repeats_x += 1
                if repeats_x == 3:
                    print("\nYOU WIN\n")
                    break
            if "O" == table[t]:
                repeats_o += 1
                if repeats_o == 3:
                    print("\nAi WINS\n")
                    break
        if repeats_x == 3 or repeats_o == 3:
            game_over = True
            break

print('THE GAME IS OVER')