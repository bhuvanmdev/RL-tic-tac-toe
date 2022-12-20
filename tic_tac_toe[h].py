def user_turn(table, first_time, chance):
    global userturn
    string_alp = "abcdefghi"
    string_num = "123456789"
    retry_try_except = True
    if first_time:
        return table
    elif not first_time:
        if chance == "X":
            while retry_try_except:
                try:
                    userturn = int(input("'X' turn: "))
                    while userturn not in range(1, 10):
                        userturn = int(input("'X' turn(ONLY 1-9 IDIOT!!): "))
                    retry_try_except = False
                except ValueError:
                    print("Type only numbers from 1-9")
        elif chance == "O":
            while retry_try_except:
                try:
                    userturn = int(input("'O' turn: "))
                    while userturn not in range(1, 10):
                        userturn = int(input("'O' turn(ONLY 1-9 IDIOT!!): "))
                        print("Type only numbers from 1-9(NO DECIMALS,APLHABETS OR WEIRD SYMBOLS)")
                    retry_try_except = False
                except ValueError:
                    print("Type only numbers from 1-9(NO DECIMALS,APLHABETS OR WEIRD SYMBOLS)")
        else:
            pass
        repeating_chance = table.find(string_alp[userturn - 1])
        for x in string_num:
            x = int(x)
            if x == userturn:
                if repeating_chance == -1:
                    return "\nits already done\n"
                else:
                    return table.replace(string_alp[x - 1], f"{chance}")
            else:
                continue



dict_win_pattern = [(0, 2, 4), (14, 16, 18), (28, 30, 32),
                    (0, 14, 28), (2, 16, 30), (4, 18, 32),
                    (0, 16, 32), (4, 16, 28)]

print("a=1|b=2|c=3|\n____________\nd=4|e=5|f=6|\n____________\ng=7|h=8|i=9|\n")
retry = True
table = "a|b|c|\n______\nd|e|f|\n______\ng|h|i|"
first_time = True
game_over = False

while retry:
    try:
        begin = int(input("'X' play first(1)\n\t\t\tOR\n'O' play first(0): "))
        while begin not in (0, 1):
            begin = int(input("'X' play first(1)\n\t\t\tOR\n'O' play first(0): "))
            print("press either one or zero")
        retry = False
    except ValueError:
        print("press either one or zero")

while table.count("X") + table.count("O") != 9 and not game_over:
    repeats_x = repeats_o = 0
    if first_time:
        table = user_turn(table=table, first_time=True, chance="69")
    if bool(begin):
        table_1 = user_turn(table=table, first_time=False, chance="X")
        if table_1 == "\nits already done\n":
            print(table_1)
            begin = 1
            first_time = False
        else:
            table = table_1
            print(table)
            begin = 0
            first_time = True

    elif not bool(begin):
        table_1 = user_turn(table=table, first_time=False, chance="O")
        if table_1 == "\nits already done\n":
            print(table_1)
            begin = 0
            first_time = False
        else:
            table = table_1
            print(table)
            begin = 1
            first_time = True


    for i in dict_win_pattern:
        repeats_o = repeats_x = 0
        for t in i:
            if "X" == table[t]:
                repeats_x += 1
                if repeats_x == 3:
                    print("\nX WINS\n")
                    break
            if "O" == table[t]:
                repeats_o += 1
                if repeats_o == 3:
                    print("\nO WINS\n")
                    break
        if repeats_x == 3 or repeats_o == 3:
            game_over = True
            break

print('THE GAME IS OVER')
