import time

# import group3 as g3  # ## import second model,put the file under the same folder
# import group12 as g12  # ## import first model,put the file under the same folder
import numpy as np
import pygame


class Player:
    def __init__(self, mat_flag=0) -> None:
        self.mat_flag = mat_flag

    def update_mat(self, event, mat):
        raise NotImplementedError


class HumanPlayer(Player):
    def update_by_man(self, event, mat):
        """
        This function detects the mouse click on the game window. Update the state matrix of the game.
        input:
            event:pygame event, which are either quit or mouse click)
            mat: 2D matrix represents the state of the game
        output:
            mat: updated matrix
        """
        step_done = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            (x, y) = event.pos
            # row = round((y - 40) / 40)
            # col = round((x - 40) / 40)
            row = round((y - 40) / 40)
            col = round((x - 40) / 40)
            if mat[row][col] == 0:
                mat[row][col] = self.mat_flag
                step_done = True
        return mat, step_done

    def update_mat(self, event, mat):
        return self.update_by_man(event, mat)


class AIPlayer(Player):
    def __init__(self, strategy=None, mcts_model=None) -> None:
        pass

    def update_by_pc(self, mat):
        """
        This is the core of the game. Write your code to give the computer the intelligence to play a Five-in-a-Row game
        with a human
        input:
            2D matrix representing the state of the game.
        output:
            2D matrix representing the updated state of the game.
        """
        return mat

    def update_mat(self, event, mat):
        return self.update_by_pc(mat), True


def draw_board(screen, size):
    screen.fill((230, 185, 70))
    for x in range(size):
        pygame.draw.line(
            screen, [0, 0, 0], [25 + 50 * x, 25], [25 + 50 * x, size * 50 - 25], 1
        )
        pygame.draw.line(
            screen, [0, 0, 0], [25, 25 + 50 * x], [size * 50 - 25, 25 + 50 * x], 1
        )
    pygame.display.update()


def update_board(screen, state):
    indices = np.where(state != 0)
    for (row, col) in list(zip(indices[0], indices[1])):
        if state[row][col] == 1:
            pygame.draw.circle(
                screen, [0, 0, 0], [25 + 50 * col, 25 + 50 * row], 15, 15
            )
        elif state[row][col] == -1:
            pygame.draw.circle(
                screen, [255, 255, 255], [25 + 50 * col, 25 + 50 * row], 15, 15
            )
    pygame.display.update()


def draw_stone(screen, mat):
    """
    This functions draws the stones according to the mat. It draws a black circle for matrix element 1(human),
    it draws a white circle for matrix element -1 (computer)
    input:
        screen: game window, onto which the stones are drawn
        mat: 2D matrix representing the game state
    output:
        none
    """
    black_color = [0, 0, 0]
    white_color = [255, 255, 255]
    # M=len(mat)
    d = int(560 / (M - 1))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i][j] == 1:
                pos = [40 + d * j, 40 + d * i]
                pygame.draw.circle(screen, black_color, pos, 18, 0)
            elif mat[i][j] == -1:
                pos = [40 + d * j, 40 + d * i]
                pygame.draw.circle(screen, white_color, pos, 18, 0)


def check_for_done(mat):
    """
    please write your own code testing if the game is over. Return a boolean variable done. If one of the players wins
    or the tie happens, return True. Otherwise return False. Print a message about the result of the game.
    input:
        2D matrix representing the state of the game
    output:
        True,1:Black win the game
        True,0:Draw
        True,-1:White win the game
        False,0:Not complete
    """
    done = False
    result = 0

    a = np.zeros((5, 5), int)
    np.fill_diagonal(a, 1)
    a = np.zeros((5, 5), int)
    b = np.flip(a.copy())
    c = np.ones((1, 5), int)
    d = np.ones((5, 1), int)

    return done, result


def main():
    global M
    M = 8
    pygame.init()
    screen = pygame.display.set_mode((50 * M, 50 * M))
    pygame.display.set_caption("Interface of Five-in-a-Row")
    draw_board(screen, M)

    done = False
    mat = np.zeros((M, M), int)
    pygame.display.update()
    # mat[int(M/2)][int(M/2)]=1
    update_board(screen, mat)

    # add players

    p1 = HumanPlayer(mat_flag=-1)
    p2 = HumanPlayer(mat_flag=1)
    players = [p1, p2]
    current_player = players[0]
    turn = 0

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

            if event.type in [pygame.MOUSEBUTTONDOWN]:
                current_player = players[turn % 2]

                ###it is player 1 turn
                mat, step_done = current_player.update_mat(event, mat)
                if not step_done:
                    continue
                ###
                update_board(screen, mat)
                print("mat=\n", mat)
                turn += 1

                # check for win or tie
                done, result = check_for_done(mat)
                if done:
                    print("winer is:", result)
                    break
                ###it is player 1 turn end

                next_player = players[turn % 2]
                if isinstance(next_player, HumanPlayer):
                    continue

                ### it is ai player turn
                mat, _ = next_player.update_mat(
                    event, -1 * mat
                )  # transform to the result for player 2 to put white stone
                ###
                mat = mat * (-1)
                update_board(screen, mat)
                print("now mat in\n", mat)
                turn += 1

                # check for win or tie
                done, result = check_for_done(mat)
                if done:
                    print("winer is:", result)
                    break
                ###it is player 2 turn end


if __name__ == "__main__":
    main()
