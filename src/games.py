import tkinter
import tkinter.messagebox

import pygame

from board import Board, BoardViewController
from players import AIPlayer, HumanPlayer


# interface function for competition
def update_by_pc(mat, mat_flag=1):
    """
    This is the core of the game. Write your code to give the computer the intelligence to play a Five-in-a-Row game
    with a human
    input:
        2D matrix representing the state of the game.
    output:
        2D matrix representing the updated state of the game.
    """
    board = Board(size=max(mat.shape))
    board.set_board_matrix(mat)
    pos = AIPlayer.get_ai_solution(board)
    if pos:
        mat[pos[0], pos[1]] = mat_flag
    return mat


class Game:
    PLAYER_BLACK = 1
    PLAYER_WHITE = -1

    def __init__(self, board_size=15):
        self.dialog = tkinter.Tk()
        self.dialog.withdraw()

        pygame.init()
        pygame.key.set_repeat(10, 15)

        # init board view and controller
        self.board = Board(board_size)

        # init view
        self.board_controller = BoardViewController(self.board)

        # init players
        self.current_player = HumanPlayer(mat_flag=1)
        self.current_player.set_board_controller(self.board_controller)
        self.board.player = self.current_player.mat_flag

        # init ai player
        self.ai_player_1 = AIPlayer(mat_flag=1, computational_power=1000)
        self.ai_player_1.set_board_controller(self.board_controller)

        self.ai_player = AIPlayer(mat_flag=-1, computational_power=1000)
        self.ai_player.set_board_controller(self.board_controller)

        # init game parameters
        self.turn = 0
        self.done = False

    def get_player_name(self, turn):
        return "black" if turn % 2 else "white"

    def start_looper(self):
        while not self.done:
            # event = pygame.event.wait()
            # if event.type == pygame.QUIT:
            #     self.done = True
            #     exit()
            # if event.type in [pygame.MOUSEBUTTONDOWN] and not (self.turn % 2):
            ###it is player 1 turn
            # action_done, self.done = self.current_player.do_action(event)
            # if not action_done:
            #     continue
            _, self.done = self.ai_player_1.do_action()
            self.turn += 1
            ###it is player 1 turn end

            ### it is ai player turn
            _, self.done = self.ai_player.do_action()
            self.turn += 1
            ###it is player 2 turn end

            if self.done:
                self.show_endgame_dialog()
            pygame.display.update()

    def reset(self):
        self.board_controller.reset()
        self.turn = 0
        self.done = False

    def start(self):
        self.board_controller.show_view()
        self.start_looper()

    def show_endgame_dialog(self):
        retry = tkinter.messagebox.askretrycancel(
            "Game Over",
            f"Do you want to try again?",
        )
        if retry:
            self.reset()
        else:
            exit()
        self.dialog.lift()


if __name__ == "__main__":
    game = Game()
    game.start()
