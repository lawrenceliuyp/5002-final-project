import time

import pygame

from board import BoardViewController
from mcts import MCTS


class Player:
    def __init__(self, mat_flag=0) -> None:
        self.mat_flag = mat_flag

    def set_board_controller(self, board_controller: BoardViewController):
        self.board_controller = board_controller

    def do_action(self, event=None):
        # stub
        # return is_win, action_done
        raise NotImplementedError


class HumanPlayer(Player):
    def do_action(self, event=None):
        """
        This function detects the mouse click on the game window. Update the state matrix of the game.
        input:
            event:pygame event, which are either quit or mouse click)
            mat: 2D matrix represents the state of the game
        output:
            mat: updated matrix
        """
        action_done, is_end = False, False
        if event.type == pygame.MOUSEBUTTONDOWN:
            (x, y) = event.pos
            # row = round((y - 40) / 40)
            # col = round((x - 40) / 40)
            row = round(abs(y - 25) / 50)
            col = round(abs(x - 25) / 50)
            action_done, is_end, = self.board_controller.move_in_position(
                position=(row, col), player_flag=self.mat_flag
            )
        return action_done, is_end


class AIPlayer(Player):
    def __init__(self, mat_flag=0, computational_power=50000) -> None:
        super().__init__(mat_flag=mat_flag)
        self.computational_power = computational_power

    def do_action(self, event=None):
        """
        This is the core of the game. Write your code to give the computer the intelligence to play a Five-in-a-Row game
        with a human
        input:
            2D matrix representing the state of the game.
        output:
            2D matrix representing the updated state of the game.
        """

        # ai suggestion
        board = self.board_controller.board
        start_time = time.time()
        first_position_in_MCTS = board.last_position
        mcts = MCTS(
            start_time,
            self.computational_power,
            first_position_in_MCTS,
            board,
        )
        suggested_position = mcts.monte_carlo_tree_search().position

        action_done, is_end = self.board_controller.move_in_position(
            position=suggested_position, player_flag=self.mat_flag
        )
        return action_done, is_end
