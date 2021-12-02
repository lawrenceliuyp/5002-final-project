import pygame

from board import Board, BoardViewController
from players import AIPlayer, HumanPlayer


class Game:
    def __init__(self, board_size=8):
        pygame.init()

        # init board view and controller
        self.board = Board(board_size)

        # init view
        self.board_controller = BoardViewController(self.board)

        # init players
        self.current_player = HumanPlayer(mat_flag=1)
        self.current_player.set_board_controller(self.board_controller)
        self.board.player = self.current_player.mat_flag

        # init ai player
        self.ai_player = AIPlayer(mat_flag=-1, computational_power=1000)
        self.ai_player.set_board_controller(self.board_controller)

        # init game parameters
        self.turn = 0
        self.done = False

    def start_looper(self):
        while not self.done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.done = True
                if event.type in [pygame.MOUSEBUTTONDOWN]:
                    ###it is player 1 turn
                    action_done, self.done = self.current_player.do_action(event)
                    if not action_done:
                        continue
                    if self.done:
                        break
                    ###it is player 1 turn end

                    ### it is ai player turn
                    _, self.done = self.ai_player.do_action()
                    if self.done:
                        break
                    ###it is player 2 turn end

    def reset(self):
        self.turn = 0
        self.done = False

    def start(self):
        self.board_controller.show_view()
        self.start_looper()


if __name__ == "__main__":
    game = Game()
    game.start()
