import random

import numpy as np
import pygame


class Board:
    def __init__(self, size=8):
        self.size = size
        self.board_matrix = np.zeros((self.size, self.size), dtype=np.int)
        self.valid_position = []
        self.last_position = None
        for i in range(size):
            for j in range(size):
                self.valid_position.append((i, j))

    def set_board_matrix(self, mat):
        self.board_matrix = mat.copy()

    def is_valid_position(self, x, y):
        return 0 <= x < self.size and 0 <= y < self.size

    def has_valid_position(self):
        return len(self.valid_position) > 0

    def get_valid_position(self):
        length = len(self.valid_position)
        random_index = random.randint(0, length - 1)
        return self.valid_position[random_index]

    def remove_valid_position(self, position):
        self.valid_position.remove(position)

    def has_neighbor(self, position: tuple, radius):
        x = position[0]
        y = position[1]
        board = self.board_matrix
        start_x, end_x = (x - radius), (x + radius)
        start_y, end_y = (y - radius), (y + radius)

        for i in range(start_y, end_y + 1):
            for j in range(start_x, end_x + 1):
                if 0 <= i < self.size and 0 <= j < self.size:
                    if board[i][j] == 1 or board[i][j] == -1:
                        return True
        return False

    def get_has_neighbour_valid_position_by_random(self):
        # 生成子节点的时候选择有棋子的位置
        has_neighbour_position_list = []

        for position in self.valid_position:
            if self.has_neighbor(position, 1):
                has_neighbour_position_list.append(position)

        random_index = random.randint(0, len(has_neighbour_position_list) - 1)
        return has_neighbour_position_list[random_index]

    def move_in_position(self, position: tuple, player_flag: int):
        self.board_matrix[position[0]][position[1]] = player_flag

    def is_win(self, position: tuple):
        """
        检查当前position下子后是否有胜者

        :param position: (x,y)
        :return: true false
        """
        board = self.board_matrix
        x, y = position[0], position[1]
        player = board[x][y]

        # 横向
        for i in range(self.size - 4):
            count = 0
            for k in range(5):
                if i + k < self.size and board[x][i + k] == player:
                    count += 1
            if count >= 5:
                return True
        # 纵向
        for i in range(self.size - 4):
            count = 0
            for k in range(5):
                if i + k < self.size and board[i + k][y] == player:
                    count += 1
            if count >= 5:
                return True

        # -45度方向
        l1 = [board.diagonal(y - x)]
        # if (l1 == player).sum() >= 5:
        #     return True
        if len(l1) >= 5:
            for i in range(len(l1) - 4):
                count = 0
                for k in range(5):
                    if l1[i + k] == player:
                        count += 1
                if count >= 5:
                    return True

        # +45度方向
        l2 = [np.fliplr(board).diagonal(self.size - 1 - y - x)]
        # if (l2 == player).sum() >= 5:
        #     return True
        # print(l2)
        if len(l2) >= 5:
            for i in range(len(l2) - 4):
                count = 0
                for k in range(5):
                    if l2[i + k] == player:
                        count += 1
                if count >= 5:
                    return True
        return False

    def reset(self):
        self.board_matrix = np.zeros((self.size, self.size), dtype=np.int)
        self.valid_position = [
            (i, j) for i in range(self.size) for j in range(self.size)
        ]
        self.last_position = None

    # def cal_score(self, x, y):
    #     shape_score = {(0, 1, 1, 0, 0): 50,
    #                    (0, 0, 1, 1, 0): 50,
    #                    (1, 1, 0, 1, 0): 200,
    #                    (0, 0, 1, 1, 1): 500,
    #                    (1, 1, 1, 0, 0): 500,
    #                    (0, 1, 1, 1, 0): 5000,
    #                    (0, 1, 0, 1, 1, 0): 5000,
    #                    (0, 1, 1, 0, 1, 0): 5000,
    #                    (1, 1, 1, 0, 1): 5000,
    #                    (1, 1, 0, 1, 1): 5000,
    #                    (1, 0, 1, 1, 1): 5000,
    #                    (1, 1, 1, 1, 0): 5000,
    #                    (0, 1, 1, 1, 1): 5000,
    #                    (0, 1, 1, 1, 1, 0): 50000,
    #                    (1, 1, 1, 1, 1): 999999}


class BoardViewController:
    def __init__(self, board: Board) -> None:
        self.board = board

    def show_view(self):
        self.screen = pygame.display.set_mode(
            (50 * self.board.size, 50 * self.board.size)
        )
        pygame.display.set_caption("Interface of Five-in-a-Row")
        self.draw_view()

    def draw_view(self):
        self.screen.fill((230, 185, 70))
        for x in range(self.board.size):
            pygame.draw.line(
                self.screen,
                [0, 0, 0],
                [25 + 50 * x, 25],
                [25 + 50 * x, self.board.size * 50 - 25],
                1,
            )
            pygame.draw.line(
                self.screen,
                [0, 0, 0],
                [25, 25 + 50 * x],
                [self.board.size * 50 - 25, 25 + 50 * x],
                1,
            )
        pygame.display.update()

    def update_view(self):
        mat = self.board.board_matrix
        indices = np.where(mat != 0)
        for (row, col) in list(zip(indices[0], indices[1])):
            if mat[row][col] == 1:
                pygame.draw.circle(
                    self.screen, [0, 0, 0], [25 + 50 * col, 25 + 50 * row], 15, 15
                )
            elif mat[row][col] == -1:
                pygame.draw.circle(
                    self.screen, [255, 255, 255], [25 + 50 * col, 25 + 50 * row], 15, 15
                )
        pygame.display.update()

    def move_in_position(self, player_flag, position):
        is_end, is_win, action_done = False, False, False
        if self.board.board_matrix[position[0]][position[1]] == 0:
            self.board.move_in_position(position=position, player_flag=player_flag)
            self.board.remove_valid_position(position)
            self.board.last_position = position
            action_done = True
        if action_done:
            is_win = self.board.is_win(position=position)
            is_end = not self.board.has_valid_position()
            if is_win:
                print("winer is:", player_flag)
            self.update_view()
        return action_done, (is_win | is_end)

    def reset(self):
        self.board.reset()
        self.draw_view()
