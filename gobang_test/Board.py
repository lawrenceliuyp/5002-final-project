from copy import deepcopy
from itertools import groupby

import numpy as np


class Board:
    def __init__(self, size=11):
        self.size = size # 设置棋盘大小（size*size），default size = 11
        self.chess = np.zeros((size, size), int)
        print(f'==> Board initializing:\n{self.chess}')
        self.update()
    
    def has_neighbor(self, position, radius):
            x = position[0]
            y = position[1]
            board = deepcopy(self.chess)
            start_x, end_x = (x - radius), (x + radius)
            start_y, end_y = (y - radius), (y + radius)

            for i in range(start_y, end_y + 1):
                for j in range(start_x, end_x + 1):
                    if 0 <= i < self.size and 0 <= j < self.size:
                        if board[i][j] == 1 or board[i][j] == -1:
                            return True
            return False

    def update(self):
        # 找到棋盘上还没有落子的点的坐标，e.g. [0,0], [1,1]
        self.vacuity = list(map(lambda x: tuple(x), np.argwhere(self.chess == 0)))
        # self.vacuity = []
        # tmp = list(map(lambda x: tuple(x), np.argwhere(self.chess == 0)))
        # for i in tmp:
        #     # x, y  = i[0], i [1]
        #     if self.has_neighbor(i, 1):
        #         self.vacuity.append(i)

        # print(f'{self.vacuity}')
        # print(self.vacuity)

    # def nbr(self, )

    def move(self, pos, player):
        self.chess[pos[0], pos[1]] = player # 在（pos[0], pos[1]）坐标的点落子
        self.update()

    def end(self, player):
        seq = list(self.chess) # 按行输出棋盘（用来判断行能不能赢）
        seq.extend(self.chess.transpose()) # 加入棋盘的转制（用来判断列能不能赢）
        fliplr = np.fliplr(self.chess) # 水平翻转棋盘
        for i in range(-self.size + 1, self.size):
            seq.append(self.chess.diagonal(i)) # 对角线
        for i in range(-self.size + 1, self.size):
            seq.append(fliplr.diagonal(i)) # 翻转后的对角线
        for seq in map(groupby, seq): # 判断输赢
            for v, i in seq:
                if v == 0: continue
                if v == player and len(list(i)) == 5:
                    return v
        return 0

    def defend(self): # 根据棋盘现在的状态，从未落子的点中选点进行防御
        for x, y in self.vacuity:
            origin = map(groupby, [
                self.chess[x],
                self.chess.transpose()[y],
                self.chess.diagonal(y - x),
                np.fliplr(self.chess).diagonal(self.size - 1 - y - x)
            ])
            origin = [x for x in origin]
            chess = deepcopy(self.chess)
            chess[x][y] = -1
            for index, seq in enumerate(
                    map(groupby, [
                        chess[x],
                        chess.transpose()[y],
                        chess.diagonal(y - x),
                        np.fliplr(chess).diagonal(self.size - 1 - y - x)
                    ])):
                seq = [(v, len(list(i))) for v, i in seq]
                org_seq = [(v, len(list(i))) for v, i in origin[index]]
                for i, v in enumerate(seq):
                    if v[0] != -1: continue
                    if v[1] >= 5: return x, y
                    if v[1] == 4 and seq.count((-1, 4)) != org_seq.count((-1, 4)):
                        if i - 1 >= 0 and seq[i - 1][0] == 0 and i + 1 < len(seq) and seq[i + 1][0] == 0: return x, y
        return None


if __name__ == "__main__":
    Board()
