import time

import numpy as np

from board import Board
from minimax import MCTS

bd = Board(8)

bd_mat = bd.board_matrix

bd_mat[4][4] = -1
bd_mat[5][4] = -1
bd_mat[6][4] = -1
start_time = time.time()

mcts = MCTS(start_time, 50000, (4, 4), bd, 8)

sug_pos = mcts.monte_carlo_tree_search_alpha()
print(sug_pos)


print(tuple([-1, -1]))
