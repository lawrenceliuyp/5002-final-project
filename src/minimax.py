import copy
import time
from enum import IntEnum
from random import randint

import numpy as np

from board import Board

AI_SEARCH_DEPTH = 4
AI_LIMITED_MOVE_NUM = 20


class CHESS_TYPE(IntEnum):
    NONE = (0,)
    SLEEP_TWO = (1,)
    LIVE_TWO = (2,)
    SLEEP_THREE = 3
    LIVE_THREE = (4,)
    CHONG_FOUR = (5,)
    LIVE_FOUR = (6,)
    LIVE_FIVE = (7,)


CHESS_TYPE_NUM = 8

FIVE = CHESS_TYPE.LIVE_FIVE.value
FOUR, THREE, TWO = (
    CHESS_TYPE.LIVE_FOUR.value,
    CHESS_TYPE.LIVE_THREE.value,
    CHESS_TYPE.LIVE_TWO.value,
)
SFOUR, STHREE, STWO = (
    CHESS_TYPE.CHONG_FOUR.value,
    CHESS_TYPE.SLEEP_THREE.value,
    CHESS_TYPE.SLEEP_TWO.value,
)

SCORE_MAX = 0x7FFFFFFF
SCORE_MIN = -1 * SCORE_MAX
SCORE_FIVE, SCORE_FOUR, SCORE_SFOUR = 100000, 10000, 1000
SCORE_THREE, SCORE_STHREE, SCORE_TWO, SCORE_STWO = 100, 10, 8, 2


class Node:
    def __init__(self, parent, position, board: Board):
        self.parent = parent
        self.is_visited = False
        self.position = position
        self.board = board
        self.num_visited = 0

        self.num_win = 0
        self.num_lose = 0

        self.children = []

    def get_uct(self, factor=np.sqrt(2)):
        v = (
            (self.num_win - self.num_lose) / self.num_visited
            if self.num_visited > 0
            else 0
        )
        return v + factor * np.sqrt(
            np.log(self.parent.num_visited) / (1 + self.num_visited)
        )

    def __str__(self):
        return f"{self.parent}-{self.position}"


class MCTS:
    def __init__(
        self, time, computational_power, first_position_in_MCTS, board: Board, chess_len
    ):
        self.root = Node(parent=None, position=first_position_in_MCTS, board=board)
        self.fully_size = 8
        self.time = time
        self.computational_power = computational_power
        self.initial_board = board
        self.len = chess_len
        # [horizon, vertical, left diagonal, right diagonal]
        self.record = [
            [[0, 0, 0, 0] for x in range(chess_len)] for y in range(chess_len)
        ]
        self.count = [[0 for x in range(CHESS_TYPE_NUM)] for i in range(2)]


import copy
import time
from enum import IntEnum
from random import randint

import numpy as np

from board import Board

AI_SEARCH_DEPTH = 4
AI_LIMITED_MOVE_NUM = 20


class CHESS_TYPE(IntEnum):
    NONE = (0,)
    SLEEP_TWO = (1,)
    LIVE_TWO = (2,)
    SLEEP_THREE = 3
    LIVE_THREE = (4,)
    CHONG_FOUR = (5,)
    LIVE_FOUR = (6,)
    LIVE_FIVE = (7,)


CHESS_TYPE_NUM = 8

FIVE = CHESS_TYPE.LIVE_FIVE.value
FOUR, THREE, TWO = (
    CHESS_TYPE.LIVE_FOUR.value,
    CHESS_TYPE.LIVE_THREE.value,
    CHESS_TYPE.LIVE_TWO.value,
)
SFOUR, STHREE, STWO = (
    CHESS_TYPE.CHONG_FOUR.value,
    CHESS_TYPE.SLEEP_THREE.value,
    CHESS_TYPE.SLEEP_TWO.value,
)

SCORE_MAX = 0x7FFFFFFF
SCORE_MIN = -1 * SCORE_MAX
SCORE_FIVE, SCORE_FOUR, SCORE_SFOUR = 100000, 10000, 1000
SCORE_THREE, SCORE_STHREE, SCORE_TWO, SCORE_STWO = 100, 10, 8, 2


class Node:
    def __init__(self, parent, position, board: Board):
        self.parent = parent
        self.is_visited = False
        self.position = position
        self.board = board
        self.num_visited = 0

        self.num_win = 0
        self.num_lose = 0

        self.children = []

    def get_uct(self, factor=np.sqrt(2)):
        v = (
            (self.num_win - self.num_lose) / self.num_visited
            if self.num_visited > 0
            else 0
        )
        return v + factor * np.sqrt(
            np.log(self.parent.num_visited) / (1 + self.num_visited)
        )

    def __str__(self):
        return f"{self.parent}-{self.position}"


class MCTS:
    def __init__(
        self, time, computational_power, first_position_in_MCTS, board: Board, chess_len
    ):

        self.root = Node(parent=None, position=first_position_in_MCTS, board=board)
        self.fully_size = 8
        self.time = time
        self.computational_power = computational_power
        self.initial_board = board
        self.len = chess_len
        # [horizon, vertical, left diagonal, right diagonal]
        self.record = [
            [[0, 0, 0, 0] for x in range(chess_len)] for y in range(chess_len)
        ]
        self.count = [[0 for x in range(CHESS_TYPE_NUM)] for i in range(2)]

    def monte_carlo_tree_search(self):
        if self.evaluate(self.initial_board.board_matrix, 1, checkWin=False) <= 0:
            return self.findBestChess(self.initial_board.board_matrix, 1)

        else:
            iter = 0
            self.plays_rave = {}  # key:move, value:visited times
            self.wins_rave = {}  # key:move, value:{player: win times}
            while self.resources_left(time.time(), iter):
                leaf = self.traverse_alpha(self.root)  # leaf = unvisited node
                simulation_result = self.rollout(leaf)
                self.backpropagate(leaf, simulation_result)
                iter += 1

            print(iter)
            print(time.time() - self.time)
            return self.best_child_by_prob(self.root)

    def monte_carlo_tree_search_alpha(self):

        res = self.findBestChess(self.initial_board.board_matrix, 1)
        use_t = time.time() - self.time
        time.sleep((4.5 - use_t))
        return res

    # For the traverse function, to avoid using up too much time or resources, you may start considering only
    # a subset of children (e.g 5 children). Increase this number or by choosing this subset smartly later.
    def traverse(self, node):
        while self.fully_expanded(node):
            node = self.best_ucb(node)

        if not self.non_terminal(node):

            return node

        else:

            return self.pick_univisted(node)
            # in case no children are present / node is terminal

    def traverse_alpha(self, node):
        while self.fully_expanded(node):
            node = self.best_ucb(node)
        if not self.non_terminal(node):
            return node

        else:
            return self.pick_univisted_has_neighbour(node)
            # in case no children are present / node is terminal

    def rollout(self, node):
        while self.non_terminal(node):
            node = self.rollout_policy(node)
        return self.result(node)

    def rollout_policy(self, node):

        return self.pick_has_neighbour_simulation(node)

    def backpropagate(self, node, result):

        self.update_stats(node, result)
        if node.parent is not None:
            self.backpropagate(node.parent, -result)

    def is_root(self, node):

        return node.parent is None

    def best_child(self, node):
        # pick child with highest number of visits
        visit_num_of_children = np.array(
            list([child.num_visited for child in node.children])
        )
        best_index = np.argmax(visit_num_of_children)

        return node.children[best_index]

    def best_child_by_prob(self, node):
        # pick child with highest win percent node

        win_percent_of_children = []
        for child in node.children:
            if child.num_visited > 0:
                win_percent_of_children.append(child.num_win / child.num_visited)
        best_index = np.argmax(win_percent_of_children)

        return node.children[best_index]

    def update_stats(self, node: Node, result):
        if result == 1:
            node.num_win += 1
        elif result == -1:
            node.num_lose += 1

        node.num_visited += 1
        return

    def fully_expanded(self, node: Node):

        return len(node.children) >= self.fully_size

    def resources_left(self, curr_time, iteration):
        if curr_time - self.time < 4.99 and iteration < self.computational_power:
            return True
        return False

    def non_terminal(self, node: Node):
        node_position = node.position
        if node.board.is_win(node_position) or node.board.has_valid_position() == False:
            # 出现胜者或者没有可下位置则代表棋局结束
            return False
        return True

    def pick_univisted(self, node):  # 生成子状态
        # node.board 对应旧状态
        new_board = copy.deepcopy(node.board)  # 得到旧状态的copy

        valid_position = new_board.get_valid_position()
        new_board.player *= -1  # 与父状态（node)交换下子方
        new_board.move_in_position(valid_position, new_board.player)
        new_board.remove_valid_position(valid_position)
        child = Node(parent=node, position=valid_position, board=new_board)
        node.children.append(child)
        return child

    def pick_univisted_has_neighbour(self, node):  # 生成子状态
        # node.board 对应旧状态

        new_board = copy.deepcopy(node.board)  # 得到旧状态的copy

        valid_position_has_neighbour = (
            new_board.get_has_neighbour_valid_position_by_random()
        )

        new_board.remove_valid_position(valid_position_has_neighbour)
        new_board.player *= -1  # 与父状态（node)交换下子方
        new_board.move_in_position(valid_position_has_neighbour, new_board.player)
        child = Node(
            parent=node, position=valid_position_has_neighbour, board=new_board
        )
        node.children.append(child)
        return child

    def pick_random(self, node: Node):  # 随机下子并生成状态
        new_board = copy.deepcopy(node.board)

        valid_position = new_board.get_valid_position()
        new_board.remove_valid_position(valid_position)
        new_board.player *= -1  # 与父状态（node)交换下子方
        new_board.move_in_position(valid_position, new_board.player)
        child = Node(parent=node, position=valid_position, board=new_board)
        node.children.append(child)
        return child

    def pick_has_neighbour_simulation(self, node: Node):  # 随机下子并生成状态
        new_board = copy.deepcopy(node.board)

        valid_position = new_board.get_has_neighbour_valid_position_by_random()
        new_board.remove_valid_position(valid_position)
        new_board.player *= -1  # 与父状态（node)交换下子方
        new_board.move_in_position(valid_position, new_board.player)
        child = Node(parent=node, position=valid_position, board=new_board)
        node.children.append(child)
        return child

    def best_ucb(self, node: Node):
        uct_of_children = np.array(list([child.get_uct() for child in node.children]))
        best_index = np.argmax(uct_of_children)
        return node.children[best_index]

    def result(self, node):
        board = node.board
        node_position = node.position
        if board.is_win(node_position):
            return 1

        elif not board.has_valid_position():
            return 1 / 2

    def reset(self):
        for y in range(self.len):
            for x in range(self.len):
                for i in range(4):
                    self.record[y][x][i] = 0

        for i in range(len(self.count)):
            for j in range(len(self.count[0])):
                self.count[i][j] = 0

    # def click(self, map, x, y, turn):
    #     map.click(x, y, turn)

    def isWin(self, board, turn):
        return self.evaluate(board, turn, True)

    # evaluate score of point, to improve pruning efficiency
    def evaluatePointScore(self, board, x, y, mine, opponent):
        dir_offset = [(1, 0), (0, 1), (1, 1), (1, -1)]  # direction from left to right
        for i in range(len(self.count)):
            for j in range(len(self.count[0])):
                self.count[i][j] = 0

        board[y][x] = mine
        self.evaluatePoint(board, x, y, mine, opponent, self.count[mine - 1])
        mine_count = self.count[mine - 1]
        board[y][x] = opponent
        self.evaluatePoint(board, x, y, opponent, mine, self.count[opponent - 1])
        opponent_count = self.count[opponent - 1]
        board[y][x] = 0

        mscore = self.getPointScore(mine_count)
        oscore = self.getPointScore(opponent_count)

        return (mscore, oscore)

    # check if has a none empty position in it's radius range
    def hasNeighbor(self, board, x, y, radius):
        start_x, end_x = (x - radius), (x + radius)
        start_y, end_y = (y - radius), (y + radius)

        for i in range(start_y, end_y + 1):
            for j in range(start_x, end_x + 1):
                if i >= 0 and i < self.len and j >= 0 and j < self.len:
                    if board[i][j] != 0:
                        return True
        return False

    # get all positions near chess
    def genmove(self, board, turn):
        fives = []
        mfours, ofours = [], []
        msfours, osfours = [], []
        if turn == 1:
            mine = 1
            opponent = -1
        else:
            mine = -1
            opponent = 1

        moves = []
        radius = 1

        for y in range(self.len):
            for x in range(self.len):
                if board[y][x] == 0 and self.hasNeighbor(board, x, y, radius):
                    mscore, oscore = self.evaluatePointScore(
                        board, x, y, mine, opponent
                    )
                    point = (max(mscore, oscore), x, y)

                    if mscore >= SCORE_FIVE or oscore >= SCORE_FIVE:
                        fives.append(point)
                    elif mscore >= SCORE_FOUR:
                        mfours.append(point)
                    elif oscore >= SCORE_FOUR:
                        ofours.append(point)
                    elif mscore >= SCORE_SFOUR:
                        msfours.append(point)
                    elif oscore >= SCORE_SFOUR:
                        osfours.append(point)

                    moves.append(point)

        if len(fives) > 0:
            return fives

        if len(mfours) > 0:
            return mfours

        if len(ofours) > 0:
            if len(msfours) == 0:
                return ofours
            else:
                return ofours + msfours

        moves.sort(reverse=True)

        # FIXME: decrease think time: only consider limited moves with higher scores
        if self.maxdepth > 2 and len(moves) > AI_LIMITED_MOVE_NUM:
            moves = moves[:AI_LIMITED_MOVE_NUM]
        return moves

    def __search(self, board, turn, depth, alpha=SCORE_MIN, beta=SCORE_MAX):
        score = self.evaluate(board, turn)
        if depth <= 0 or abs(score) >= SCORE_FIVE:
            return score

        moves = self.genmove(board, turn)
        bestmove = None
        self.alpha += len(moves)

        # if there are no moves, just return the score
        if len(moves) == 0:
            return score

        for _, x, y in moves:
            board[y][x] = turn

            if turn == 1:
                op_turn = -1
            else:
                op_turn = 1

            score = -self.__search(board, op_turn, depth - 1, -beta, -alpha)

            board[y][x] = 0
            self.belta += 1

            # alpha/beta pruning
            if score > alpha:
                alpha = score
                bestmove = (x, y)
                if alpha >= beta:
                    break

        if depth == self.maxdepth and bestmove:
            self.bestmove = bestmove

        return alpha

    def search(self, board, turn, depth=4):
        self.maxdepth = depth
        self.bestmove = None
        score = self.__search(board, turn, depth)
        x, y = self.bestmove
        return score, x, y

    def findBestChess(self, board, turn):
        time1 = time.time()
        self.alpha = 0
        self.belta = 0
        score, x, y = self.search(board, turn, AI_SEARCH_DEPTH)
        time2 = time.time()

        # print(f'tims: {(time2 - time1)}, (x,y): {(x, y)}')
        # print('time[%.2f] (%d, %d), score[%d] alpha[%d] belta[%d]' % (
        #     (time2 - time1), x, y, score, self.alpha, self.belta))
        return (x, y)

    def getPointScore(self, count):
        score = 0
        if count[FIVE] > 0:
            return SCORE_FIVE

        if count[FOUR] > 0:
            return SCORE_FOUR

        # FIXME: the score of one chong four and no live three should be low, set it to live three
        if count[SFOUR] > 1:
            score += count[SFOUR] * SCORE_SFOUR
        elif count[SFOUR] > 0 and count[THREE] > 0:
            score += count[SFOUR] * SCORE_SFOUR
        elif count[SFOUR] > 0:
            score += SCORE_THREE

        if count[THREE] > 1:
            score += 5 * SCORE_THREE
        elif count[THREE] > 0:
            score += SCORE_THREE

        if count[STHREE] > 0:
            score += count[STHREE] * SCORE_STHREE
        if count[TWO] > 0:
            score += count[TWO] * SCORE_TWO
        if count[STWO] > 0:
            score += count[STWO] * SCORE_STWO

        return score

    # calculate score, FIXME: May Be Improved
    def getScore(self, mine_count, opponent_count):
        mscore, oscore = 0, 0
        if mine_count[FIVE] > 0:
            return (SCORE_FIVE, 0)
        if opponent_count[FIVE] > 0:
            return (0, SCORE_FIVE)

        if mine_count[SFOUR] >= 2:
            mine_count[FOUR] += 1
        if opponent_count[SFOUR] >= 2:
            opponent_count[FOUR] += 1

        if mine_count[FOUR] > 0:
            return (9050, 0)
        if mine_count[SFOUR] > 0:
            return (9040, 0)

        if opponent_count[FOUR] > 0:
            return (0, 9030)
        if opponent_count[SFOUR] > 0 and opponent_count[THREE] > 0:
            return (0, 9020)

        if mine_count[THREE] > 0 and opponent_count[SFOUR] == 0:
            return (9010, 0)

        if (
            opponent_count[THREE] > 1
            and mine_count[THREE] == 0
            and mine_count[STHREE] == 0
        ):
            return (0, 9000)

        if opponent_count[SFOUR] > 0:
            oscore += 400

        if mine_count[THREE] > 1:
            mscore += 500
        elif mine_count[THREE] > 0:
            mscore += 100

        if opponent_count[THREE] > 1:
            oscore += 2000
        elif opponent_count[THREE] > 0:
            oscore += 400

        if mine_count[STHREE] > 0:
            mscore += mine_count[STHREE] * 10
        if opponent_count[STHREE] > 0:
            oscore += opponent_count[STHREE] * 10

        if mine_count[TWO] > 0:
            mscore += mine_count[TWO] * 6
        if opponent_count[TWO] > 0:
            oscore += opponent_count[TWO] * 6

        if mine_count[STWO] > 0:
            mscore += mine_count[STWO] * 2
        if opponent_count[STWO] > 0:
            oscore += opponent_count[STWO] * 2

        return (mscore, oscore)

    def evaluate(self, board, turn, checkWin=False):
        self.reset()

        if turn == 1:
            mine = 1
            opponent = -1
        else:
            mine = -1
            opponent = 1

        for y in range(self.len):
            for x in range(self.len):
                if board[y][x] == mine:
                    self.evaluatePoint(board, x, y, mine, opponent)
                elif board[y][x] == opponent:
                    self.evaluatePoint(board, x, y, opponent, mine)

        mine_count = self.count[mine - 1]
        opponent_count = self.count[opponent - 1]
        if checkWin:
            return mine_count[FIVE] > 0
        else:
            mscore, oscore = self.getScore(mine_count, opponent_count)
            return mscore - oscore

    def evaluatePoint(self, board, x, y, mine, opponent, count=None):
        dir_offset = [(1, 0), (0, 1), (1, 1), (1, -1)]  # direction from left to right
        ignore_record = True
        if count is None:
            count = self.count[mine - 1]
            ignore_record = False
        for i in range(4):
            if self.record[y][x][i] == 0 or ignore_record:
                self.analysisLine(board, x, y, i, dir_offset[i], mine, opponent, count)

    # line is fixed len 9: XXXXMXXXX
    def getLine(self, board, x, y, dir_offset, mine, opponent):
        line = [0 for i in range(9)]

        tmp_x = x + (-5 * dir_offset[0])
        tmp_y = y + (-5 * dir_offset[1])
        for i in range(9):
            tmp_x += dir_offset[0]
            tmp_y += dir_offset[1]
            if tmp_x < 0 or tmp_x >= self.len or tmp_y < 0 or tmp_y >= self.len:
                line[i] = opponent  # set out of range as opponent chess
            else:
                line[i] = board[tmp_y][tmp_x]

        return line

    def analysisLine(self, board, x, y, dir_index, dir, mine, opponent, count):
        # record line range[left, right] as analysized
        def setRecord(self, x, y, left, right, dir_index, dir_offset):
            tmp_x = x + (-5 + left) * dir_offset[0]
            tmp_y = y + (-5 + left) * dir_offset[1]
            for i in range(left, right + 1):
                tmp_x += dir_offset[0]
                tmp_y += dir_offset[1]
                self.record[tmp_y][tmp_x][dir_index] = 1

        empty = 0
        left_idx, right_idx = 4, 4

        line = self.getLine(board, x, y, dir, mine, opponent)

        while right_idx < 8:
            if line[right_idx + 1] != mine:
                break
            right_idx += 1
        while left_idx > 0:
            if line[left_idx - 1] != mine:
                break
            left_idx -= 1

        left_range, right_range = left_idx, right_idx
        while right_range < 8:
            if line[right_range + 1] == opponent:
                break
            right_range += 1
        while left_range > 0:
            if line[left_range - 1] == opponent:
                break
            left_range -= 1

        chess_range = right_range - left_range + 1
        if chess_range < 5:
            setRecord(self, x, y, left_range, right_range, dir_index, dir)
            return CHESS_TYPE.NONE

        setRecord(self, x, y, left_idx, right_idx, dir_index, dir)

        m_range = right_idx - left_idx + 1

        # M:mine chess, P:opponent chess or out of range, X: empty
        if m_range >= 5:
            count[FIVE] += 1

        # Live Four : XMMMMX
        # Chong Four : XMMMMP, PMMMMX
        if m_range == 4:
            left_empty = right_empty = False
            if line[left_idx - 1] == empty:
                left_empty = True
            if line[right_idx + 1] == empty:
                right_empty = True
            if left_empty and right_empty:
                count[FOUR] += 1
            elif left_empty or right_empty:
                count[SFOUR] += 1

        # Chong Four : MXMMM, MMMXM, the two types can both exist
        # Live Three : XMMMXX, XXMMMX
        # Sleep Three : PMMMX, XMMMP, PXMMMXP
        if m_range == 3:
            left_empty = right_empty = False
            left_four = right_four = False
            if line[left_idx - 1] == empty:
                if line[left_idx - 2] == mine:  # MXMMM
                    setRecord(self, x, y, left_idx - 2, left_idx - 1, dir_index, dir)
                    count[SFOUR] += 1
                    left_four = True
                left_empty = True

            if line[right_idx + 1] == empty:
                if line[right_idx + 2] == mine:  # MMMXM
                    setRecord(self, x, y, right_idx + 1, right_idx + 2, dir_index, dir)
                    count[SFOUR] += 1
                    right_four = True
                right_empty = True

            if left_four or right_four:
                pass
            elif left_empty and right_empty:
                if chess_range > 5:  # XMMMXX, XXMMMX
                    count[THREE] += 1
                else:  # PXMMMXP
                    count[STHREE] += 1
            elif left_empty or right_empty:  # PMMMX, XMMMP
                count[STHREE] += 1

        # Chong Four: MMXMM, only check right direction
        # Live Three: XMXMMX, XMMXMX the two types can both exist
        # Sleep Three: PMXMMX, XMXMMP, PMMXMX, XMMXMP
        # Live Two: XMMX
        # Sleep Two: PMMX, XMMP
        if m_range == 2:
            left_empty = right_empty = False
            left_three = right_three = False
            if line[left_idx - 1] == empty:
                if line[left_idx - 2] == mine:
                    setRecord(self, x, y, left_idx - 2, left_idx - 1, dir_index, dir)
                    if line[left_idx - 3] == empty:
                        if line[right_idx + 1] == empty:  # XMXMMX
                            count[THREE] += 1
                        else:  # XMXMMP
                            count[STHREE] += 1
                        left_three = True
                    elif line[left_idx - 3] == opponent:  # PMXMMX
                        if line[right_idx + 1] == empty:
                            count[STHREE] += 1
                            left_three = True

                left_empty = True

            if line[right_idx + 1] == empty:
                if line[right_idx + 2] == mine:
                    if line[right_idx + 3] == mine:  # MMXMM
                        setRecord(
                            self, x, y, right_idx + 1, right_idx + 2, dir_index, dir
                        )
                        count[SFOUR] += 1
                        right_three = True
                    elif line[right_idx + 3] == empty:
                        # setRecord(self, x, y, right_idx+1, right_idx+2, dir_index, dir)
                        if left_empty:  # XMMXMX
                            count[THREE] += 1
                        else:  # PMMXMX
                            count[STHREE] += 1
                        right_three = True
                    elif left_empty:  # XMMXMP
                        count[STHREE] += 1
                        right_three = True

                right_empty = True

            if left_three or right_three:
                pass
            elif left_empty and right_empty:  # XMMX
                count[TWO] += 1
            elif left_empty or right_empty:  # PMMX, XMMP
                count[STWO] += 1

        # Live Two: XMXMX, XMXXMX only check right direction
        # Sleep Two: PMXMX, XMXMP
        if m_range == 1:
            left_empty = right_empty = False
            if line[left_idx - 1] == empty:
                if line[left_idx - 2] == mine:
                    if line[left_idx - 3] == empty:
                        if line[right_idx + 1] == opponent:  # XMXMP
                            count[STWO] += 1
                left_empty = True

            if line[right_idx + 1] == empty:
                if line[right_idx + 2] == mine:
                    if line[right_idx + 3] == empty:
                        if left_empty:  # XMXMX
                            # setRecord(self, x, y, left_idx, right_idx+2, dir_index, dir)
                            count[TWO] += 1
                        else:  # PMXMX
                            count[STWO] += 1
                elif line[right_idx + 2] == empty:
                    if (
                        line[right_idx + 3] == mine and line[right_idx + 4] == empty
                    ):  # XMXXMX
                        count[TWO] += 1

        return CHESS_TYPE.NONE
