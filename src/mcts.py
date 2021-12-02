import copy
import time

import numpy as np

from board import Board


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
        v = (self.num_win - self.num_lose) / self.num_visited if self.num_visited > 0 else 0
        return v + factor * np.sqrt(np.log(self.parent.num_visited) / (1 + self.num_visited))

    def __str__(self):
        return f"{self.parent}-{self.position}"


class MCTS:
    def __init__(self, time, computational_power, first_position_in_MCTS, board):
        self.root = Node(parent=None, position=first_position_in_MCTS, board=board)
        self.fully_size = 8
        self.time = time
        self.computational_power = computational_power

    def monte_carlo_tree_search(self):
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

        valid_position_has_neighbour = new_board.get_has_neighbour_valid_position_by_random()

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
