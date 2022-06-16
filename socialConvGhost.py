from agent import Agent
from node import Node
import numpy as np
import copy
from time import sleep

N_ACTIONS = 4
DOWN, LEFT, UP, RIGHT = range(N_ACTIONS)
ROLES = [DOWN, LEFT, UP, RIGHT]

class SocialConvGhost(Agent):
    def __init__(self, agent_id, n_agents):
        super(SocialConvGhost, self).__init__(f"SocialConvGhost")
        self.agent_id = agent_id
        self.n_agents = n_agents
        self.n_actions = N_ACTIONS
        self.position = [13, 11 + agent_id]
        self.observation = None
        self.alive = True

    def see(self, observation: np.ndarray):
        self.observation = observation
        
    def reset_position(self):
        self.position = [13, 11 + self.agent_id]

    def set_position(self, line, column):
        self.position = [line, column]

    def ret_move(self, map: np.ndarray, path: np.ndarray, pacman):
        start = path[0]
        nxt = path[1]
        move = [start[0] - nxt[0], start[1] - nxt[1]]
        if (map[nxt[0], nxt[1]] != 1):
            if move == [-1, 0]:
                return 0
            elif move == [0, 1]:
                return 1
            elif move == [1, 0]:
                return 2
            elif move == [0, -1]:
                return 3
        else:
            return 4

    def check_validity(self, map: np.ndarray, pacman):
        options = []
        for i in ROLES:
            if (i == DOWN) and (29 > pacman[0] + 1 > 0) and map[pacman[0] + 1, pacman[1]] != 1:
                options.append(i)
            elif (i == LEFT) and (26 > pacman[1] - 1 > 0) and map[pacman[0], pacman[1] - 1] != 1:
                options.append(i)
            elif (i == UP) and (29 > pacman[0] - 1 > 0) and map[pacman[0] - 1, pacman[1]] != 1:
                options.append(i)
            elif (i == RIGHT) and (26 > pacman[1] + 1 > 0) and map[pacman[0], pacman[1] + 1] != 1:
                options.append(i)
        return options

    def choose_role (self, options):
        if len(options) > self.agent_id:
            return options[self.agent_id]
        else:
            return 4

    def define_target (self, role, pacman):
        target = []
        if (role == DOWN):
            target = [pacman[0] + 1, pacman[1]]
        elif (role == LEFT):
            target = [pacman[0], pacman[1] - 1]
        elif (role == UP):
            target = [pacman[0] - 1, pacman[1]]
        elif (role == RIGHT):
            target = [pacman[0], pacman[1] + 1]
        elif (role == 4):
            target = pacman
        return target


    def action(self, map: np.ndarray, step) -> int:
        pacman = self.observation[(self.n_agents) * 2:]
        start_node = Node(None, self.position)
        start_node.g = start_node.h = start_node.f = 0

        role = self.choose_role(self.check_validity(map, pacman))

        target = self.define_target(role, pacman)

        end_node = Node(None, target)
        end_node.g = end_node.h = end_node.f = 0

        open_list = []
        closed_list = []

        open_list.append(start_node)

        while len(open_list) > 0:

            current_node = open_list[0]
            current_index = 0
            for index, item in enumerate(open_list):

                if item.f < current_node.f:
                    current_node = item
                    current_index = index

            open_list.pop(current_index)
            closed_list.append(current_node)

            if current_node == end_node:
                path = []
                current = current_node
                while current is not None:
                    path.append(current.position)
                    current = current.parent
                if len(path) == 1:
                    path.append(path[0])
                    path[0] = pacman
                elif len(path) == 2:
                    path.insert(0, pacman)
                return self.ret_move(map, path[::-1], pacman)
            
            children = []
            for new_position in [[0, -1], [0, 1], [-1, 0], [1, 0]]:

                node_position = [current_node.position[0] + new_position[0], current_node.position[1] + new_position[1]]

                if node_position[0] > 28 or node_position[0] < 0 or node_position[1] > 25 or node_position[1] < 0:
                    continue

                if map[node_position[0]][node_position[1]] != 0:
                    continue

                new_node = Node(current_node, node_position)

                if new_node in closed_list:
                    continue

                children.append(new_node)

            for child in children:

                for closed_child in closed_list:
                    if child == closed_child:
                        continue

                child.g = current_node.g + 1
                child.h = abs(child.position[0] - end_node.position[0]) + abs(child.position[1] - end_node.position[1])
                child.f = child.g + child.h

                for open_node in open_list:
                    if child == open_node and child.g > open_node.g:
                        continue

                open_list.append(child)
