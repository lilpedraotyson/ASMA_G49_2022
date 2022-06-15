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

    def ret_move(self, path: np.ndarray, pacman):
        start = path[0]
        '''if len(path) > 1:
            nxt = path[1]
        else:
            nxt = pacman'''
        nxt = path[1]
        move = [start[0] - nxt[0], start[1] - nxt[1]]
        if move == [-1, 0]:
            return 0
        elif move == [0, 1]:
            return 1
        elif move == [1, 0]:
            return 2
        elif move == [0, -1]:
            return 3

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
            cop = copy.deepcopy(ROLES)
            for i in options:
                cop.remove(i)
            return cop[self.agent_id - len(options)]

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
        return target


    def action(self, map: np.ndarray, step) -> int:
        # Create start and end node
        pacman = self.observation[(self.n_agents) * 2:]
        start_node = Node(None, self.position)
        start_node.g = start_node.h = start_node.f = 0

        role = self.choose_role(self.check_validity(map, pacman))

        target = self.define_target(role, pacman)
        print(target)

        end_node = Node(None, target)
        end_node.g = end_node.h = end_node.f = 0

        # Initialize both open and closed list
        open_list = []
        closed_list = []

        # Add the start node
        open_list.append(end_node)

        # Loop until you find the end
        while len(open_list) > 0:

            # Get the current node
            current_node = open_list[0]
            current_index = 0
            for index, item in enumerate(open_list):

                if item.f < current_node.f:
                    current_node = item
                    current_index = index

            # Pop current off open list, add to closed list
            open_list.pop(current_index)
            closed_list.append(current_node)

            # Found the goal
            if current_node == start_node:
                path = []
                current = current_node
                while current is not None:
                    path.append(current.position)
                    current = current.parent
                if len(path) == 1:
                    path.append(pacman)
                return self.ret_move(path, pacman)
            
            # Generate children
            children = []
            for new_position in [[0, -1], [0, 1], [-1, 0], [1, 0]]: # Adjacent squares

                # Get node position
                node_position = [current_node.position[0] + new_position[0], current_node.position[1] + new_position[1]]

                # Make sure within range
                if node_position[0] > 28 or node_position[0] < 0 or node_position[1] > 25 or node_position[1] < 0:
                    continue

                # Make sure walkable terrain
                if map[node_position[0]][node_position[1]] != 0 and node_position != self.position:
                    continue

                # Create new node
                new_node = Node(current_node, node_position)

                if new_node in closed_list:
                    continue

                # Append
                children.append(new_node)

            # Loop through children
            for child in children:

                # Child is on the closed list
                for closed_child in closed_list:
                    if child == closed_child:
                        continue

                # Create the f, g, and h values
                child.g = current_node.g + 1
                child.h = abs(child.position[0] - start_node.position[0]) + abs(child.position[1] - start_node.position[1])
                child.f = child.g + child.h

                # Child is already in the open list
                for open_node in open_list:
                    if child == open_node and child.g > open_node.g:
                        continue

                # Add the child to the open list
                open_list.append(child)
