from agent import Agent
from node import Node
import numpy as np
from time import sleep

N_ACTIONS = 4
DOWN, LEFT, UP, RIGHT = range(N_ACTIONS)

class Ghost(Agent):
    def __init__(self, agent_id, n_agents):
        super(Ghost, self).__init__(f"Ghost")
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

    def ret_move(self, path: np.ndarray):
        start = path[0]
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

    def action(self, map: np.ndarray, step) -> int:
        # Create start and end node
        start_node = Node(None, self.position)
        start_node.g = start_node.h = start_node.f = 0
        end_node = Node(None, self.observation[(self.n_agents) * 2:])
        end_node.g = end_node.h = end_node.f = 0

        # Initialize both open and closed list
        open_list = []
        closed_list = []

        # Add the start node
        open_list.append(start_node)

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
            if current_node == end_node:
                path = []
                current = current_node
                while current is not None:
                    path.append(current.position)
                    current = current.parent
                #print(path[::-1])
                return self.ret_move(path[::-1])
            
            # Generate children
            children = []
            for new_position in [[0, -1], [0, 1], [-1, 0], [1, 0]]: # Adjacent squares

                # Get node position
                node_position = [current_node.position[0] + new_position[0], current_node.position[1] + new_position[1]]

                # Make sure within range
                if node_position[0] > 28 or node_position[0] < 0 or node_position[1] > 25 or node_position[1] < 0:
                    continue

                # Make sure walkable terrain
                if map[node_position[0]][node_position[1]] != 0:
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
                child.h = abs(child.position[0] - end_node.position[0]) + abs(child.position[1] - end_node.position[1])
                child.f = child.g + child.h

                # Child is already in the open list
                for open_node in open_list:
                    if child == open_node and child.g > open_node.g:
                        continue

                # Add the child to the open list
                open_list.append(child)