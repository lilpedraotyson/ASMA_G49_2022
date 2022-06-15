from agent import Agent
from node import Node
import numpy as np
from time import sleep

N_ACTIONS = 4
DOWN, LEFT, UP, RIGHT = range(N_ACTIONS)

class RandomGhost(Agent):
    def __init__(self, agent_id, n_agents):
        super(RandomGhost, self).__init__(f"RandomGhost")
        self.agent_id = agent_id
        self.n_agents = n_agents
        self.n_actions = N_ACTIONS
        self.position = [13, 11 + agent_id]
        self.observation = None
        self.orientation = np.random.randint(4)
        self.flag = 0
        self.alive = True

    def see(self, observation: np.ndarray):
        self.observation = observation
        
    def reset_position(self):
        self.alive = True
        self.position = [13, 11 + self.agent_id]
        self.flag = 0

    def set_position(self, line, column):
        self.position = [line, column]
    
    def next_position(self, curr_pos, move):
        next_pos = [curr_pos[0], curr_pos[1]]
        if move == 0:  # down
            next_pos = [curr_pos[0] + 1, curr_pos[1]]
        elif move == 1:  # left
            next_pos = [curr_pos[0], curr_pos[1] - 1]
        elif move == 2:  # up
            next_pos = [curr_pos[0] - 1, curr_pos[1]]
        elif move == 3:  # right
            next_pos = [curr_pos[0], curr_pos[1] + 1]
        
        return next_pos

    def action(self, map: np.ndarray, step) -> int:
        if self.flag > 500:
            self.alive = False
            return np.random.randint(4)
        if np.remainder(step, 10) == 0 and ((11 <= self.position[0] <= 15) and (9 <= self.position[1] <= 16)):
            self.orientation = 2
        opt = self.possible_ways(map)
        if (len(opt) > 1) and not ((11 <= self.position[0] <= 15) and (9 <= self.position[1] <= 16)) and not (self.position == [10, 12] or self.position == [10, 13]):
            self.orientation = opt[np.random.randint(len(opt))]
        pos = self.next_position(self.position, self.orientation)
        if not (0 <= pos[0] < 29) or not (0 <= pos[1] < 26) or not (map[pos[0]][pos[1]] == 0):
            self.flag += 1
            self.orientation = opt[np.random.randint(len(opt))]
            self.action(map, step)
        else:
            self.flag = 0
            return self.orientation

    def possible_ways(self, map: np.ndarray):
        options = []
        for i in range (4):
            if (i != self.orientation):
                if (i == 0) and (29 > self.position[0] + 1 > 0) and map[self.position[0] + 1, self.position[1]] == 0:
                    options.append(i)
                elif (i == 1) and (26 > self.position[1] - 1 > 0) and map[self.position[0], self.position[1] - 1] == 0:
                    options.append(i)
                elif (i == 2) and (29 > self.position[0] - 1 > 0) and map[self.position[0] - 1, self.position[1]] == 0:
                    options.append(i)
                elif (i == 3) and (26 > self.position[1] + 1 > 0) and map[self.position[0], self.position[1] + 1] == 0:
                    options.append(i)
        return options
