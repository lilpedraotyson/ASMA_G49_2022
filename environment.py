import copy
import gym
import numpy as np
from gym import spaces

from time import sleep #Delete in the end

from ma_gym.envs.utils.action_space import MultiAgentActionSpace
from ma_gym.envs.utils.observation_space import MultiAgentObservationSpace
from ma_gym.envs.utils.draw import draw_grid, fill_cell, draw_circle, write_cell_text

from pacman import Pacman

class Environment(gym.Env):
    def __init__(self, n_ghosts):
        #global variables
        self.n_ghosts = n_ghosts
        self.grid = (29, 26)
        #self.ghosts_done = [None for _ in range(self.n_ghosts)]
        self.step_count = 0
        self.viewer = None
        self.map = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0],
                            [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0],
                            [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0],
                            [0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0],
                            [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                            [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                            [0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0],
                            [0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0],
                            [0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
                            [0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
                            [0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
                            [0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0],
                            [0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0],
                            [1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0],
                            [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0],
                            [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                            [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1],
                            [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1],
                            [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

        #action_space = number of action of each agent (UP, DOWN, LEFT, RIGHT)
        self.action_space = MultiAgentActionSpace([spaces.Discrete(4) for _ in range(self.n_ghosts+1)])

        #create agents
        self.pacman = Pacman()
        self.ghosts = None

        #observation
        self.observation_space = MultiAgentObservationSpace([spaces.Box(low= np.array([0, 0]), high= np.array([29, 26])) for _ in range(self.n_ghosts+1)])
 

    def render(self, mode='human'):
        img = copy.copy(draw_grid(self.grid[0], self.grid[1], cell_size=35, fill='white'))

        for i in range (29):
            for j in range (26):
                if self.map[i][j] == 1:
                    fill_cell(img, [i, j], cell_size=35, fill='black', margin=0.1)

        for i in range(self.n_ghosts):
            fill_cell(img, self.ghosts[i].position, cell_size=35, fill=(3, 190, 252), margin=0.1)
        
        #fill_cell(img, self.pacman.get_position(), cell_size=35, fill=(252, 223, 3), margin=0.1)
        self.draw_pacman_vision(self.pacman.position, img)

        for i in range(self.n_ghosts):
            draw_circle(img, self.ghosts[i].position, cell_size=35, fill=(2, 15, 250))
            write_cell_text(img, text=str(i + 1), pos=self.ghosts[i].position, cell_size=35,
                            fill='white', margin=0.4)
    
        draw_circle(img, self.pacman.position, cell_size=35, fill=(252, 248, 5))
        write_cell_text(img, text="P", pos=self.pacman.position, cell_size=35,
                        fill='black', margin=0.4)

        img = np.asarray(img)
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def reset(self):
        self.step_count = 0
        #self.ghosts_done = [False for _ in range(self.n_ghosts)]
        self.reward = {_: 0 for _ in range(self.n_ghosts+1)}

        agent_positions = []
        #set positions
        for i in range(self.n_ghosts):
            self.map[self.ghosts[i].position[0], self.ghosts[i].position[1]] = 0
            self.ghosts[i].reset_position()
            self.map[self.ghosts[i].position[0], self.ghosts[i].position[1]] = 2
            agent_positions.append(self.ghosts[i].position[0])
            agent_positions.append(self.ghosts[i].position[1])
        
        self.pacman.reset_position()
        agent_positions.append(self.pacman.position[0])
        agent_positions.append(self.pacman.position[1])
        return agent_positions

    def step(self, agents_action):
        self.step_count += 1

        #move
        for agent_i, action in enumerate(agents_action):
            if agent_i < self.n_ghosts:
                self.map[self.ghosts[agent_i].position[0], self.ghosts[agent_i].position[1]] = 0
                new_ghost_position = self.next_position(copy.copy(self.ghosts[agent_i].position), action, agent_i)
                if (self.map[new_ghost_position[0], new_ghost_position[1]] != 0):
                    new_ghost_position = self.next_position(copy.copy(self.ghosts[agent_i].position), self.ghosts[agent_i].action(self.map, self.step_count), agent_i)
                self.ghosts[agent_i].set_position(new_ghost_position[0], new_ghost_position[1])
                self.map[self.ghosts[agent_i].position[0], self.ghosts[agent_i].position[1]] = 2

                if self.ghosts[agent_i].position == self.pacman.position:
                    self.pacman.kill()

            elif self.pacman.is_alive():
                new_pacman_position = self.next_position(copy.copy(self.pacman.position), action, agent_i)
                self.pacman.set_position(new_pacman_position[0], new_pacman_position[1])

        agent_positions = []

        for i in range(self.n_ghosts):
            agent_positions.append(self.ghosts[i].position[0])
            agent_positions.append(self.ghosts[i].position[1])

        agent_positions.append(self.pacman.position[0])
        agent_positions.append(self.pacman.position[1])

        return agent_positions, self.pacman.is_alive()

    def next_position(self, curr_pos, move, id):
        next_pos = [curr_pos[0], curr_pos[1]]
        if move == 0:  # down
            next_pos = [curr_pos[0] + 1, curr_pos[1]]
        elif move == 1:  # left
            next_pos = [curr_pos[0], curr_pos[1] - 1]
        elif move == 2:  # up
            next_pos = [curr_pos[0] - 1, curr_pos[1]]
        elif move == 3:  # right
            next_pos = [curr_pos[0], curr_pos[1] + 1]

        if id == self.n_ghosts and not self.way_out():
            if (self.is_valid(next_pos)):
                return next_pos
            else:
                return curr_pos

        if id == self.n_ghosts and (not self.is_valid(next_pos) or not self.check_ghost(self.pacman.orientation)):
            next_pos = self.next_position(curr_pos, self.pacman.action(1), id)

        return next_pos

    def is_valid(self, pos):
        return (0 <= pos[0] < self.grid[0]) and (0 <= pos[1] < self.grid[1]) and (self.map[pos[0]][pos[1]] == 0)

    def draw_pacman_vision(self, position, img):
        #for i in range(4):
        i = self.pacman.orientation
        for pos in range(29):
            if i == 0: #down
                next_fill = [position[0] + pos, position[1]]
            elif i == 1: #left
                next_fill = [position[0], position[1] - pos]
            elif i == 2: #up
                next_fill = [position[0] - pos, position[1]]
            elif i == 3: #right
                next_fill = [position[0], position[1] + pos]
            else:
                next_fill = position
            if self.is_valid(next_fill):
                fill_cell(img, next_fill, cell_size=35, fill=(252, 223, 3), margin=0.1)
            else:
                break

    def check_ghost(self, i):
        if i == 0:
            for pos in range(29 - self.pacman.position[0]):
                if (28 < self.pacman.position[0] + pos < 0) or self.map[self.pacman.position[0] + pos, self.pacman.position[1]] == 1:
                    break
                elif self.map[self.pacman.position[0] + pos, self.pacman.position[1]] == 2: #down
                    return False
        elif i == 1:
            for pos in range (self.pacman.position[1] + 1):
                if (25 < self.pacman.position[1] - pos < 0) or self.map[self.pacman.position[0], self.pacman.position[1] - pos] == 1:
                    break
                elif self.map[self.pacman.position[0], self.pacman.position[1] - pos] == 2: #left
                    return False
        elif i == 2:
            for pos in range (self.pacman.position[0] + 1):
                if (28 < self.pacman.position[0] - pos < 0) or self.map[self.pacman.position[0] - pos, self.pacman.position[1]] == 1:
                    break
                elif self.map[self.pacman.position[0] - pos, self.pacman.position[1]] == 2: #up
                    return False
        elif i == 3:
            for pos in range(26 - self.pacman.position[1]):
                if (25 < self.pacman.position[1] + pos < 0) or self.map[self.pacman.position[0], self.pacman.position[1] + pos] == 1:
                    break
                elif self.map[self.pacman.position[0], self.pacman.position[1] + pos] == 2: #right
                    return False
        
        return True

    def way_out(self):
        options = []
        for i in range (4):
            if (self.check_ghost(i)):
                if (i == 0) and (29 > self.pacman.position[0] + 1 > 0) and self.map[self.pacman.position[0] + 1, self.pacman.position[1]] == 0:
                    options.append(i)
                    #return True
                elif (i == 1) and (26 > self.pacman.position[1] - 1 > 0) and self.map[self.pacman.position[0], self.pacman.position[1] - 1] == 0:
                    options.append(i)
                    #return True
                elif (i == 2) and (29 > self.pacman.position[0] - 1 > 0) and self.map[self.pacman.position[0] - 1, self.pacman.position[1]] == 0:
                    options.append(i)
                    #return True
                elif (i == 3) and (26 > self.pacman.position[1] + 1 > 0) and self.map[self.pacman.position[0], self.pacman.position[1] + 1] == 0:
                    options.append(i)
                    #return True
        if len(options) > 0:
            return True
        else:
            return False

