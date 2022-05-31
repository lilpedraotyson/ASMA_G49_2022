import copy
import gym
import numpy as np
from gym import spaces

from ma_gym.envs.utils.action_space import MultiAgentActionSpace
from ma_gym.envs.utils.observation_space import MultiAgentObservationSpace
from ma_gym.envs.utils.draw import draw_grid, fill_cell, draw_circle, write_cell_text

class Pacman(gym.Env):
    def __init__(self, n_ghosts):
        #global variables
        self.n_ghosts = n_ghosts
        self.grid = (29, 26)
        self.pacman_alive = None
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
                            [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0],
                            [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

        #action_space = number of action of each agent (UP, DOWN, LEFT, RIGHT)
        self.action_space = MultiAgentActionSpace([spaces.Discrete(4) for _ in range(self.n_ghosts+1)])

        #position -> starts has none
        self.ghost_position = {_: None for _ in range(self.n_ghosts)}
        self.pacman_position = None

        #observation
        self.observation_space = MultiAgentObservationSpace([spaces.Box(low= np.array([0, 0]), high= np.array([29, 26])) for _ in range(self.n_ghosts+1)])

        #reward
        self.reward = None

    def step(self, agents_action):
        self.step_count += 1

        #move
        for agent_i, action in enumerate(agents_action):
            if agent_i < self.n_ghosts:
                self.ghost_position[agent_i] = self.next_position(copy.copy(self.ghost_position[agent_i]), action)
            elif self.pacman_alive:
                self.pacman_position = self.next_position(copy.copy(self.pacman_position), action)

        for i in self.ghost_position:
            #se apanhar
            if self.ghost_position[i] == self.pacman_position:
                self.reward[i] = 5
                self.reward[self.n_ghosts] = -5
                self.pacman_alive = False
            
            #se n apanhar
            else:
                self.reward[i] += -1
                self.reward[self.n_ghosts] += 1

        return [self.ghost_position[i] for i in self.ghost_position] + [self.pacman_position], self.reward, self.pacman_alive 

    def render(self, mode='human'):
        img = copy.copy(draw_grid(self.grid[0], self.grid[1], cell_size=35, fill='white'))

        for i in range (29):
            for j in range (26):
                if self.map[i][j] == 1:
                    fill_cell(img, [i, j], cell_size=35, fill='black', margin=0.1)

        for i in range(self.n_ghosts):
            fill_cell(img, self.ghost_position[i], cell_size=35, fill=(3, 190, 252), margin=0.1)
        
        fill_cell(img, self.pacman_position, cell_size=35, fill=(252, 223, 3), margin=0.1)

        for i in range(self.n_ghosts):
            draw_circle(img, self.ghost_position[i], cell_size=35, fill=(2, 15, 250))
            write_cell_text(img, text=str(i + 1), pos=self.ghost_position[i], cell_size=35,
                            fill='white', margin=0.4)
    
        draw_circle(img, self.pacman_position, cell_size=35, fill=(252, 248, 5))
        write_cell_text(img, text="P", pos=self.pacman_position, cell_size=35,
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
        self.pacman_alive = True
        #self.ghosts_done = [False for _ in range(self.n_ghosts)]
        self.reward = {_: 0 for _ in range(self.n_ghosts+1)}

        #set positions
        positions = self.position()
        counter = 0 
        for i in self.ghost_position:
            self.ghost_position[i] = positions[counter]
            counter += 1
        
        self.pacman_position = positions[counter]

        return positions, self.ghost_position, self.pacman_position

    def position(self):

        ghost_position = []
        for i in range(self.n_ghosts):
            ghost_position.append([np.random.randint(26), np.random.randint(29)])
        
        pacman_position = [np.random.randint(26), np.random.randint(29)]

        return np.array(ghost_position + [pacman_position])

    def next_position(self, curr_pos, move):
        if move == 0:  # down
            next_pos = [curr_pos[0] + 1, curr_pos[1]]
            if not self.is_valid(next_pos):
                next_pos = [curr_pos[0], curr_pos[1]]
        elif move == 1:  # left
            next_pos = [curr_pos[0], curr_pos[1] - 1]
            if not self.is_valid(next_pos):
                next_pos = [curr_pos[0], curr_pos[1]]
        elif move == 2:  # up
            next_pos = [curr_pos[0] - 1, curr_pos[1]]
            if not self.is_valid(next_pos):
                next_pos = [curr_pos[0], curr_pos[1]]
        elif move == 3:  # right
            next_pos = [curr_pos[0], curr_pos[1] + 1]
            if not self.is_valid(next_pos):
                next_pos = [curr_pos[0], curr_pos[1]]
        #elif move == 4:  # no-op
        return next_pos

    def is_valid(self, pos):
        return (0 <= pos[0] < self.grid[0]) and (0 <= pos[1] < self.grid[1])