import copy
import gym
import numpy as np
from gym import spaces

from time import sleep #Delete in the end

from ma_gym.envs.utils.action_space import MultiAgentActionSpace
from ma_gym.envs.utils.observation_space import MultiAgentObservationSpace
from ma_gym.envs.utils.draw import draw_grid, fill_cell, draw_circle, write_cell_text

from ghost import Ghost
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
        self.ghosts = {i: Ghost(i, self.n_ghosts) for i in range(self.n_ghosts)}

        #observation
        self.observation_space = MultiAgentObservationSpace([spaces.Box(low= np.array([0, 0]), high= np.array([29, 26])) for _ in range(self.n_ghosts+1)])

        #reward
        self.reward = None 

    def render(self, mode='human'):
        img = copy.copy(draw_grid(self.grid[0], self.grid[1], cell_size=35, fill='white'))

        for i in range (29):
            for j in range (26):
                if self.map[i][j] == 1:
                    fill_cell(img, [i, j], cell_size=35, fill='black', margin=0.1)

        for i in range(self.n_ghosts):
            fill_cell(img, self.ghosts[i].get_position(), cell_size=35, fill=(3, 190, 252), margin=0.1)
        
        fill_cell(img, self.pacman.get_position(), cell_size=35, fill=(252, 223, 3), margin=0.1)

        for i in range(self.n_ghosts):
            draw_circle(img, self.ghosts[i].get_position(), cell_size=35, fill=(2, 15, 250))
            write_cell_text(img, text=str(i + 1), pos=self.ghosts[i].get_position(), cell_size=35,
                            fill='white', margin=0.4)
    
        draw_circle(img, self.pacman.get_position(), cell_size=35, fill=(252, 248, 5))
        write_cell_text(img, text="P", pos=self.pacman.get_position(), cell_size=35,
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

        ghost_position = []
        #set positions
        for i in range(self.n_ghosts):
            self.map[self.ghosts[i].get_position()[0], self.ghosts[i].get_position()[1]] = 0
            self.ghosts[i].reset_position()
            self.map[self.ghosts[i].get_position()[0], self.ghosts[i].get_position()[1]] = 2
            ghost_position.append(self.ghosts[i].get_position())
        
        #self.map[self.pacman.get_position()[0], self.pacman.get_position()[1]] = 0
        self.pacman.reset_position()
        #self.map[self.pacman.get_position()[0], self.pacman.get_position()[1]] = 3
        return ghost_position, self.pacman.get_position()

    def step(self, agents_action):
        self.step_count += 1

        #move
        for agent_i, action in enumerate(agents_action):
            if agent_i < self.n_ghosts:
                self.map[self.ghosts[agent_i].get_position()[0], self.ghosts[agent_i].get_position()[1]] = 0
                new_ghost_position = self.next_position(copy.copy(self.ghosts[agent_i].get_position()), action, agent_i)
                self.ghosts[agent_i].set_position(new_ghost_position[0], new_ghost_position[1])
                self.map[self.ghosts[agent_i].get_position()[0], self.ghosts[agent_i].get_position()[1]] = 2
            elif self.pacman.is_alive():
                #self.map[self.pacman.get_position()[0], self.pacman.get_position()[1]] = 0
                new_pacman_position = self.next_position(copy.copy(self.pacman.get_position()), action, agent_i)
                self.pacman.set_position(new_pacman_position[0], new_pacman_position[1])
                #self.map[self.pacman.get_position()[0], self.pacman.get_position()[1]] = 3

        for i in range(self.n_ghosts):
            #se apanhar
            if self.ghosts[i].get_position() == self.pacman.get_position():
                self.reward[i] = 5
                self.reward[self.n_ghosts] = -5
                self.pacman.kill()
            
            #se n apanhar
            else:
                self.reward[i] += -1
                self.reward[self.n_ghosts] += 1

        return [self.ghosts[i].get_position() for i in range(self.n_ghosts)] + [self.pacman.get_position()], self.reward, self.pacman.is_alive()

    def next_position(self, curr_pos, move, id):
        if move == 0:  # down
            next_pos = [curr_pos[0] + 1, curr_pos[1]]
            if not self.is_valid(next_pos, id):
                #next_pos = [curr_pos[0], curr_pos[1]]
                next_pos = self.next_position(curr_pos, np.random.randint(4), id)
        elif move == 1:  # left
            next_pos = [curr_pos[0], curr_pos[1] - 1]
            if not self.is_valid(next_pos, id):
                #next_pos = [curr_pos[0], curr_pos[1]]
                next_pos = self.next_position(curr_pos, np.random.randint(4), id)
        elif move == 2:  # up
            next_pos = [curr_pos[0] - 1, curr_pos[1]]
            if not self.is_valid(next_pos, id):
                #next_pos = [curr_pos[0], curr_pos[1]]
                next_pos = self.next_position(curr_pos, np.random.randint(4), id)
        elif move == 3:  # right
            next_pos = [curr_pos[0], curr_pos[1] + 1]
            if not self.is_valid(next_pos, id):
                #next_pos = [curr_pos[0], curr_pos[1]]
                next_pos = self.next_position(curr_pos, np.random.randint(4), id)
        #elif move == 4:  # no-op
        return next_pos

    def is_valid(self, pos, id):
        if (id < self.n_ghosts):
            return (0 <= pos[0] < self.grid[0]) and (0 <= pos[1] < self.grid[1]) and (self.map[pos[0]][pos[1]] != 1) and (self.map[pos[0]][pos[1]] != 2)
        else:
            return (0 <= pos[0] < self.grid[0]) and (0 <= pos[1] < self.grid[1]) and (self.map[pos[0]][pos[1]] == 0)