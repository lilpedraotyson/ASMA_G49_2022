import gym
import numpy as np
from time import sleep

from environment import Environment

if __name__ == '__main__':
    n_agents = 4
    environment = Environment(n_agents - 1)
    observation = environment.reset()
    print('ghost position: {} pacman position: {}'.format(observation[:(n_agents - 1) * 2], observation[(n_agents - 1) * 2:]))
    
    sleep(1)
    alive = True
    while alive:
        for i in range(n_agents - 1):
            environment.ghosts[i].see(observation) 
        next_obs, reward, alive = environment.step([environment.ghosts[i].action() for i in range(n_agents - 1)] + [environment.pacman.action()])
        print('positions: {} reward: {} pacman alive?: {}'.format(next_obs, reward, alive))
        observation = next_obs
        environment.render()
        sleep(1)