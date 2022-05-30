import gym
import numpy as np
from environment import Pacman

from time import sleep

if __name__ == '__main__':
    n_agents = 4
    environment = Pacman(n_agents - 1)
    pos, ghost_pos, pac_pos = environment.reset()
    print('positions: {} ghost position: {} pacman position: {}'.format(pos, ghost_pos, pac_pos))
    
    sleep(1)
    alive = True
    while alive:
        pos, reward, alive = environment.step([np.random.randint(4) for _ in range(n_agents)])
        print('positions: {} reward: {} pacman alive?: {}'.format(pos, reward, alive))
        environment.render()
        sleep(1)