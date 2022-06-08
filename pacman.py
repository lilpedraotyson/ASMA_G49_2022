from agent import Agent
import numpy as np

N_ACTIONS = 4
DOWN, LEFT, UP, RIGHT = range(N_ACTIONS)

class Pacman(Agent):
    def __init__(self):
        super(Pacman, self).__init__(f"Pacman")
        self.n_actions = N_ACTIONS
        self.position = [16, 11]
        self.alive = True
        self.orientation = 4
        self.observation = None

    def see(self, observation: np.ndarray):
        self.observation = observation
        
    def reset_position(self):
        self.position = [16, 11]
        self.alive = True

    def set_position(self, line, column):
        self.position = [line, column]

    def is_alive(self):
        return self.alive

    def kill(self):
        self.alive = False

    def action(self, step) -> int:
        if (step == 1):
            self.orientation = np.random.randint(4)

        return self.orientation