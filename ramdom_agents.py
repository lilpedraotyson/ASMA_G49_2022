import gym
import numpy as np
import matplotlib.pyplot as plt
from time import sleep

from environment import Environment

def showResults(agent, n_steps):
    fig, ax = plt.subplots()
    x = np.arange(len(agent))
    rects1 = ax.bar(x, n_steps, 0.35, alpha=0.5, label='average step counter', align='center')

    ax.set_ylabel('Number of Steps')
    ax.set_title('Number of steps per agent')
    ax.set_xticks(x, agent)
    ax.legend()

    ax.bar_label(rects1, padding=3)

    fig.tight_layout()

    plt.show()

def run_multiple_agents(environment, n_agents, episodes, alive=True):
    observation = environment.reset()
    print('ghost position: {} pacman position: {}'.format(observation[:(n_agents - 1) * 2], observation[(n_agents - 1) * 2:]))
    
    #sleep(0.5)
    meanaux = []
    for _ in range(episodes):
        print("----------------------------------------------------------------------------------------------------")
        environment.reset()
        alive = environment.pacman.alive
        while alive:
            for i in range(n_agents - 1):
                environment.ghosts[i].see(observation)
                environment.pacman.see(observation)
            next_obs, alive = environment.step([environment.ghosts[i].action(environment.map) for i in range(n_agents - 1)] + [environment.pacman.action(environment.step_count)])
            print('positions: {} pacman alive?: {}'.format(next_obs, alive))
            observation = next_obs
            environment.render()
            #sleep(0.5)
        meanaux.append(environment.step_count)
    print(meanaux)
    return sum(meanaux) / len(meanaux)

if __name__ == '__main__':
    n_agents = 4
    environment = Environment(n_agents - 1)
    episodes = 10

    mean = run_multiple_agents(environment, n_agents, episodes)
    print(mean)
    showResults(["agent"], [mean])